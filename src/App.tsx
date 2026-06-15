import { useEffect, useRef, useState } from "react";
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

type GuideState = "move_left" | "move_right" | "move_up" | "move_down" | "hold_steady" | "no_face";

type IrisPayload = {
  left: { x: number; y: number };
  right: { x: number; y: number };
  leftRadius: number;
  rightRadius: number;
};

type IrisPoint = { x: number; y: number; t: number };

type MotionAssessment = {
  circular: boolean;
  loops: number;
  smoothness: number;
  radiusConsistency: number;
  direction: "clockwise" | "counterclockwise" | "unclear";
};

type ExerciseSummary = {
  durationSec: number;
  completedAtIso: string;
  assessment: MotionAssessment;
  gazeAtCameraCount: number;
};

const guideText: Record<GuideState, string> = {
  move_left: "Move slightly left",
  move_right: "Move slightly right",
  move_up: "Move slightly up",
  move_down: "Move slightly down",
  hold_steady: "Perfect. Hold steady.",
  no_face: "Align your face inside the boundary",
};

const DEFAULT_ASSESSMENT: MotionAssessment = {
  circular: false,
  loops: 0,
  smoothness: 0,
  radiusConsistency: 0,
  direction: "unclear",
};

const LEFT_IRIS_CENTER_INDEX = 468;
const LEFT_IRIS_RING_INDEXES = [469, 470, 471, 472] as const;
const RIGHT_IRIS_CENTER_INDEX = 473;
const RIGHT_IRIS_RING_INDEXES = [474, 475, 476, 477] as const;
const DURATIONS = [
  { value: 15, label: "15s" },
  { value: 60, label: "1m" },
  { value: 120, label: "2m" },
  { value: 300, label: "5m" },
] as const;
const DETECTION_INTERVAL_MS = 220;

function getLandmarkPoint(
  landmarks: Array<{ x: number; y: number }>,
  index: number,
  width: number,
  height: number
): { x: number; y: number } {
  const point = landmarks[index];
  if (!point) {
    return { x: 0, y: 0 };
  }
  return { x: point.x * width, y: point.y * height };
}

function estimateIrisRadius(
  landmarks: Array<{ x: number; y: number }>,
  center: { x: number; y: number },
  ringIndexes: readonly number[],
  width: number,
  height: number
): number {
  const ringPoints = ringIndexes.map((index) => landmarks[index]).filter(Boolean);
  if (!ringPoints.length) {
    return 6;
  }
  const avgDistance =
    ringPoints.reduce((sum, point) => {
      const x = point.x * width;
      const y = point.y * height;
      return sum + Math.hypot(x - center.x, y - center.y);
    }, 0) / ringPoints.length;
  return Math.max(4, Math.min(12, avgDistance));
}

function playTimerDoneSound() {
  if (typeof window === "undefined") {
    return;
  }
  const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  if (!AudioCtx) {
    return;
  }

  const audioContext = new AudioCtx();
  const now = audioContext.currentTime;
  const masterGain = audioContext.createGain();
  masterGain.connect(audioContext.destination);
  masterGain.gain.setValueAtTime(0.0001, now);

  const alarmPulse = (baseFrequency: number, start: number, duration: number) => {
    const oscA = audioContext.createOscillator();
    const oscB = audioContext.createOscillator();
    const pulseGain = audioContext.createGain();

    // Slightly detuned dual oscillators produce a more alarm-like tone.
    oscA.type = "square";
    oscB.type = "square";
    oscA.frequency.setValueAtTime(baseFrequency, start);
    oscB.frequency.setValueAtTime(baseFrequency * 1.015, start);

    pulseGain.gain.setValueAtTime(0.0001, start);
    pulseGain.gain.exponentialRampToValueAtTime(0.18, start + 0.02);
    pulseGain.gain.exponentialRampToValueAtTime(0.0001, start + duration);

    oscA.connect(pulseGain);
    oscB.connect(pulseGain);
    pulseGain.connect(masterGain);

    oscA.start(start);
    oscB.start(start);
    oscA.stop(start + duration);
    oscB.stop(start + duration);
  };

  // Alarm cadence: alternating high/low pulses repeated quickly.
  const pulseDuration = 0.18;
  const pulseGap = 0.08;
  const totalPulses = 14;
  for (let i = 0; i < totalPulses; i += 1) {
    const start = now + i * (pulseDuration + pulseGap);
    const frequency = i % 2 === 0 ? 1320 : 880;
    alarmPulse(frequency, start, pulseDuration);
  }

  const totalDurationMs = Math.ceil((totalPulses * (pulseDuration + pulseGap) + 0.4) * 1000);
  window.setTimeout(() => {
    void audioContext.close();
  }, totalDurationMs);
}

function notifyTimerDone() {
  playTimerDoneSound();
  if (typeof navigator !== "undefined" && "vibrate" in navigator) {
    navigator.vibrate?.([200, 120, 260]);
  }
}

function isStraightGaze(
  face: Array<{ x: number; y: number }>,
  leftIris: { x: number; y: number },
  rightIris: { x: number; y: number },
  width: number,
  height: number
): boolean {
  const leftOuter = getLandmarkPoint(face, 33, width, height);
  const leftInner = getLandmarkPoint(face, 133, width, height);
  const leftTop = getLandmarkPoint(face, 159, width, height);
  const leftBottom = getLandmarkPoint(face, 145, width, height);

  const rightInner = getLandmarkPoint(face, 362, width, height);
  const rightOuter = getLandmarkPoint(face, 263, width, height);
  const rightTop = getLandmarkPoint(face, 386, width, height);
  const rightBottom = getLandmarkPoint(face, 374, width, height);

  const leftCenter = { x: (leftOuter.x + leftInner.x) / 2, y: (leftTop.y + leftBottom.y) / 2 };
  const rightCenter = { x: (rightInner.x + rightOuter.x) / 2, y: (rightTop.y + rightBottom.y) / 2 };

  const leftHalfW = Math.max(1, Math.abs(leftInner.x - leftOuter.x) / 2);
  const rightHalfW = Math.max(1, Math.abs(rightOuter.x - rightInner.x) / 2);
  const leftHalfH = Math.max(1, Math.abs(leftBottom.y - leftTop.y) / 2);
  const rightHalfH = Math.max(1, Math.abs(rightBottom.y - rightTop.y) / 2);

  const leftDx = Math.abs((leftIris.x - leftCenter.x) / leftHalfW);
  const leftDy = Math.abs((leftIris.y - leftCenter.y) / leftHalfH);
  const rightDx = Math.abs((rightIris.x - rightCenter.x) / rightHalfW);
  const rightDy = Math.abs((rightIris.y - rightCenter.y) / rightHalfH);

  return leftDx <= 0.4 && leftDy <= 0.5 && rightDx <= 0.4 && rightDy <= 0.5;
}

function assessCircularMotion(points: IrisPoint[]): MotionAssessment {
  if (points.length < 12) {
    return DEFAULT_ASSESSMENT;
  }

  const center = points.reduce(
    (acc, point) => ({ x: acc.x + point.x / points.length, y: acc.y + point.y / points.length }),
    { x: 0, y: 0 }
  );

  const radii = points.map((point) => Math.hypot(point.x - center.x, point.y - center.y));
  const meanRadius = radii.reduce((sum, radius) => sum + radius, 0) / radii.length;
  const radiusVariance = radii.reduce((sum, radius) => sum + (radius - meanRadius) ** 2, 0) / radii.length;
  const radiusStdDev = Math.sqrt(radiusVariance);
  const radiusCv = meanRadius > 0 ? radiusStdDev / meanRadius : 1;
  const radiusConsistency = Math.max(0, 1 - radiusCv);

  const angles = points.map((point) => Math.atan2(point.y - center.y, point.x - center.x));
  let signedAngleTotal = 0;
  let absAngleTotal = 0;
  let turningChanges = 0;
  let previousDelta = 0;
  for (let index = 1; index < angles.length; index += 1) {
    let delta = angles[index] - angles[index - 1];
    while (delta > Math.PI) delta -= 2 * Math.PI;
    while (delta < -Math.PI) delta += 2 * Math.PI;
    signedAngleTotal += delta;
    absAngleTotal += Math.abs(delta);

    if (index > 1 && Math.sign(delta) !== Math.sign(previousDelta) && Math.abs(delta) > 0.02) {
      turningChanges += 1;
    }
    previousDelta = delta;
  }

  const loops = absAngleTotal / (2 * Math.PI);
  const smoothness = Math.max(0, 1 - turningChanges / Math.max(1, points.length - 2));
  const direction =
    Math.abs(signedAngleTotal) < Math.PI / 2
      ? "unclear"
      : signedAngleTotal > 0
        ? "counterclockwise"
        : "clockwise";

  const circular = loops >= 0.75 && smoothness >= 0.55 && radiusConsistency >= 0.45;

  return {
    circular,
    loops: Number(loops.toFixed(2)),
    smoothness: Number((smoothness * 100).toFixed(1)),
    radiusConsistency: Number((radiusConsistency * 100).toFixed(1)),
    direction,
  };
}

function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const loopRef = useRef<number | null>(null);
  const loopTimeoutRef = useRef<number | null>(null);
  const landmarkerRef = useRef<FaceLandmarker | null>(null);
  const isExercisingRef = useRef(false);
  const irisTrackRef = useRef<IrisPoint[]>([]);
  const latestIrisRef = useRef<IrisPayload | null>(null);
  const smoothedIrisRef = useRef<IrisPayload | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const recordingChunksRef = useRef<Blob[]>([]);
  const countdownIntervalRef = useRef<number | null>(null);
  const wasExercisingRef = useRef(false);
  const exerciseStartedRef = useRef(false);
  const selectedDurationRef = useRef(60);
  const remainingSecRef = useRef(60);
  const motionAssessmentRef = useRef<MotionAssessment>(DEFAULT_ASSESSMENT);
  const gazeAtCameraCountRef = useRef(0);
  const wasGazingAtCameraRef = useRef(false);

  const [guideState, setGuideState] = useState<GuideState>("no_face");
  const [iris, setIris] = useState<IrisPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isExercising, setIsExercising] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);
  const [motionAssessment, setMotionAssessment] = useState<MotionAssessment>(DEFAULT_ASSESSMENT);
  const [recordedClipUrl, setRecordedClipUrl] = useState<string | null>(null);
  const [selectedDurationSec, setSelectedDurationSec] = useState(60);
  const [remainingSec, setRemainingSec] = useState(60);
  const [exerciseSummary, setExerciseSummary] = useState<ExerciseSummary | null>(null);
  const [alignmentError, setAlignmentError] = useState<string | null>(null);
  const [completionNotice, setCompletionNotice] = useState<string | null>(null);
  const [mirrorMode, setMirrorMode] = useState(false);
  const toggleExercise = () => setIsExercising((prev) => !prev);

  useEffect(() => {
    selectedDurationRef.current = selectedDurationSec;
  }, [selectedDurationSec]);

  useEffect(() => {
    remainingSecRef.current = remainingSec;
  }, [remainingSec]);

  useEffect(() => {
    motionAssessmentRef.current = motionAssessment;
  }, [motionAssessment]);

  const beginTimedExercise = () => {
    if (exerciseStartedRef.current) {
      return;
    }
    exerciseStartedRef.current = true;
    setAlignmentError(null);

    if (countdownIntervalRef.current) {
      window.clearInterval(countdownIntervalRef.current);
    }
    countdownIntervalRef.current = window.setInterval(() => {
      setRemainingSec((prev) => {
        if (prev <= 1) {
          if (countdownIntervalRef.current) {
            window.clearInterval(countdownIntervalRef.current);
            countdownIntervalRef.current = null;
          }
          notifyTimerDone();
          setCompletionNotice("Exercise complete.");
          window.setTimeout(() => setCompletionNotice(null), 3500);
          setIsExercising(false);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    if (videoRef.current && typeof MediaRecorder !== "undefined") {
      try {
        const videoElement = videoRef.current as HTMLVideoElement & {
          captureStream?: () => MediaStream;
          webkitCaptureStream?: () => MediaStream;
        };
        const recordStream = videoElement.captureStream?.() ?? videoElement.webkitCaptureStream?.();
        if (!recordStream) {
          throw new Error("Capture stream unavailable");
        }
        const mimeType = MediaRecorder.isTypeSupported("video/webm;codecs=vp8") ? "video/webm;codecs=vp8" : "video/webm";
        recordingChunksRef.current = [];
        const recorder = new MediaRecorder(recordStream, { mimeType });
        recorder.ondataavailable = (event) => {
          if (event.data.size > 0) {
            recordingChunksRef.current.push(event.data);
          }
        };
        recorder.onstop = () => {
          if (recordingChunksRef.current.length === 0) {
            return;
          }
          const clipBlob = new Blob(recordingChunksRef.current, { type: "video/webm" });
          setRecordedClipUrl(URL.createObjectURL(clipBlob));
        };
        recorder.start(200);
        recorderRef.current = recorder;
      } catch {
        setError("Recording is not supported in this browser.");
      }
    }
  };

  useEffect(() => {
    let mounted = true;

    const start = async () => {
      try {
        if (!navigator.mediaDevices || typeof navigator.mediaDevices.getUserMedia !== "function") {
          const insecureContextMessage =
            "Camera API is unavailable in this browser/context. On mobile, open the app over HTTPS or from localhost on the same device.";
          const genericMessage =
            "Camera API is unavailable in this browser/context. Use a modern browser and allow camera permissions.";
          setError(window.isSecureContext ? genericMessage : insecureContextMessage);
          return;
        }

        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
          audio: false,
        });
        streamRef.current = stream;

        if (!videoRef.current) return;
        const video = videoRef.current;
        if (video.srcObject !== stream) {
          video.srcObject = stream;
        }
        await new Promise<void>((resolve) => {
          if (video.readyState >= HTMLMediaElement.HAVE_METADATA) {
            resolve();
            return;
          }
          const onLoaded = () => {
            video.removeEventListener("loadedmetadata", onLoaded);
            resolve();
          };
          video.addEventListener("loadedmetadata", onLoaded);
        });
        try {
          await video.play();
        } catch (playError) {
          // Browser may interrupt play() during rapid reload/src swaps.
          if (!(playError instanceof DOMException && playError.name === "AbortError")) {
            throw playError;
          }
        }

        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );
        landmarkerRef.current = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
          },
          runningMode: "VIDEO",
          numFaces: 1,
        });
        setCameraReady(true);

        const detectAndSend = () => {
          if (!mounted || !videoRef.current || !overlayRef.current || !landmarkerRef.current) {
            return;
          }

          const video = videoRef.current;
          const overlay = overlayRef.current;
          if (video.videoWidth === 0 || video.videoHeight === 0) {
            loopRef.current = window.requestAnimationFrame(detectAndSend);
            return;
          }

          overlay.width = video.videoWidth;
          overlay.height = video.videoHeight;

          const ctx = overlay.getContext("2d");
          if (!ctx) {
            loopRef.current = window.requestAnimationFrame(detectAndSend);
            return;
          }

          ctx.clearRect(0, 0, overlay.width, overlay.height);

          const boundary = {
            x: overlay.width * 0.25,
            y: overlay.height * 0.2,
            w: overlay.width * 0.5,
            h: overlay.height * 0.6,
          };

          ctx.strokeStyle = "#22c55e";
          ctx.lineWidth = 1;
          ctx.strokeRect(boundary.x, boundary.y, boundary.w, boundary.h);

          const latestIris = latestIrisRef.current;
          if (latestIris) {
            ctx.strokeStyle = "#22d3ee";
            ctx.fillStyle = "rgba(34, 211, 238, 0.35)";
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(latestIris.left.x, latestIris.left.y, latestIris.leftRadius * 0.8, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
            ctx.beginPath();
            ctx.arc(latestIris.right.x, latestIris.right.y, latestIris.rightRadius * 0.8, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();

            ctx.fillStyle = "#22d3ee";
            ctx.beginPath();
            ctx.arc(latestIris.left.x, latestIris.left.y, 2, 0, Math.PI * 2);
            ctx.fill();
            ctx.beginPath();
            ctx.arc(latestIris.right.x, latestIris.right.y, 2, 0, Math.PI * 2);
            ctx.fill();

            ctx.font = "12px sans-serif";
            ctx.fillText("L", latestIris.left.x + 8, latestIris.left.y - 8);
            ctx.fillText("R", latestIris.right.x + 8, latestIris.right.y - 8);
          }

          const result = landmarkerRef.current.detectForVideo(video, performance.now());
          const face = result.faceLandmarks?.[0];

          if (!face) {
            setGuideState("no_face");
            setAlignmentError("Align your face inside the boundary.");
            loopRef.current = window.requestAnimationFrame(detectAndSend);
            return;
          }

          const xs = face.map((point) => point.x * overlay.width);
          const ys = face.map((point) => point.y * overlay.height);
          const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
          const cy = (Math.min(...ys) + Math.max(...ys)) / 2;

          const withinBoundary =
            cx >= boundary.x &&
            cx <= boundary.x + boundary.w &&
            cy >= boundary.y &&
            cy <= boundary.y + boundary.h;

          if (cx < boundary.x) setGuideState("move_right");
          else if (cx > boundary.x + boundary.w) setGuideState("move_left");
          else if (cy < boundary.y) setGuideState("move_down");
          else if (cy > boundary.y + boundary.h) setGuideState("move_up");
          else setGuideState("hold_steady");

          const leftIris = getLandmarkPoint(face, LEFT_IRIS_CENTER_INDEX, overlay.width, overlay.height);
          const rightIris = getLandmarkPoint(face, RIGHT_IRIS_CENTER_INDEX, overlay.width, overlay.height);
          const leftRadius = estimateIrisRadius(
            face,
            leftIris,
            LEFT_IRIS_RING_INDEXES,
            overlay.width,
            overlay.height
          );
          const rightRadius = estimateIrisRadius(
            face,
            rightIris,
            RIGHT_IRIS_RING_INDEXES,
            overlay.width,
            overlay.height
          );
          const parsed: IrisPayload = {
            left: { x: Math.round(leftIris.x), y: Math.round(leftIris.y) },
            right: { x: Math.round(rightIris.x), y: Math.round(rightIris.y) },
            leftRadius,
            rightRadius,
          };

          const previous = smoothedIrisRef.current;
          const alpha = 0.35;
          const stabilized: IrisPayload = previous
            ? {
                left: {
                  x: Math.round(previous.left.x + alpha * (parsed.left.x - previous.left.x)),
                  y: Math.round(previous.left.y + alpha * (parsed.left.y - previous.left.y)),
                },
                right: {
                  x: Math.round(previous.right.x + alpha * (parsed.right.x - previous.right.x)),
                  y: Math.round(previous.right.y + alpha * (parsed.right.y - previous.right.y)),
                },
                leftRadius: previous.leftRadius + alpha * (parsed.leftRadius - previous.leftRadius),
                rightRadius: previous.rightRadius + alpha * (parsed.rightRadius - previous.rightRadius),
              }
            : parsed;

          smoothedIrisRef.current = stabilized;
          setIris(stabilized);
          latestIrisRef.current = stabilized;

          const straight = isStraightGaze(face, stabilized.left, stabilized.right, overlay.width, overlay.height);

          if (!withinBoundary) {
            setAlignmentError("Center your face inside the boundary.");
          } else if (!straight && !exerciseStartedRef.current) {
            // Only warn about iris direction before exercise starts (during exercise, eye movement is expected)
            setAlignmentError(isExercisingRef.current ? "Iris are not straight. Look directly at the camera to begin." : "Iris are not straight. Look directly at the camera.");
          } else if (exerciseStartedRef.current) {
            // Exercise running — clear alignment errors; eye movement away from camera is correct
            setAlignmentError(null);
          } else {
            setAlignmentError(null);
          }

          if (isExercisingRef.current) {
            if (straight && !exerciseStartedRef.current) {
              beginTimedExercise();
            }

            if (!exerciseStartedRef.current) {
              loopTimeoutRef.current = window.setTimeout(() => {
                loopRef.current = window.requestAnimationFrame(detectAndSend);
              }, DETECTION_INTERVAL_MS) as unknown as number;
              return;
            }

            // Count distinct camera-gaze events during exercise
            if (straight && !wasGazingAtCameraRef.current) {
              gazeAtCameraCountRef.current += 1;
              wasGazingAtCameraRef.current = true;
            } else if (!straight) {
              wasGazingAtCameraRef.current = false;
            }

            const centerX = (stabilized.left.x + stabilized.right.x) / 2;
            const centerY = (stabilized.left.y + stabilized.right.y) / 2;
            const nextPoints = [...irisTrackRef.current, { x: centerX, y: centerY, t: Date.now() }].slice(-140);
            irisTrackRef.current = nextPoints;
            setMotionAssessment(assessCircularMotion(nextPoints));
          }

          loopTimeoutRef.current = window.setTimeout(() => {
            loopRef.current = window.requestAnimationFrame(detectAndSend);
          }, DETECTION_INTERVAL_MS) as unknown as number;
        };

        loopRef.current = window.requestAnimationFrame(detectAndSend);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to initialize camera");
      }
    };

    void start();

    return () => {
      mounted = false;
      if (loopRef.current) window.cancelAnimationFrame(loopRef.current);
      if (loopTimeoutRef.current) window.clearTimeout(loopTimeoutRef.current);
      if (countdownIntervalRef.current) window.clearInterval(countdownIntervalRef.current);
      streamRef.current?.getTracks().forEach((track) => track.stop());
      landmarkerRef.current?.close();
      if (recorderRef.current && recorderRef.current.state !== "inactive") {
        recorderRef.current.stop();
      }
    };
  }, []);

  useEffect(
    () => () => {
      if (recordedClipUrl) {
        URL.revokeObjectURL(recordedClipUrl);
      }
    },
    [recordedClipUrl]
  );

  useEffect(() => {
    // Remaining time mirrors selected duration while idle.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setRemainingSec(selectedDurationSec);
  }, [selectedDurationSec]);

  /* eslint-disable react-hooks/set-state-in-effect */
  useEffect(() => {
    const wasExercising = wasExercisingRef.current;
    isExercisingRef.current = isExercising;
    if (isExercising) {
      wasExercisingRef.current = true;
      exerciseStartedRef.current = false;
      setError(null);
      setAlignmentError("Look straight at the camera to start.");
      irisTrackRef.current = [];
      gazeAtCameraCountRef.current = 0;
      wasGazingAtCameraRef.current = false;
      setMotionAssessment(DEFAULT_ASSESSMENT);
      setExerciseSummary(null);
      setRemainingSec(selectedDurationRef.current);
      if (recordedClipUrl) {
        URL.revokeObjectURL(recordedClipUrl);
        setRecordedClipUrl(null);
      }

      if (countdownIntervalRef.current) {
        window.clearInterval(countdownIntervalRef.current);
      }

      return;
    }

    if (!isExercising && wasExercising) {
      if (countdownIntervalRef.current) {
        window.clearInterval(countdownIntervalRef.current);
        countdownIntervalRef.current = null;
      }
      irisTrackRef.current = [];
      smoothedIrisRef.current = null;
      setExerciseSummary({
        durationSec: selectedDurationRef.current - remainingSecRef.current,
        completedAtIso: new Date().toISOString(),
        assessment: motionAssessmentRef.current,
        gazeAtCameraCount: gazeAtCameraCountRef.current,
      });
      exerciseStartedRef.current = false;
      wasExercisingRef.current = false;
      setAlignmentError(null);
    }
    if (!isExercising && recorderRef.current && recorderRef.current.state !== "inactive") {
      recorderRef.current.stop();
    }
  }, [isExercising, recordedClipUrl]);
  /* eslint-enable react-hooks/set-state-in-effect */

  const guideBarMod = alignmentError
    ? "warn"
    : guideState === "hold_steady"
      ? "steady"
      : guideState === "no_face"
        ? "no-face"
        : "warn";

  const progressPct =
    isExercising && selectedDurationSec > 0
      ? ((selectedDurationSec - remainingSec) / selectedDurationSec) * 100
      : 0;

  return (
    <main className="page">
      <header className="app-header">
        <h1 className="app-title">Eye Exercise</h1>
        <span className={`tracking-badge${cameraReady ? " tracking-badge--active" : ""}`}>
          {cameraReady ? "● Active" : "○ Starting…"}
        </span>
      </header>

      {/* Real errors only (e.g. camera permission denied) */}
      {error && (
        <div className="alert alert--error" role="alert">
          <span className="alert-icon">⚠</span>
          {error}
        </div>
      )}
      {completionNotice && (
        <div className="alert alert--success" role="alert">
          <span className="alert-icon">✓</span>
          {completionNotice}
        </div>
      )}

      <div className={`timer-card${isExercising ? " timer-card--active" : ""}`}>
        <div className="timer-display">
          <span>{Math.floor(remainingSec / 60)}</span>
          <span className={`timer-colon${isExercising ? " timer-colon--pulse" : ""}`}>:</span>
          <span>{String(remainingSec % 60).padStart(2, "0")}</span>
        </div>
        <div className="timer-label">{isExercising ? "remaining" : "set time"}</div>
      </div>

      <div className={`duration-selector${isExercising ? " duration-selector--hidden" : ""}`}>
        {DURATIONS.map(({ value, label }) => (
          <label
            key={value}
            className={[
              "duration-pill",
              selectedDurationSec === value ? "duration-pill--active" : "",
              isExercising ? "duration-pill--disabled" : "",
            ]
              .filter(Boolean)
              .join(" ")}
          >
            <input
              type="radio"
              name="exercise-duration"
              value={value}
              checked={selectedDurationSec === value}
              onChange={() => setSelectedDurationSec(value)}
              disabled={isExercising}
            />
            {label}
          </label>
        ))}
      </div>

      <button
        type="button"
        className={`exercise-btn exercise-btn--inline${isExercising ? " exercise-btn--stop" : " exercise-btn--start"}`}
        onClick={toggleExercise}
        disabled={!cameraReady}
      >
        {isExercising ? "⏹ Stop Exercise" : "▶ Start Exercise"}
      </button>

      <section className="panels">
        <div className="panel">
          <div className={`guide-bar guide-bar--${guideBarMod}`}>
            {alignmentError ?? guideText[guideState]}
          </div>
          <div className="stack">
            <video
              ref={videoRef}
              autoPlay
              muted
              playsInline
              className={`video${mirrorMode ? " video--mirror" : ""}`}
            />
            <canvas ref={overlayRef} className="overlay" />
            <button
              type="button"
              className="mirror-btn"
            onClick={() => setMirrorMode((prev) => !prev)}
              title="Toggle mirror"
              aria-label="Toggle mirror view"
            >
              ⟺
            </button>
          </div>
          {iris ? (
            <details className="iris-details">
              <summary>Iris coordinates</summary>
              <p className="coords">
                Left: ({iris.left.x}, {iris.left.y}) | Right: ({iris.right.x}, {iris.right.y})
              </p>
            </details>
          ) : (
            <p className="coords">Waiting for iris detection…</p>
          )}
        </div>

        <div className="panel">
          <h2 className="panel-title">Exercise Summary</h2>
          {!exerciseSummary && !isExercising && (
            <p className="coords">Start an exercise to view summary and recording.</p>
          )}
          {recordedClipUrl && (
            <>
              <h3 className="panel-subtitle">Recording</h3>
              <video src={recordedClipUrl} controls className="preview" />
            </>
          )}
          {exerciseSummary && !isExercising && (
            <div className="summary-grid">
              <div className="summary-card">
                <span className="summary-label">Duration</span>
                <span className="summary-value">{exerciseSummary.durationSec}s</span>
              </div>
              <div className="summary-card">
                <span className="summary-label">Completed</span>
                <span className="summary-value">
                  {new Date(exerciseSummary.completedAtIso).toLocaleTimeString()}
                </span>
              </div>
              <div className="summary-card">
                <span className="summary-label">Circular</span>
                <span
                  className={`summary-value${exerciseSummary.assessment.circular ? " summary-value--good" : " summary-value--bad"}`}
                >
                  {exerciseSummary.assessment.circular ? "YES" : "NO"}
                </span>
              </div>
              <div className="summary-card">
                <span className="summary-label">Loops</span>
                <span className="summary-value">{exerciseSummary.assessment.loops}</span>
              </div>
              <div className="summary-card">
                <span className="summary-label">Direction</span>
                <span className="summary-value">{exerciseSummary.assessment.direction}</span>
              </div>
              <div className="summary-card">
                <span className="summary-label">Smoothness</span>
                <span className="summary-value">{exerciseSummary.assessment.smoothness}%</span>
              </div>
              <div className="summary-card">
                <span className="summary-label">Radius Consistency</span>
                <span className="summary-value">{exerciseSummary.assessment.radiusConsistency}%</span>
              </div>
              <div className="summary-card">
                <span className="summary-label">Camera Gazes</span>
                <span className={`summary-value${exerciseSummary.gazeAtCameraCount === 0 ? " summary-value--good" : " summary-value--bad"}`}>
                  {exerciseSummary.gazeAtCameraCount}×
                </span>
              </div>
            </div>
          )}
        </div>
      </section>

      {/* Sticky bottom bar — mobile only */}
      <div
        className={[
          "sticky-bar",
          isExercising ? "sticky-bar--active" : "",
          completionNotice ? "sticky-bar--complete" : "",
        ]
          .filter(Boolean)
          .join(" ")}
      >
        {/* Progress strip */}
        <div className="sticky-bar__progress">
          <div className="sticky-bar__progress-fill" style={{ width: `${progressPct}%` }} />
        </div>
        <div className="sticky-bar__controls">
          <div className="sticky-bar__timer">
            <span>{Math.floor(remainingSec / 60)}</span>
            <span className={`timer-colon${isExercising ? " timer-colon--pulse" : ""}`}>:</span>
            <span>{String(remainingSec % 60).padStart(2, "0")}</span>
          </div>
          <button
            type="button"
            className={`exercise-btn exercise-btn--sticky${isExercising ? " exercise-btn--stop" : " exercise-btn--start"}`}
            onClick={toggleExercise}
            disabled={!cameraReady}
          >
            {isExercising ? "⏹ Stop" : "▶ Start"}
          </button>
        </div>
      </div>
    </main>
  );
}

export default App;
