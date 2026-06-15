# Eye Exercise - Phase 1 Scaffold

This phase includes:
- React frontend with webcam boundary guidance prompts.
- FastAPI backend with websocket iris detection pipeline.
- Backend iris preview rendered in frontend.

## Project structure
- `frontend/` React + Vite app
- `backend/` FastAPI service

## Run backend
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Run frontend
```bash
cd frontend
npm install
npm run dev
```

Open frontend from Vite URL (usually `http://localhost:5173`).
