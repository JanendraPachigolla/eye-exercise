# Eye Exercise

React + TypeScript + Vite app for webcam-based eye exercise tracking and guidance.

## Project Structure

- `src/` application code
- `dist/` production build output

## Run Locally

```bash
npm install
npm run dev
```

Open the app from the Vite URL (usually `http://localhost:5173`).

## Quality Checks

```bash
npm run lint
npm run build
```

## Deploy To GitHub Pages

This project is configured for GitHub Pages hosting.

```bash
npm run deploy
```

The command builds the app and publishes `dist/` to the `gh-pages` branch.

For automatic deploys with GitHub Actions:

- Go to repository **Settings > Pages**
- Set **Source** to **GitHub Actions**
- Push to `main` and the workflow in `.github/workflows/deploy-pages.yml` will deploy the site
