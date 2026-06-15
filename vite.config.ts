import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import basicSsl from '@vitejs/plugin-basic-ssl'

// https://vite.dev/config/
export default defineConfig({
  // Relative asset paths keep the build portable for GitHub Pages project sites.
  base: './',
  plugins: [react(), basicSsl()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    https: {},
    proxy: {
      '/ws/iris': {
        target: 'ws://127.0.0.1:8000',
        ws: true,
      },
    },
  },
})
