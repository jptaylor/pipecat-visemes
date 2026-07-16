import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // WebRTC signaling goes to the Pipecat dev runner; media flows P2P.
    proxy: {
      '/api': 'http://localhost:7860',
    },
  },
})
