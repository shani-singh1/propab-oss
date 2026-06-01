import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The API base is configurable via VITE_API_BASE (defaults to the local stack).
// We also proxy /api -> backend in dev so EventSource/fetch can use same-origin paths.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.VITE_API_TARGET || "http://localhost:8000",
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ""),
      },
    },
  },
});
