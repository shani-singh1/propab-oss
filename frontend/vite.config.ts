import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      "/research": "http://127.0.0.1:8000",
      "/stream": "http://127.0.0.1:8000",
      "/sessions": "http://127.0.0.1:8000",
      "/health": "http://127.0.0.1:8000",
      "/tools": "http://127.0.0.1:8000",
    },
  },
});
