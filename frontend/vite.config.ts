import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev-server proxy: the frontend calls relative /api/* paths, proxied to
// the backend so there's no CORS friction locally. In the container, nginx
// (see nginx.conf) performs the same /api -> backend:8000 proxy.
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.VITE_API_PROXY_TARGET || "http://localhost:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
