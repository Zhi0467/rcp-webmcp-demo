import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8421",
    },
  },
  build: {
    outDir: "dist",
    // A running source-mode window may still lazy-load a chunk from the previous
    // build. Keep content-hashed assets until the server is restarted; packaging
    // uses `build:clean` so stale chunks never ship.
    emptyOutDir: false,
  },
});
