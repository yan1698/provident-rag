import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const api = "http://127.0.0.1:8000";

export default defineConfig({
  base: "/ui/",
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/query": api,
      "/ingest": api,
      "/documents": api,
      "/library": api,
      "/collections": api,
      "/embed": api,
      "/metrics": api,
      "/traces": api,
      "/tags": api,
      "/eval": api,
      "/agent": api,
      "/feedback": api,
      "/health": api,
      "/search": api,
      "/verify": api,
      "/auth": api,
      "/graph": api,
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    minify: false,
    sourcemap: false,
    rollupOptions: {
      output: {
        entryFileNames: "assets/app.js",
        chunkFileNames: "assets/[name].js",
        assetFileNames: "assets/[name][extname]",
      },
    },
  },
});
