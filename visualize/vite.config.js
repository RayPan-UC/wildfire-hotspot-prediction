import { defineConfig } from 'vite';

export default defineConfig({
  build: {
    outDir: 'dist',
    rollupOptions: {
      input: {
        main:       'index.html',
        fire_growth: 'fire_growth.html',
      },
    },
  },
  server: {
    port: 5173,
    proxy: {
      // During `npm run dev`, proxy /data/ to the Python server
      '/data': {
        target: 'http://localhost:8765',
        changeOrigin: true,
      },
    },
  },
});
