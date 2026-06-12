import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import { alphaTab } from '@coderline/alphatab-vite';

export default defineConfig({
  plugins: [react(), alphaTab()],
  server: {
    port: 5173,
    proxy: {
      '/compose': 'http://localhost:8001',
      '/inpaint_preview': 'http://localhost:8001',
      '/commit_draft': 'http://localhost:8001',
      '/discard_draft': 'http://localhost:8001',
      '/alt_positions': 'http://localhost:8001',
      '/apply_fingering': 'http://localhost:8001',
      '/append_measures': 'http://localhost:8001',
      '/generate_measures': 'http://localhost:8001',
      '/api/convert-to-guitar': 'http://localhost:8001',
      '/health': 'http://localhost:8001',
    },
  },
  test: {
    include: ['src/**/*.test.ts', 'src/**/*.test.tsx'],
    environment: 'node',
  },
});
