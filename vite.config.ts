import { cpSync, existsSync } from 'node:fs';
import { defineConfig, Plugin } from 'vite';

/** 构建后把随包真实地形 data/ 拷入 dist/(开发模式由 Vite 直接从根目录提供) */
function copyBathyData(): Plugin {
  return {
    name: 'copy-bathy-data',
    apply: 'build',
    closeBundle() {
      if (existsSync('data')) {
        cpSync('data', 'dist/data', { recursive: true });
      }
    },
  };
}

export default defineConfig({
  plugins: [copyBathyData()],
});
