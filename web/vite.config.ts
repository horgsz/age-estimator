import { defineConfig } from 'vite';
import { resolve } from 'node:path';

/**
 * `base` decides whether the deployed assets resolve at all.
 *
 * GitHub Pages serves this as a PROJECT site, at
 * `https://horgsz.github.io/age-estimator/`, not at the domain root. Vite's
 * default `base` of `/` would emit `/assets/index-*.js`, which 404s there.
 * `VITE_BASE` is set by `.github/workflows/pages.yml`; locally it stays `/` so
 * the dev server and `make dev` are unaffected.
 *
 * The browser engine reads the same value back at runtime via
 * `import.meta.env.BASE_URL`, which is how it finds `models/` and `ort/`.
 */
const base = process.env.VITE_BASE ?? '/';

export default defineConfig({
  base,
  server: {
    port: 5173,
    strictPort: true,
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        // The parity page. Shipped with the site on purpose: it is how the
        // deployed build -- not a local approximation of it -- is checked
        // against the Python path, and it is the page the CI harness drives.
        parity: resolve(__dirname, 'parity.html'),
      },
    },
  },
  optimizeDeps: {
    // onnxruntime-web ships prebuilt ESM with its own WASM loader; letting
    // esbuild pre-bundle it can rewrite the import.meta URLs it uses to find
    // the .wasm files.
    exclude: ['onnxruntime-web'],
  },
});
