/**
 * The browser half of the parity harness: drive `parity.html` headlessly.
 *
 * Serves the *built* static site over plain HTTP (not the dev server -- the
 * point is to measure what ships), opens the parity page in headless Chromium,
 * and runs every fixture through `window.__parity`.
 *
 * Usage:
 *   node parity/run_browser.mjs --dist web/dist --base /age-estimator/ \
 *        --cases parity/fixtures/cases.json --out /tmp/browser.json
 *
 * Playwright is not a dependency of the app and is installed on demand
 * (`npx playwright install chromium`); the harness says so rather than failing
 * with a module-resolution error.
 */

import { createReadStream } from 'node:fs';
import { readFile, stat, writeFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { extname, join, normalize, resolve } from 'node:path';
import { parseArgs } from 'node:util';

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.wasm': 'application/wasm',
  '.onnx': 'application/octet-stream',
  '.map': 'application/json; charset=utf-8',
  '.png': 'image/png',
};

/**
 * A static file server that mimics GitHub Pages closely enough to matter.
 *
 * Specifically: it sends `Content-Length` (so the download progress bar has
 * something real to show) and does NOT send `Cross-Origin-Opener-Policy` or
 * `Cross-Origin-Embedder-Policy`. Pages cannot send those either, so
 * `SharedArrayBuffer` is unavailable here exactly as it is in production. If
 * the harness quietly enabled cross-origin isolation, it would be testing a
 * threaded WASM path that the real deployment can never take.
 */
function serve(dist, base) {
  const root = resolve(dist);
  const prefix = base.endsWith('/') ? base : `${base}/`;
  return createServer(async (req, res) => {
    try {
      let pathname = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
      if (prefix !== '/' && pathname.startsWith(prefix)) {
        pathname = `/${pathname.slice(prefix.length)}`;
      }
      if (pathname.endsWith('/')) pathname += 'index.html';
      const filePath = join(root, normalize(pathname).replace(/^(\.\.[/\\])+/, ''));
      const info = await stat(filePath);
      if (!info.isFile()) throw new Error('not a file');
      res.writeHead(200, {
        'content-type': MIME[extname(filePath)] ?? 'application/octet-stream',
        'content-length': info.size,
        'cache-control': 'no-cache',
      });
      createReadStream(filePath).pipe(res);
    } catch {
      res.writeHead(404, { 'content-type': 'text/plain' });
      res.end('not found');
    }
  });
}

function listen(server) {
  return new Promise((resolvePort) => {
    server.listen(0, '127.0.0.1', () => resolvePort(server.address().port));
  });
}

async function main() {
  const { values } = parseArgs({
    options: {
      dist: { type: 'string', default: 'web/dist' },
      base: { type: 'string', default: '/' },
      cases: { type: 'string', default: 'parity/fixtures/cases.json' },
      out: { type: 'string', default: 'parity/browser.json' },
      url: { type: 'string' },
      throttle: { type: 'string' },
    },
  });

  let chromium;
  try {
    ({ chromium } = await import('playwright'));
  } catch {
    console.error(
      'playwright is not installed. Run:\n' +
        '  cd web && npm install --no-save playwright && npx playwright install chromium',
    );
    process.exit(2);
  }

  const spec = JSON.parse(await readFile(values.cases, 'utf8'));
  const fixturesDir = resolve(values.cases, '..');

  let server = null;
  let pageUrl = values.url;
  if (!pageUrl) {
    server = serve(values.dist, values.base);
    const port = await listen(server);
    const prefix = values.base.endsWith('/') ? values.base : `${values.base}/`;
    pageUrl = `http://127.0.0.1:${port}${prefix}parity.html`;
  }

  const browser = await chromium.launch();
  const page = await browser.newPage();
  const consoleErrors = [];
  page.on('console', (msg) => {
    if (msg.type() === 'error') consoleErrors.push(msg.text());
  });
  page.on('pageerror', (err) => consoleErrors.push(String(err)));

  console.log(`opening ${pageUrl}`);
  await page.goto(pageUrl, { waitUntil: 'load' });
  await page.waitForFunction(() => window.__parityReady === true, null, { timeout: 60_000 });

  // Prove the single-threaded WASM assumption rather than assuming it. If
  // cross-origin isolation somehow WERE on, onnxruntime-web could take a
  // threaded path that GitHub Pages can never offer, and this harness would be
  // measuring something the deployment cannot do.
  const environment = await page.evaluate(() => ({
    crossOriginIsolated: globalThis.crossOriginIsolated === true,
    hasSharedArrayBuffer: typeof SharedArrayBuffer !== 'undefined',
    hardwareConcurrency: navigator.hardwareConcurrency,
  }));
  console.log('  environment:', JSON.stringify(environment));

  const results = [];
  for (const testCase of spec.cases) {
    const bytes = await readFile(join(fixturesDir, testCase.image));
    const dataUrl = `data:image/png;base64,${bytes.toString('base64')}`;
    const started = Date.now();
    const response = await page.evaluate(
      (request) => window.__parity(request),
      {
        dataUrl,
        model: testCase.model,
        ...(testCase.crop_margin === undefined ? {} : { cropMargin: testCase.crop_margin }),
        ...(testCase.boxes === undefined ? {} : { boxes: testCase.boxes }),
      },
    );
    results.push({ id: testCase.id, model: testCase.model, ...response });
    console.log(
      `  ${testCase.id}: ${response.faces.length} face(s) ` +
        `(detect ${response.timings.detectMs.toFixed(1)}ms, ` +
        `infer ${response.timings.inferMs.toFixed(1)}ms, wall ${Date.now() - started}ms)`,
    );
  }

  // Cold-load timing for a first-time visitor, measured last so the warm run
  // above cannot have primed it.
  const cold = await measureColdLoad(
    browser,
    pageUrl,
    join(fixturesDir, spec.cases[0].image),
    spec.cases[0].model,
    values.throttle ?? null,
  );
  console.log(
    `  cold load: nav ${cold.navigationMs}ms, first result ${cold.firstResultMs}ms, ` +
      `${(cold.transferredBytes / 1048576).toFixed(2)} MB transferred`,
  );

  await browser.close();
  if (server) server.close();

  await writeFile(
    values.out,
    `${JSON.stringify({ source: 'browser', environment, cold, results }, null, 2)}\n`,
  );
  console.log(`wrote ${values.out}`);

  if (consoleErrors.length > 0) {
    console.error('browser console errors:');
    for (const line of consoleErrors) console.error(`  ${line}`);
    process.exit(1);
  }
}

/**
 * How long a first-time visitor actually waits.
 *
 * Measured in a fresh browser context, so an empty HTTP cache: the 6.2 MB model
 * is downloaded for real. `firstResultMs` is the number that matters -- time
 * from "page is interactive" to "an age is on screen" -- and it includes the
 * detector download, the model download, WASM instantiation and the first
 * inference, because a visitor waits for all of them.
 *
 * Reporting `DOMContentLoaded` instead would be flattering and useless: the
 * page is interactive within a few hundred milliseconds and then does nothing
 * visible until the model lands.
 *
 * `--throttle` applies a network profile through CDP, because an unthrottled
 * number measured from a datacentre or over loopback says nothing about the
 * experience this UX was designed around. The download is ~9 MB compressed and
 * the whole point of the progress bar is the case where that takes a while.
 */
const THROTTLE_PROFILES = {
  // Roughly the "Fast 4G" and "Slow 4G" presets in Chrome DevTools, plus a
  // deliberately unkind one. Throughput is bytes/second.
  '4g': { downloadThroughput: (10 * 1024 * 1024) / 8, uploadThroughput: (3 * 1024 * 1024) / 8, latency: 40 },
  'slow-4g': { downloadThroughput: (1.6 * 1024 * 1024) / 8, uploadThroughput: (750 * 1024) / 8, latency: 150 },
  '3g': { downloadThroughput: (400 * 1024) / 8, uploadThroughput: (400 * 1024) / 8, latency: 400 },
};

async function measureColdLoad(browser, pageUrl, imagePath, model, throttle) {
  const context = await browser.newContext();
  const page = await context.newPage();

  if (throttle) {
    const profile = THROTTLE_PROFILES[throttle];
    if (!profile) {
      throw new Error(
        `unknown --throttle ${throttle}; known: ${Object.keys(THROTTLE_PROFILES).join(', ')}`,
      );
    }
    const session = await context.newCDPSession(page);
    await session.send('Network.enable');
    await session.send('Network.emulateNetworkConditions', { offline: false, ...profile });
  }

  const transferred = { total: 0, byType: {} };
  page.on('response', (response) => {
    const length = Number(response.headers()['content-length'] ?? 0);
    if (!Number.isFinite(length) || length <= 0) return;
    transferred.total += length;
    const key = new URL(response.url()).pathname.split('.').pop() ?? 'other';
    transferred.byType[key] = (transferred.byType[key] ?? 0) + length;
  });

  const navStart = Date.now();
  await page.goto(pageUrl, { waitUntil: 'load' });
  await page.waitForFunction(() => window.__parityReady === true, null, { timeout: 60_000 });
  const navigationMs = Date.now() - navStart;

  const bytes = await readFile(imagePath);
  const dataUrl = `data:image/png;base64,${bytes.toString('base64')}`;
  const firstStart = Date.now();
  const response = await page.evaluate((request) => window.__parity(request), { dataUrl, model });
  const firstResultMs = Date.now() - firstStart;

  await context.close();
  return {
    throttle: throttle ?? 'none',
    navigationMs,
    firstResultMs,
    detectMs: response.timings.detectMs,
    inferMs: response.timings.inferMs,
    transferredBytes: transferred.total,
    byType: transferred.byType,
  };
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
