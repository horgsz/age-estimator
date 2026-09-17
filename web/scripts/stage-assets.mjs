/**
 * Stage everything the browser build needs into `web/public/`.
 *
 * Three kinds of asset end up there, none of which live in `web/` normally:
 *
 *   models/*.onnx   the two age models, copied from the repo's `checkpoints/`
 *                   (which is read-only and outside Vite's root)
 *   models/*.onnx   the YuNet face detector, downloaded and checksummed the
 *                   same way `server/detector.py` does it
 *   ort/*.wasm      onnxruntime-web's WASM binaries, which must be served from
 *                   a path the app controls so `ort.env.wasm.wasmPaths` can
 *                   point at it under the Pages base path
 *
 * `web/public/` is gitignored: these are build inputs copied from elsewhere,
 * and committing a second copy of a 6.2 MB model is how the two get out of sync.
 * `models/models.json` is the exception -- it IS committed, because generating
 * it requires torch (see `server/tools/export_static_registry.py`) and CI should
 * not install a deep-learning stack to publish a static site. A server test
 * regenerates it and fails if it has drifted.
 */

import { createHash } from 'node:crypto';
import { copyFile, mkdir, readFile, stat, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const WEB_ROOT = resolve(HERE, '..');
const REPO_ROOT = resolve(WEB_ROOT, '..');
const PUBLIC_DIR = join(WEB_ROOT, 'public');
const MODELS_OUT = join(PUBLIC_DIR, 'models');
const ORT_OUT = join(PUBLIC_DIR, 'ort');

const YUNET_FILENAME = 'face_detection_yunet_2023mar.onnx';
const YUNET_DYNAMIC_FILENAME = 'face_detection_yunet_2023mar.dynamic.onnx';
/** sha256 of the git-lfs object referenced by opencv_zoo@main. */
const YUNET_SHA256 = '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4';
const ZOO_DIR = 'opencv/opencv_zoo/main/models/face_detection_yunet/';
const YUNET_URLS = [
  // opencv_zoo keeps the ONNX files in git-lfs, so the plain raw.githubusercontent
  // URL returns a 131-byte pointer file. This host serves the real object.
  `https://media.githubusercontent.com/media/${ZOO_DIR}${YUNET_FILENAME}`,
  `https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/${YUNET_FILENAME}`,
];

/**
 * What onnxruntime-web fetches at runtime once `ort.env.wasm.wasmPaths` points
 * here: the Emscripten glue module and the WASM binary it instantiates.
 *
 * The `.jsep.*` pair is deliberately NOT staged. It is the WebGPU/WebNN build,
 * it is 21 MB, and this app pins the execution provider to `wasm`, so shipping
 * it would double the deployment for a code path that can never run.
 *
 * "threaded" in the filename is a build-time property of the binary, not a
 * promise to use threads: the same module runs single-threaded when
 * `numThreads` is 1, which is what `src/browser/ort.ts` sets, because GitHub
 * Pages cannot send the COOP/COEP headers `SharedArrayBuffer` requires.
 */
const ORT_WASM_FILES = ['ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm'];

async function exists(path) {
  try {
    await stat(path);
    return true;
  } catch {
    return false;
  }
}

function sha256(buffer) {
  return createHash('sha256').update(buffer).digest('hex');
}

async function stageAgeModels() {
  const registryPath = join(MODELS_OUT, 'models.json');
  if (!(await exists(registryPath))) {
    throw new Error(
      `${registryPath} is missing. Generate it with:\n` +
        '  .venv/bin/python -m server.tools.export_static_registry ' +
        '--out web/public/models/models.json',
    );
  }
  const registry = JSON.parse(await readFile(registryPath, 'utf8'));

  for (const model of registry.models) {
    if (!model.asset) continue;
    const src = join(REPO_ROOT, 'checkpoints', model.asset);
    const dst = join(MODELS_OUT, model.asset);
    if (!(await exists(src))) {
      throw new Error(`${src} is missing; cannot stage model "${model.key}".`);
    }
    await copyFile(src, dst);

    // The registry's figures are keyed by this digest. Copying the wrong file
    // here would hand one model's accuracy numbers to another's weights, which
    // is the exact failure the digest keying exists to prevent -- so it is
    // checked at build time rather than trusted.
    const digest = sha256(await readFile(dst)).slice(0, 12);
    if (digest !== model.sha256) {
      throw new Error(
        `Digest mismatch staging "${model.key}": ${model.asset} hashes to ${digest}, ` +
          `but models.json expects ${model.sha256}. Regenerate models.json with ` +
          'server/tools/export_static_registry.py.',
      );
    }
    console.log(`  models/${model.asset} (${(await stat(dst)).size} bytes, sha256 ${digest})`);
  }
}

async function stageYunet() {
  const dst = join(MODELS_OUT, YUNET_FILENAME);
  if (await exists(dst)) {
    const buffer = await readFile(dst);
    if (sha256(buffer) === YUNET_SHA256) {
      console.log(`  models/${YUNET_FILENAME} (cached)`);
      return;
    }
  }

  // Reuse the copy the local server already downloaded, if there is one.
  const serverCopy = join(REPO_ROOT, 'server', 'models', YUNET_FILENAME);
  if (await exists(serverCopy)) {
    const buffer = await readFile(serverCopy);
    if (sha256(buffer) === YUNET_SHA256) {
      await writeFile(dst, buffer);
      console.log(`  models/${YUNET_FILENAME} (from server/models)`);
      return;
    }
  }

  let lastError;
  for (const url of YUNET_URLS) {
    try {
      console.log(`  downloading ${url}`);
      const response = await fetch(url);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const buffer = Buffer.from(await response.arrayBuffer());
      const digest = sha256(buffer);
      if (digest !== YUNET_SHA256) {
        throw new Error(`checksum mismatch: expected ${YUNET_SHA256}, got ${digest}`);
      }
      await writeFile(dst, buffer);
      console.log(`  models/${YUNET_FILENAME} (${buffer.length} bytes)`);
      return;
    } catch (err) {
      lastError = err;
      console.warn(`  failed: ${err.message}`);
    }
  }
  throw new Error(
    `Could not obtain ${YUNET_FILENAME}. Place it in ${MODELS_OUT} manually. ` +
      `Last error: ${lastError?.message}`,
  );
}

/**
 * Make YuNet's input shape dynamic.
 *
 * The published ONNX declares `input: [1, 3, 640, 640]` as fixed dimensions.
 * OpenCV never notices, because `cv::dnn` re-infers every shape itself and
 * ignores what the file claims -- which is how `cv2.FaceDetectorYN` runs the
 * same weights at 400x512 or 1024x768. onnxruntime does NOT ignore it: it
 * validates the input against the declaration and refuses anything that is not
 * literally 640x640.
 *
 * Padding every frame out to 640x640 to satisfy it was the alternative, and it
 * was rejected: it would change the detector's input relative to the Python
 * path, and the boxes with it, which is exactly the divergence running the same
 * weights was supposed to eliminate.
 *
 * So the declaration is corrected instead. Both dimensions are rewritten from
 * `dim_value: 640` to a symbolic `dim_param`, which is what the model should
 * have declared: YuNet is fully convolutional and its own graph reshapes from
 * the runtime shape, as the unpatched outputs at other sizes demonstrate.
 *
 * The edit is length-preserving by construction. In protobuf a
 * `TensorShapeProto.Dimension` holding 640 is `0A 03 08 80 05` (field 1,
 * 3 bytes, dim_value varint 640); a one-character `dim_param` is
 * `0A 03 12 01 <char>` -- also 5 bytes. No parent message length changes, so
 * nothing else in the file has to be understood, let alone rewritten. The
 * assertions below pin that down: exactly two matches, identical file length,
 * and exactly six differing bytes -- three per dimension, since the two leading
 * framing bytes are shared. Anything else and the upstream file has
 * changed shape and this function must be re-examined rather than trusted.
 *
 * The weights are untouched, and the parity harness proves it end to end: the
 * boxes this produces are compared against `cv2.FaceDetectorYN`'s and must
 * match exactly.
 */
async function stageDynamicYunet() {
  const src = join(MODELS_OUT, YUNET_FILENAME);
  const original = await readFile(src);
  const patched = Buffer.from(original);

  // `dim_value: 640`, as a TensorShapeProto.Dimension submessage.
  const fixed = Buffer.from([0x0a, 0x03, 0x08, 0x80, 0x05]);
  const offsets = [];
  for (let at = patched.indexOf(fixed); at !== -1; at = patched.indexOf(fixed, at + 1)) {
    offsets.push(at);
  }
  if (offsets.length !== 2) {
    throw new Error(
      `Expected exactly 2 fixed 640 dimensions in ${YUNET_FILENAME}, found ${offsets.length}. ` +
        'The upstream model has changed; re-check stageDynamicYunet().',
    );
  }

  // `dim_param: "H"` / `dim_param: "W"` -- same five bytes.
  offsets.forEach((at, i) => {
    patched.set([0x0a, 0x03, 0x12, 0x01, i === 0 ? 0x48 : 0x57], at);
  });

  if (patched.length !== original.length) throw new Error('patch changed the file length');
  let changed = 0;
  for (let i = 0; i < original.length; i++) if (original[i] !== patched[i]) changed++;
  if (changed !== 6) {
    throw new Error(`patch touched ${changed} bytes, expected 6 (3 per dimension)`);
  }

  await writeFile(join(MODELS_OUT, YUNET_DYNAMIC_FILENAME), patched);
  console.log(`  models/${YUNET_DYNAMIC_FILENAME} (${patched.length} bytes, 6 bytes changed)`);
}

async function stageOrtRuntime() {
  const ortDist = join(WEB_ROOT, 'node_modules', 'onnxruntime-web', 'dist');
  for (const name of ORT_WASM_FILES) {
    const src = join(ortDist, name);
    if (!(await exists(src))) {
      console.warn(`  skipping ${name} (not in onnxruntime-web/dist)`);
      continue;
    }
    await copyFile(src, join(ORT_OUT, name));
    console.log(`  ort/${name} (${(await stat(src)).size} bytes)`);
  }
}

async function main() {
  await mkdir(MODELS_OUT, { recursive: true });
  await mkdir(ORT_OUT, { recursive: true });
  console.log('staging static assets into web/public/');
  await stageAgeModels();
  await stageYunet();
  await stageDynamicYunet();
  await stageOrtRuntime();
  console.log('done');
}

main().catch((err) => {
  console.error(`stage-assets: ${err.message}`);
  process.exit(1);
});
