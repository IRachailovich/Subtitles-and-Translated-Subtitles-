import { cp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(fileURLToPath(new URL('.', import.meta.url)), '..');
const source = resolve(root, 'cloud_web');
const output = resolve(root, 'dist-cloud-web');
const apiBaseUrl = String(process.env.SUBGEN_API_BASE_URL || '').replace(/\/$/, '');

if (!apiBaseUrl.startsWith('https://')) {
  throw new Error('SUBGEN_API_BASE_URL must be the HTTPS URL of the deployed Cloud Run API.');
}

await rm(output, { recursive: true, force: true });
await mkdir(output, { recursive: true });
await cp(source, output, { recursive: true, force: true });
await mkdir(resolve(output, 'assets'), { recursive: true });
await cp(resolve(root, 'web', 'assets', 'app-icon.png'), resolve(output, 'assets', 'app-icon.png'));
await writeFile(
  resolve(output, 'runtime-config.js'),
  `window.SUBGEN_RUNTIME_CONFIG = Object.freeze(${JSON.stringify({ apiBaseUrl })});\n`,
  'utf8',
);

const serviceWorkerPath = resolve(output, 'service-worker.js');
const serviceWorker = await readFile(serviceWorkerPath, 'utf8');
await writeFile(serviceWorkerPath, serviceWorker.replace(
  "const CACHE = 'subgen-cloud-v2';",
  `const CACHE = 'subgen-cloud-v2-${Date.now()}';`,
), 'utf8');
