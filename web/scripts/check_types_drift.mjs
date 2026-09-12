/**
 * Cross-platform script to verify that generated TypeScript types match OpenAPI schema without drift.
 * Works natively on Windows, macOS, and Linux without depending on Unix `diff` or `rm`.
 */
import fs from 'node:fs';
import path from 'node:path';
import { execSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const webDir = path.resolve(__dirname, '..');

const openapiPath = path.join(webDir, 'openapi.json');
const currentTypesPath = path.join(webDir, 'src', 'api', 'types.generated.ts');
const tempTypesPath = path.join(webDir, 'src', 'api', 'types.temp.ts');

if (!fs.existsSync(openapiPath)) {
  console.error(`❌ openapi.json not found at ${openapiPath}. Run export_openapi.py first.`);
  process.exit(1);
}

try {
  // 1. Generate temp types from openapi.json
  execSync(`npx openapi-typescript "${openapiPath}" -o "${tempTypesPath}"`, {
    cwd: webDir,
    stdio: 'inherit',
  });

  if (!fs.existsSync(currentTypesPath)) {
    console.error(`❌ Current types file not found at ${currentTypesPath}. Run "npm run gen:types" to generate.`);
    if (fs.existsSync(tempTypesPath)) fs.unlinkSync(tempTypesPath);
    process.exit(1);
  }

  // 2. Read and compare
  const currentContent = fs.readFileSync(currentTypesPath, 'utf-8').trim();
  const tempContent = fs.readFileSync(tempTypesPath, 'utf-8').trim();

  // Cleanup temp file
  fs.unlinkSync(tempTypesPath);

  if (currentContent !== tempContent) {
    console.error('❌ TypeScript types are out of sync with FastAPI OpenAPI schema!');
    console.error('👉 Run "npm run gen:types" and commit the changes to src/api/types.generated.ts');
    process.exit(1);
  }

  console.log('✅ TypeScript types are perfectly up to date with OpenAPI schema.');
  process.exit(0);
} catch (err) {
  if (fs.existsSync(tempTypesPath)) {
    fs.unlinkSync(tempTypesPath);
  }
  console.error('❌ Error checking type drift:', err.message);
  process.exit(1);
}
