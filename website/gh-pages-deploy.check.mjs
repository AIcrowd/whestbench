import test from 'node:test';
import assert from 'node:assert/strict';
import { access, readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const websiteRoot = path.dirname(fileURLToPath(import.meta.url));

test('build-generated llms artifacts exist and are non-empty', async () => {
  for (const rel of ['public/llms.txt', 'public/llms-full.txt', 'out/llms.txt', 'out/.nojekyll']) {
    await access(path.join(websiteRoot, rel));
  }
  const llms = await readFile(path.join(websiteRoot, 'public', 'llms.txt'), 'utf8');
  assert.match(llms, /^# whestbench/m);
  assert.ok(llms.length > 100);
});

test('docs route avoids runtime filesystem reads (static-export safe)', async () => {
  const src = await readFile(path.join(websiteRoot, 'app/docs/[[...slug]]/page.tsx'), 'utf8');
  assert.doesNotMatch(src, /node:fs|process\.cwd\(/);
});
