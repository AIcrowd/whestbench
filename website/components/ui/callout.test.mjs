import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const source = readFileSync(join(__dirname, 'callout.tsx'), 'utf8');

test('Callout component source exports the Callout function', () => {
  assert.match(source, /export function Callout\s*\(/);
});

test('Callout supports default and accent variants', () => {
  assert.match(source, /variant === 'default'/);
  assert.match(source, /variant === 'accent'/);
});

test('Callout applies spec-aligned tokens', () => {
  // Kicker tracking
  assert.match(source, /var\(--tracking-kicker-loose\)/);
  // Radius-xl container
  assert.match(source, /var\(--radius-xl\)/);
  // Coral-light accent ground
  assert.match(source, /var\(--coral-light\)/);
  // Gray-50 default ground
  assert.match(source, /var\(--gray-50\)/);
});
