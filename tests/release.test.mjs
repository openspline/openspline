import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { chmodSync, mkdtempSync, readFileSync, rmSync, writeFileSync, mkdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

const script = fileURLToPath(new URL('../scripts/publish-npm.mjs', import.meta.url));

function runRelease(t, responses) {
  const root = mkdtempSync(join(tmpdir(), 'openspline-release-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const workspaces = Object.keys(responses);
  writeFileSync(join(root, 'package.json'), JSON.stringify({ workspaces }));
  for (const workspace of workspaces) {
    mkdirSync(join(root, workspace));
    writeFileSync(join(root, workspace, 'package.json'), JSON.stringify({
      name: `@test/${workspace}`, version: '0.1.2',
    }));
  }
  const npm = join(root, 'npm');
  writeFileSync(npm, `#!${process.execPath}
const fs = require('node:fs');
const args = process.argv.slice(2);
if (args[0] === 'view') {
  const name = args[1].slice('@test/'.length).split('@')[0];
  const response = ${JSON.stringify(responses)}[name];
  if (response === 'published') {
    console.log(JSON.stringify('0.1.2'));
  } else {
    console.log(JSON.stringify({ error: { code: response } }));
    process.exitCode = 1;
  }
} else if (args[0] === 'publish') {
  fs.appendFileSync('published.jsonl', JSON.stringify(args) + '\\n');
} else {
  process.exitCode = 2;
}
`);
  chmodSync(npm, 0o755);
  const result = spawnSync(process.execPath, [script], {
    cwd: root, encoding: 'utf8', env: { ...process.env, PATH: `${root}:${process.env.PATH}` },
  });
  let published = [];
  try {
    published = readFileSync(join(root, 'published.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
  } catch (error) {
    if (error.code !== 'ENOENT') throw error;
  }
  return { ...result, published };
}

test('partial release retries publish only missing workspace versions', (t) => {
  const result = runRelease(t, { node: 'E404', browser: 'published', react: 'published' });
  assert.equal(result.status, 0, result.stderr);
  assert.deepEqual(result.published, [['publish', '--workspace', 'node', '--access', 'public']]);
});

test('an already published release succeeds without publishing again', (t) => {
  const result = runRelease(t, { node: 'published', browser: 'published' });
  assert.equal(result.status, 0, result.stderr);
  assert.deepEqual(result.published, []);
});

test('registry authentication and network failures stop publication', (t) => {
  for (const error of ['E401', 'E403', 'ETIMEDOUT']) {
    const result = runRelease(t, { node: error, browser: 'E404' });
    assert.notEqual(result.status, 0);
    assert.deepEqual(result.published, []);
  }
});
