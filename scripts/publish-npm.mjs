import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';

const { workspaces } = JSON.parse(readFileSync('package.json', 'utf8'));

for (const workspace of workspaces) {
  const { name, version, private: isPrivate } = JSON.parse(
    readFileSync(`${workspace}/package.json`, 'utf8'),
  );
  if (isPrivate) continue;

  const release = `${name}@${version}`;
  try {
    execFileSync('npm', ['view', release, 'version', '--json'], { encoding: 'utf8' });
    console.log(`Skipping ${release}: already published.`);
    continue;
  } catch (error) {
    let code;
    try {
      code = JSON.parse(error.stdout).error?.code;
    } catch {
      // Invalid registry responses must not be treated as missing releases.
    }
    if (code !== 'E404') throw error;
  }

  execFileSync('npm', ['publish', '--workspace', workspace, '--access', 'public'], {
    stdio: 'inherit',
  });
}
