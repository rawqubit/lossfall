// Loads index.html in headless chromium, lets it train, and asserts the page
// actually did something: steps advanced, loss fell, text appeared, the
// attention canvas rendered. Skips cleanly when no chromium is installed.
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { existsSync } from 'node:fs';
import test from 'node:test';
import assert from 'node:assert/strict';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..');
const page = 'file://' + join(root, 'index.html');

const hasChrome = ['/usr/bin/chromium', '/usr/bin/chromium-browser', '/usr/bin/google-chrome'].some((p) => existsSync(p)) || process.env.CI === 'true';

test('the page trains, writes, and draws in a real browser', { skip: !hasChrome && 'no chromium on this machine' }, () => {
  const raw = execFileSync(process.execPath, [join(here, 'cdp.mjs'), page, '4000', '/tmp/lossfall-page.png', '1280', 'none'], {
    encoding: 'utf8',
    timeout: 120000,
  });
  const report = JSON.parse(raw);
  assert.deepEqual(report.errors, [], 'the page logged errors');
  const { stats } = report;
  const steps = Number(stats.steps.replace(/,/g, ''));
  assert.ok(steps > 30, `only ${steps} steps ran`);
  assert.ok(Number(stats.loss) < 3.4, `loss did not fall: ${stats.loss}`);
  assert.ok(stats.sampleLen > 20, 'nothing was sampled');
  assert.ok(stats.attentionLit > 100, 'the attention canvas drew nothing');
  assert.equal(stats.scrollWidth, stats.clientWidth, 'the page overflows horizontally');
});
