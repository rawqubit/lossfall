// Drive headless chromium over CDP: load a page, let it run, read the DOM,
// take a screenshot. Usage: node cdp.mjs <url> <waitMs> <outPng> [width] [full]
import { spawn, execFileSync } from 'node:child_process';
import { writeFileSync } from 'node:fs';

const [url, waitMs = '6000', outPng = '/tmp/shot.png', width = '1280', mode = 'full'] = process.argv.slice(2);
const port = 9333 + Math.floor(Math.random() * 200);

const BIN = [
  process.env.CHROME_BIN,
  'chromium', 'chromium-browser', 'google-chrome', 'google-chrome-stable',
  '/usr/bin/google-chrome-stable'
].filter(Boolean).find((bin) => {
  try { execFileSync(bin, ['--version'], { stdio: 'ignore' }); return true; } catch { return false; }
});
if (!BIN) { console.error('no chromium binary found'); process.exit(3); }

const chrome = spawn(BIN, [
  '--headless=new', '--no-sandbox', '--disable-gpu', '--hide-scrollbars',
  `--remote-debugging-port=${port}`, `--window-size=${width},1200`,
  '--user-data-dir=/tmp/cdp-profile-' + port, 'about:blank'
], { stdio: 'ignore' });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function target() {
  for (let i = 0; i < 60; i++) {
    try {
      const res = await fetch(`http://127.0.0.1:${port}/json`);
      const list = await res.json();
      const page = list.find((t) => t.type === 'page');
      if (page) return page;
    } catch {}
    await sleep(200);
  }
  throw new Error('no devtools target');
}

const page = await target();
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((r, j) => { ws.onopen = r; ws.onerror = j; });

let id = 0;
const pending = new Map();
ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  if (msg.id && pending.has(msg.id)) { pending.get(msg.id)(msg); pending.delete(msg.id); }
};
function send(method, params = {}) {
  id += 1;
  ws.send(JSON.stringify({ id, method, params }));
  return new Promise((resolve) => pending.set(id, resolve));
}

const errors = [];
await send('Runtime.enable');
await send('Log.enable');
await send('Page.enable');
ws.addEventListener('message', (event) => {
  const msg = JSON.parse(event.data);
  if (msg.method === 'Runtime.exceptionThrown') errors.push(msg.params.exceptionDetails.text + ' ' + (msg.params.exceptionDetails.exception?.description || ''));
  if (msg.method === 'Log.entryAdded' && msg.params.entry.level === 'error') errors.push(msg.params.entry.text);
});

await send('Page.navigate', { url });
await sleep(Number(waitMs));

const probe = `(() => {
  const el = (id) => document.getElementById(id);
  const canvas = el('attn');
  let lit = 0, total = 0;
  try {
    const px = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
    for (let i = 0; i < px.length; i += 4) { total++; if (px[i] > 40) lit++; }
  } catch (e) { lit = -1; }
  return JSON.stringify({
    steps: el('steps')?.textContent,
    tokens: el('tokens')?.textContent,
    loss: el('loss')?.firstChild?.nodeValue,
    delta: el('lossDelta')?.textContent,
    rate: el('rate')?.textContent,
    params: el('params')?.textContent,
    runtime: el('runtime')?.textContent,
    sampleLen: el('sample')?.textContent.length,
    sample: el('sample')?.textContent,
    before: el('before')?.textContent.slice(0, 60),
    attentionLit: lit,
    attentionTotal: total,
    scrollWidth: document.documentElement.scrollWidth,
    clientWidth: document.documentElement.clientWidth,
    chartNodes: el('chart')?.childNodes.length,
    bodyHeight: Math.round(document.body.getBoundingClientRect().height)
  });
})()`;
const result = await send('Runtime.evaluate', { expression: probe, returnByValue: true, awaitPromise: false });
const stats = JSON.parse(result.result?.result?.value ?? '{}');

let clip;
if (mode === 'full') {
  const metrics = await send('Page.getLayoutMetrics');
  const h = Math.ceil(metrics.result.cssContentSize.height);
  clip = { x: 0, y: 0, width: Number(width), height: h, scale: 1 };
  await send('Emulation.setDeviceMetricsOverride', { width: Number(width), height: h, deviceScaleFactor: 2, mobile: false });
  await sleep(600);
}
const shot = await send('Page.captureScreenshot', { format: 'png', clip, captureBeyondViewport: true });
if (shot.result?.data) writeFileSync(outPng, Buffer.from(shot.result.data, 'base64'));

console.log(JSON.stringify({ stats, errors, screenshot: outPng }, null, 1));
ws.close();
chrome.kill();
process.exit(0);
