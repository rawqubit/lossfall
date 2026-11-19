// Train from scratch and print what the model writes at a few checkpoints.
// Same config, same seed and same corpus as the page, so the numbers below are
// reproducible:  node examples/trace.mjs
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
new Function(readFileSync(join(root, 'engine.js'), 'utf8'))();
const { Model, prepare } = globalThis.Lossfall;

const data = prepare(readFileSync(join(root, 'corpus.txt'), 'utf8'));
const model = new Model({ vocab: data.vocab.length, heads: 4, ctx: 16, d: 32, layers: 2, batch: 8, lr: 3e-3, seed: 8675309 });
const marks = [0, 100, 250, 600, 1500, 3000, 6000, 9000];
const prime = [data.vocab.indexOf('\n')];
const rows = [];
let ema = null;

for (let step = 0; step <= Math.max(...marks); step++) {
  if (marks.includes(step)) {
    rows.push({ step, loss: ema, sample: model.sample(160, 0.8, prime, data.vocab) });
  }
  const loss = model.trainStep(data.ids);
  ema = ema === null ? loss : ema * 0.99 + loss * 0.01;
}

for (const row of rows) {
  const loss = row.loss === null ? '—' : row.loss.toFixed(2);
  console.log(`${String(row.step).padStart(4)}  ${loss}  ${JSON.stringify(row.sample)}`);
}