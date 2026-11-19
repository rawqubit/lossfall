// Tests for the model. Run: node --test tests/
// engine.js is the model; build.py inlines it into index.html so the published
// page is one self-contained file.
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import assert from 'node:assert/strict';
import test from 'node:test';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..');
new Function(readFileSync(join(root, 'engine.js'), 'utf8'))();
const { Model, prepare, backward } = globalThis.Lossfall;

const corpus = readFileSync(join(root, 'corpus.txt'), 'utf8');
const data = prepare(corpus);

function tiny(overrides = {}) {
  return new Model({ vocab: data.vocab.length, ctx: 6, d: 8, heads: 2, layers: 1, batch: 1, seed: 42, ...overrides });
}

test('the corpus encodes to a small vocabulary', () => {
  assert.ok(data.vocab.length > 20 && data.vocab.length < 64);
  assert.ok(data.chars > 8000, `corpus encoded to ${data.chars} symbols`);
});

test('prepare lowercases and drops symbols it cannot represent', () => {
  const raw = 'Hello\tthere — world\ntoday';
  const cleaned = raw.toLowerCase().replace(/[^a-z0-9 .,;:'"!?\-\n]/g, '');
  const p = prepare(raw);
  assert.equal(p.chars, cleaned.length);
  assert.ok(!p.vocab.includes('\t') && !p.vocab.includes('—'));
  assert.deepEqual(p.vocab, [...new Set(cleaned)].sort());
});

// The test that matters: the gradients this file computes by hand have to match
// what a finite difference of the loss says they are.
test('every hand-written backward pass matches a finite difference', () => {
  const model = tiny();
  const seq = model.cfg.ctx;
  const ids = Array.from({ length: seq }, (_, i) => i % data.vocab.length);
  const targets = Array.from({ length: seq }, (_, i) => (i + 1) % data.vocab.length);
  const keep = new Uint8Array(seq).fill(1);
  const loss = () => model.forward(ids, targets, keep, 1, seq, false).loss.d[0];
  const eps = 1e-5;
  let checked = 0;
  let worst = 0;

  for (const param of model.params) {
    const out = model.forward(ids, targets, keep, 1, seq, false);
    for (const p of model.params) p.g.fill(0);
    backward(out.loss);
    const analytic = Array.from(param.g);

    for (const idx of [0, Math.floor(param.d.length / 3), param.d.length - 1]) {
      if (idx >= param.d.length) continue;
      const original = param.d[idx];
      param.d[idx] = original + eps;
      const high = loss();
      param.d[idx] = original - eps;
      const low = loss();
      param.d[idx] = original;
      const numeric = (high - low) / (2 * eps);
      const scale = Math.max(1e-3, Math.abs(numeric), Math.abs(analytic[idx]));
      const error = Math.abs(numeric - analytic[idx]) / scale;
      worst = Math.max(worst, error);
      assert.ok(error < 2e-3, `gradient mismatch: analytic ${analytic[idx]} vs numeric ${numeric}`);
      checked++;
    }
  }

  assert.ok(checked >= 30, `checked only ${checked} parameters`);
  assert.ok(worst < 2e-3, `worst relative error ${worst}`);
});

test('training lowers the loss', () => {
  const model = tiny({ ctx: 16, d: 32, heads: 2, layers: 1, batch: 4, lr: 3e-3 });
  const losses = [];
  for (let i = 0; i < 40; i++) losses.push(model.trainStep(data.ids));
  const first = losses.slice(0, 4).reduce((a, b) => a + b, 0) / 4;
  const last = losses.slice(-4).reduce((a, b) => a + b, 0) / 4;
  assert.ok(last < first - 0.15, `loss should fall: ${first.toFixed(3)} -> ${last.toFixed(3)}`);
});

test('sampling returns the requested number of known symbols', () => {
  const model = tiny();
  const prime = [0, 1, 2];
  const text = model.sample(40, 0.8, prime, data.vocab);
  assert.equal(text.length, 40);
  for (const ch of text) assert.ok(data.vocab.includes(ch), `unexpected symbol ${JSON.stringify(ch)}`);
});

test('sampling is deterministic for a fixed seed', () => {
  const prime = [1, 2, 3];
  assert.equal(tiny().sample(30, 0.9, prime, data.vocab), tiny().sample(30, 0.9, prime, data.vocab));
});

test('the shared vocabulary is sorted so indexes are stable', () => {
  assert.deepEqual(data.vocab, [...data.vocab].sort());
});

test('training counts steps and tokens, and the parameter count is real', () => {
  const model = tiny();
  assert.equal(model.size(), model.params.reduce((n, p) => n + p.d.length, 0));
  model.trainStep(data.ids);
  assert.equal(model.step, 1);
  assert.equal(model.tokens, model.cfg.ctx * model.cfg.batch);
});
