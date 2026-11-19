/* lossfall engine — a character-level transformer with a hand-written
   autograd. No dependencies, no DOM, no network. Exposes Lossfall on the
   global object so the same file can be tested under node. */
(function (root) {
  'use strict';

  var EPS = 1e-5;

  /* --- tensors and ops --------------------------------------------------
     Every tensor is 2-D: rows x cols, backed by one Float64Array. The
     backward pass accumulates into .g, so a node can fan out. */

  function T(data, rows, cols, parents, back) {
    this.d = data;
    this.r = rows;
    this.c = cols;
    this.parents = parents || null;
    this.back = back || null;
    this.g = new Float64Array(rows * cols);
  }

  function zeros(r, c) {
    return new T(new Float64Array(r * c), r, c);
  }

  function build(r, c, fn) {
    var t = zeros(r, c);
    for (var i = 0; i < r; i++) for (var j = 0; j < c; j++) t.d[i * c + j] = fn(i, j);
    return t;
  }

  function matmul(a, b) {
    var out = zeros(a.r, b.c);
    for (var i = 0; i < a.r; i++) {
      var ao = i * a.c, oo = i * out.c;
      for (var k = 0; k < a.c; k++) {
        var av = a.d[ao + k];
        if (av === 0) continue;
        var bo = k * b.c;
        for (var j = 0; j < b.c; j++) out.d[oo + j] += av * b.d[bo + j];
      }
    }
    out.parents = [a, b];
    out.back = function (g) {
      for (var i = 0; i < a.r; i++) for (var k = 0; k < a.c; k++) {
        var s = 0, oo = i * out.c, bo = k * b.c;
        for (var j = 0; j < b.c; j++) s += g[oo + j] * b.d[bo + j];
        a.g[i * a.c + k] += s;
      }
      for (var i2 = 0; i2 < a.r; i2++) for (var k2 = 0; k2 < a.c; k2++) {
        var av = a.d[i2 * a.c + k2];
        if (av === 0) continue;
        var oo2 = i2 * out.c, bo2 = k2 * b.c;
        for (var j2 = 0; j2 < b.c; j2++) b.g[bo2 + j2] += av * g[oo2 + j2];
      }
    };
    return out;
  }

  function transpose(x) {
    var out = build(x.c, x.r, function (i, j) { return x.d[j * x.c + i]; });
    out.parents = [x];
    out.back = function (g) {
      for (var i = 0; i < x.r; i++) for (var j = 0; j < x.c; j++) x.g[i * x.c + j] += g[j * x.c + i];
    };
    return out;
  }

  function add(a, b) {
    var out = zeros(a.r, a.c);
    for (var i = 0; i < out.d.length; i++) out.d[i] = a.d[i] + b.d[i];
    out.parents = [a, b];
    out.back = function (g) {
      for (var i = 0; i < g.length; i++) { a.g[i] += g[i]; b.g[i] += g[i]; }
    };
    return out;
  }

  function badd(x, row) {
    var out = zeros(x.r, x.c);
    for (var i = 0; i < x.r; i++) for (var j = 0; j < x.c; j++) out.d[i * x.c + j] = x.d[i * x.c + j] + row.d[j];
    out.parents = [x, row];
    out.back = function (g) {
      for (var i = 0; i < x.r; i++) for (var j = 0; j < x.c; j++) {
        x.g[i * x.c + j] += g[i * x.c + j];
        row.g[j] += g[i * x.c + j];
      }
    };
    return out;
  }

  function smul(x, s) {
    var out = zeros(x.r, x.c);
    for (var i = 0; i < x.d.length; i++) out.d[i] = x.d[i] * s;
    out.parents = [x];
    out.back = function (g) { for (var i = 0; i < g.length; i++) x.g[i] += g[i] * s; };
    return out;
  }

  function gelu(x) {
    var C = Math.sqrt(2 / Math.PI), K = 0.044715;
    var out = zeros(x.r, x.c);
    var th = new Float64Array(x.d.length);
    for (var i = 0; i < x.d.length; i++) {
      var v = x.d[i];
      var u = C * (v + K * v * v * v);
      th[i] = Math.tanh(u);
      out.d[i] = 0.5 * v * (1 + th[i]);
    }
    out.parents = [x];
    out.back = function (g) {
      for (var i = 0; i < g.length; i++) {
        var v = x.d[i];
        var du = C * (1 + 3 * K * v * v);
        var dtanh = 1 - th[i] * th[i];
        x.g[i] += g[i] * (0.5 * (1 + th[i]) + 0.5 * v * dtanh * du);
      }
    };
    return out;
  }

  function lnorm(x, gamma, beta) {
    var out = zeros(x.r, x.c);
    var xhat = new Float64Array(x.d.length);
    var inv = new Float64Array(x.r);
    for (var i = 0; i < x.r; i++) {
      var mean = 0;
      for (var j = 0; j < x.c; j++) mean += x.d[i * x.c + j];
      mean /= x.c;
      var v = 0;
      for (var j2 = 0; j2 < x.c; j2++) { var t = x.d[i * x.c + j2] - mean; v += t * t; }
      v /= x.c;
      var s = 1 / Math.sqrt(v + EPS);
      inv[i] = s;
      for (var j3 = 0; j3 < x.c; j3++) {
        var h = (x.d[i * x.c + j3] - mean) * s;
        xhat[i * x.c + j3] = h;
        out.d[i * x.c + j3] = h * gamma.d[j3] + beta.d[j3];
      }
    }
    out.parents = [x, gamma, beta];
    out.back = function (g) {
      for (var i = 0; i < x.r; i++) {
        var dsum = 0, dxh = 0;
        for (var j = 0; j < x.c; j++) {
          var dy = g[i * x.c + j];
          gamma.g[j] += dy * xhat[i * x.c + j];
          beta.g[j] += dy;
          var dxhj = dy * gamma.d[j];
          dsum += dxhj;
          dxh += dxhj * xhat[i * x.c + j];
        }
        for (var j2 = 0; j2 < x.c; j2++) {
          var dy2 = g[i * x.c + j2];
          var dxhj2 = dy2 * gamma.d[j2];
          x.g[i * x.c + j2] += inv[i] * (dxhj2 - dsum / x.c - xhat[i * x.c + j2] * dxh / x.c);
        }
      }
    };
    return out;
  }

  function gather(table, ids) {
    var out = zeros(ids.length, table.c);
    for (var i = 0; i < ids.length; i++) {
      var src = ids[i] * table.c, dst = i * table.c;
      for (var j = 0; j < table.c; j++) out.d[dst + j] = table.d[src + j];
    }
    out.parents = [table];
    out.back = function (g) {
      for (var i = 0; i < ids.length; i++) {
        var dst = ids[i] * table.c, src = i * table.c;
        for (var j = 0; j < table.c; j++) table.g[dst + j] += g[src + j];
      }
    };
    return out;
  }

  function sliceCols(x, start, width) {
    var out = build(x.r, width, function (i, j) { return x.d[i * x.c + start + j]; });
    out.parents = [x];
    out.back = function (g) {
      for (var i = 0; i < x.r; i++) for (var j = 0; j < width; j++) x.g[i * x.c + start + j] += g[i * width + j];
    };
    return out;
  }

  function concatCols(parts) {
    var rows = parts[0].r;
    var cols = 0;
    for (var p = 0; p < parts.length; p++) cols += parts[p].c;
    var out = zeros(rows, cols);
    var off = 0;
    for (var q = 0; q < parts.length; q++) {
      var part = parts[q];
      for (var i = 0; i < rows; i++) for (var j = 0; j < part.c; j++) out.d[i * cols + off + j] = part.d[i * part.c + j];
      off += part.c;
    }
    out.parents = parts;
    out.back = function (g) {
      var o = 0;
      for (var q2 = 0; q2 < parts.length; q2++) {
        var part2 = parts[q2];
        for (var i2 = 0; i2 < rows; i2++) for (var j2 = 0; j2 < part2.c; j2++) part2.g[i2 * part2.c + j2] += g[i2 * cols + o + j2];
        o += part2.c;
      }
    };
    return out;
  }

  /* masked softmax over each row, with the causal/sequence mask applied.
     Applied to scaled Q.K^T so the mask never produces NaN. */
  function attention(scores, mask) {
    var r = scores.r, c = scores.c;
    var probs = zeros(r, c);
    for (var i = 0; i < r; i++) {
      var max = -Infinity;
      for (var j = 0; j < c; j++) if (mask[i * c + j] && scores.d[i * c + j] > max) max = scores.d[i * c + j];
      var sum = 0;
      for (var j2 = 0; j2 < c; j2++) {
        var v = mask[i * c + j2] ? Math.exp(scores.d[i * c + j2] - max) : 0;
        probs.d[i * c + j2] = v;
        sum += v;
      }
      var inv = sum > 0 ? 1 / sum : 0;
      for (var j3 = 0; j3 < c; j3++) probs.d[i * c + j3] *= inv;
    }
    probs.parents = [scores];
    probs.back = function (g) {
      for (var i = 0; i < r; i++) {
        var dot = 0;
        for (var j = 0; j < c; j++) dot += g[i * c + j] * probs.d[i * c + j];
        for (var j2 = 0; j2 < c; j2++) scores.g[i * c + j2] += probs.d[i * c + j2] * (g[i * c + j2] - dot);
      }
    };
    return probs;
  }

  /* mean cross entropy over the rows we keep, with softmax folded in */
  function crossEntropy(logits, targets, keep) {
    var r = logits.r, c = logits.c;
    var probs = zeros(r, c);
    var loss = 0, n = 0;
    for (var i = 0; i < r; i++) {
      if (!keep[i]) continue;
      var max = -Infinity;
      for (var j = 0; j < c; j++) if (logits.d[i * c + j] > max) max = logits.d[i * c + j];
      var sum = 0;
      for (var j2 = 0; j2 < c; j2++) {
        var v = Math.exp(logits.d[i * c + j2] - max);
        probs.d[i * c + j2] = v;
        sum += v;
      }
      for (var j3 = 0; j3 < c; j3++) probs.d[i * c + j3] /= sum;
      loss += -Math.log(Math.max(probs.d[i * c + targets[i]], 1e-12));
      n++;
    }
    n = Math.max(1, n);
    var out = new T(new Float64Array([loss / n]), 1, 1, [logits], function (g) {
      var s = g[0] / n;
      for (var i2 = 0; i2 < r; i2++) {
        if (!keep[i2]) continue;
        for (var j4 = 0; j4 < c; j4++) logits.g[i2 * c + j4] += s * probs.d[i2 * c + j4];
        logits.g[i2 * c + targets[i2]] -= s;
      }
    });
    return out;
  }

  function backward(root) {
    var order = [];
    var stack = [[root, false]];
    while (stack.length) {
      var item = stack.pop();
      var node = item[0], expanded = item[1];
      if (expanded) { order.push(node); continue; }
      if (node.done) continue;
      node.done = true;
      stack.push([node, true]);
      if (node.parents) for (var i = 0; i < node.parents.length; i++) if (!node.parents[i].done) stack.push([node.parents[i], false]);
    }
    root.g[0] = 1;
    for (var i = order.length - 1; i >= 0; i--) {
      var n = order[i];
      if (n.back) n.back(n.g);
    }
  }

  /* --- the model ------------------------------------------------------- */

  function Model(opts) {
    var o = { vocab: 32, ctx: 32, d: 48, heads: 4, layers: 2, batch: 6, lr: 3e-3, seed: 1234567 };
    if (opts) for (var k in opts) if (Object.prototype.hasOwnProperty.call(opts, k)) o[k] = opts[k];
    this.cfg = o;
    var seed = o.seed >>> 0 || 1;
    var rng = function () {
      seed ^= seed << 13; seed >>>= 0;
      seed ^= seed >>> 17;
      seed ^= seed << 5; seed >>>= 0;
      return seed / 4294967296;
    };
    this.rng = rng;

    var d = o.d, V = o.vocab, sd = 1 / Math.sqrt(d), self = this;
    this.params = [];
    function param(r, c, scale) {
      var t = build(r, c, function () { return (rng() * 2 - 1) * scale; });
      self.params.push(t);
      return t;
    }

    this.Wte = param(V, d, 0.08);
    this.Wpe = param(o.ctx, d, 0.02);
    this.layers = [];
    for (var l = 0; l < o.layers; l++) {
      this.layers.push({
        Wq: param(d, d, sd), Wk: param(d, d, sd), Wv: param(d, d, sd), Wo: param(d, d, sd),
        g1: param(1, d, 0.05), b1: param(1, d, 0),
        W1: param(d, 4 * d, sd), c1: param(1, 4 * d, 0),
        W2: param(4 * d, d, sd), c2: param(1, d, 0),
        g2: param(1, d, 0.05), b2: param(1, d, 0)
      });
    }
    this.gf = param(1, d, 0.05);
    this.bf = param(1, d, 0);
    this.Wlm = param(d, V, sd);

    this.step = 0;
    this.tokens = 0;
    this.t = 0;
    this.m = [];
    this.v = [];
    for (var i = 0; i < this.params.length; i++) {
      this.m.push(new Float64Array(this.params[i].d.length));
      this.v.push(new Float64Array(this.params[i].d.length));
    }
    this.lastAttn = null;
  }

  Model.prototype.size = function () {
    var n = 0;
    for (var i = 0; i < this.params.length; i++) n += this.params[i].d.length;
    return n;
  };

  Model.prototype.forward = function (ids, targets, keep, batch, seq, watch) {
    var o = this.cfg, d = o.d, heads = o.heads, hd = d / heads;
    var rows = batch * seq;
    var mask = new Uint8Array(rows * rows);
    for (var i = 0; i < rows; i++) {
      var si = (i / seq) | 0, pi = i % seq;
      for (var j = 0; j < rows; j++) {
        var sj = (j / seq) | 0, pj = j % seq;
        mask[i * rows + j] = (si === sj && pj <= pi) ? 1 : 0;
      }
    }
    var pos = new Array(rows);
    for (var p = 0; p < rows; p++) pos[p] = p % seq;

    var x = add(gather(this.Wte, ids), gather(this.Wpe, pos));
    this.lastAttn = null;
    for (var l = 0; l < this.layers.length; l++) {
      var L = this.layers[l];
      var h = lnorm(x, L.g1, L.b1);
      var q = matmul(h, L.Wq), k = matmul(h, L.Wk), v = matmul(h, L.Wv);
      var outs = [];
      for (var hh = 0; hh < heads; hh++) {
        var qh = sliceCols(q, hh * hd, hd), kh = sliceCols(k, hh * hd, hd), vh = sliceCols(v, hh * hd, hd);
        var scores = smul(matmul(qh, transpose(kh)), 1 / Math.sqrt(hd));
        var probs = attention(scores, mask);
        if (watch && l === this.layers.length - 1) this.lastAttn = { probs: probs.d, rows: rows, seq: seq, heads: heads, head: hh };
        outs.push(matmul(probs, vh));
      }
      x = add(x, matmul(concatCols(outs), L.Wo));
      var h2 = lnorm(x, L.g2, L.b2);
      var act = gelu(badd(matmul(h2, L.W1), L.c1));
      x = add(x, badd(matmul(act, L.W2), L.c2));
    }
    var hf = lnorm(x, this.gf, this.bf);
    var logits = matmul(hf, this.Wlm);
    return { loss: crossEntropy(logits, targets, keep), logits: logits };
  };

  Model.prototype.update = function () {
    var params = this.params, M = this.m, V = this.v, cfg = this.cfg;
    this.t++;
    var norm = 0;
    for (var i = 0; i < params.length; i++) {
      var gg = params[i].g;
      for (var j = 0; j < gg.length; j++) norm += gg[j] * gg[j];
    }
    norm = Math.sqrt(norm);
    var scale = norm > 1 ? 1 / norm : 1;
    var b1 = 0.9, b2 = 0.999, eps = 1e-8;
    var bc1 = 1 - Math.pow(b1, this.t), bc2 = 1 - Math.pow(b2, this.t);
    var lr = cfg.lr * Math.min(1, this.t / 40);
    for (var p = 0; p < params.length; p++) {
      var W = params[p].d, G = params[p].g, mp = M[p], vp = V[p];
      for (var k = 0; k < W.length; k++) {
        var grad = G[k] * scale;
        mp[k] = b1 * mp[k] + (1 - b1) * grad;
        vp[k] = b2 * vp[k] + (1 - b2) * grad * grad;
        W[k] -= lr * (mp[k] / bc1) / (Math.sqrt(vp[k] / bc2) + eps);
      }
      params[p].g.fill(0);
    }
  };

  Model.prototype.trainStep = function (data) {
    var o = this.cfg, batch = o.batch, seq = Math.min(o.ctx, 32);
    var rows = batch * seq;
    var ids = new Int32Array(rows), targets = new Int32Array(rows), keep = new Uint8Array(rows);
    for (var b = 0; b < batch; b++) {
      var start = Math.floor(this.rng() * (data.length - seq - 1));
      for (var t = 0; t < seq; t++) {
        var ix = b * seq + t;
        ids[ix] = data[start + t];
        targets[ix] = data[start + t + 1];
        keep[ix] = 1;
      }
    }
    var out = this.forward(ids, targets, keep, batch, seq, false);
    var loss = out.loss.d[0];
    backward(out.loss);
    this.update();
    this.step++;
    this.tokens += rows;
    return loss;
  };

  Model.prototype.sample = function (n, temperature, prime, vocab) {
    var ctx = this.cfg.ctx;
    var ids = [];
    for (var i = 0; i < prime.length; i++) ids.push(prime[i]);
    if (!ids.length) ids.push(Math.floor(this.rng() * this.cfg.vocab));
    var text = '';
    var V = this.cfg.vocab;
    for (var s = 0; s < n; s++) {
      var window = ids.slice(Math.max(0, ids.length - ctx));
      this.lastWindow = window;
      var rows = window.length;
      var targets = new Int32Array(rows), keep = new Uint8Array(rows);
      var out = this.forward(window, targets, keep, 1, rows, s === n - 1);
      var last = (rows - 1) * V;
      var logits = out.logits.d;
      var max = -Infinity;
      for (var j = 0; j < V; j++) if (logits[last + j] > max) max = logits[last + j];
      var probs = new Float64Array(V), sum = 0;
      for (var j2 = 0; j2 < V; j2++) {
        var v = Math.exp((logits[last + j2] - max) / temperature);
        probs[j2] = v;
        sum += v;
      }
      var r = this.rng() * sum, pick = V - 1, acc = 0;
      for (var j3 = 0; j3 < V; j3++) { acc += probs[j3]; if (r <= acc) { pick = j3; break; } }
      ids.push(pick);
      text += vocab[pick];
    }
    return text;
  };

  Model.prototype.attentionMatrix = function () {
    var a = this.lastAttn;
    if (!a) return null;
    return { probs: a.probs, n: a.rows, ids: (this.lastWindow || []).slice() };
  };

  function prepare(text, maxVocab) {
    var lowered = text.toLowerCase().replace(/[^a-z0-9 .,;:'"!?\-\n]/g, '');
    var seen = Object.create(null);
    for (var i = 0; i < lowered.length; i++) seen[lowered[i]] = true;
    var vocab = Object.keys(seen).sort();
    if (maxVocab && vocab.length > maxVocab) {
      vocab.sort(function (a, b) { return (lowered.split(b).length - 1) - (lowered.split(a).length - 1); });
      vocab = vocab.slice(0, maxVocab);
    }
    var index = Object.create(null);
    for (var j = 0; j < vocab.length; j++) index[vocab[j]] = j;
    var ids = new Int32Array(lowered.length), kept = 0;
    for (var k = 0; k < lowered.length; k++) {
      if (lowered[k] in index) ids[kept++] = index[lowered[k]];
    }
    return { vocab: vocab, ids: ids.subarray(0, kept), chars: kept };
  }

  root.Lossfall = { Model: Model, prepare: prepare, backward: backward, T: T };
})(typeof globalThis !== "undefined" ? globalThis : window);
