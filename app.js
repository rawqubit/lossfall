/* lossfall ui — wiring for the model in engine.js. Nothing here computes
   gradients; it reads numbers out of the model and draws them. */
(function () {
  'use strict';

  var Model = globalThis.Lossfall.Model;
  var prepare = globalThis.Lossfall.prepare;

  var CONFIG = { ctx: 16, d: 32, heads: 4, layers: 2, batch: 8, lr: 3e-3 };
  var BUDGET_MS = 11;
  var SAMPLE_MS = 620;
  var MAX_POINTS = 900;

  var $ = function (id) { return document.getElementById(id); };
  var state = {
    text: window.CORPUS || '',
    data: null,
    model: null,
    running: true,
    history: [],
    samples: [],
    temperature: 0.85,
    lastSample: 0,
    lastFrame: 0,
    rate: 0,
    stepAtSecond: 0,
    secondAt: 0,
    startedAt: 0,
    source: 'sonnets'
  };

  function fmt(n) { return n.toLocaleString('en-US'); }
  function clock(ms) {
    var s = Math.floor(ms / 1000);
    return Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0');
  }

  function build(text, source) {
    state.text = text;
    state.source = source || state.source;
    state.data = prepare(text);
    state.model = new Model(Object.assign({ vocab: state.data.vocab.length, seed: (Date.now() % 100000) + 7 }, CONFIG));
    state.history = [];
    state.samples = [];
    state.startedAt = performance.now();
    state.secondAt = state.startedAt;
    state.stepAtSecond = 0;
    state.rate = 0;
    state.lastSample = 0;
    $('params').textContent = fmt(state.model.size());
    $('random').textContent = 'random = ' + Math.log(state.data.vocab.length).toFixed(2);
    $('vocabSize').textContent = state.data.vocab.length + ' symbols';
    $('corpusMeta').textContent = fmt(state.data.chars) + ' characters';
    $('sample').textContent = '';
    $('before').textContent = '';
    $('atStep').textContent = '';
    $('charsOut').textContent = '';
    $('lossDelta').textContent = '';
    $('beat').textContent = '';
    $('chart').innerHTML = '';
    drawAttention(null);
    render();
  }

  /* --- training loop -------------------------------------------------- */
  function loop(now) {
    if (state.running && state.model) {
      var deadline = now + BUDGET_MS;
      var steps = 0;
      while (performance.now() < deadline && steps < 40) {
        var loss = state.model.trainStep(state.data.ids);
        state.history.push({ step: state.model.step, loss: loss });
        if (state.history.length > MAX_POINTS) state.history.shift();
        steps++;
      }
      if (now - state.secondAt >= 1000) {
        state.rate = (state.model.step - state.stepAtSecond) / ((now - state.secondAt) / 1000);
        state.stepAtSecond = state.model.step;
        state.secondAt = now;
      }
      if (now - state.lastSample > SAMPLE_MS) {
        state.lastSample = now;
        takeSample();
      }
    }
    if (now - state.lastFrame > 90) {
      state.lastFrame = now;
      render();
    }
    requestAnimationFrame(loop);
  }

  function takeSample() {
    if (!state.model || state.model.step < 2) return;
    var text;
    try {
      text = state.model.sample(96, state.temperature, prime(), state.data.vocab);
    } catch (error) {
      return;
    }
    state.samples.push({ step: state.model.step, text: text });
    if (state.samples.length > 4) state.samples.shift();
    drawAttention(state.model.attentionMatrix());
    $('sample').textContent = text;
    var previous = state.samples[state.samples.length - 2];
    $('before').textContent = previous ? previous.text.slice(0, 120) : '';
    $('atStep').textContent = 'step ' + fmt(state.model.step);
    $('charsOut').textContent = text.length + ' symbols';
  }

  function prime() {
    var index = state.data.vocab.indexOf('\n');
    if (index < 0) index = state.data.vocab.indexOf(' ');
    return index >= 0 ? [index] : [0];
  }

  /* --- rendering ------------------------------------------------------ */
  function render() {
    if (!state.model) return;
    var losses = state.history;
    var recent = losses.slice(-40);
    var avg = recent.length ? recent.reduce(function (a, b) { return a + b.loss; }, 0) / recent.length : null;
    $('loss').firstChild.nodeValue = avg === null ? '—' : avg.toFixed(2);
    $('steps').textContent = fmt(state.model.step);
    $('tokens').textContent = fmt(state.model.tokens);
    $('rate').textContent = state.rate ? state.rate.toFixed(0) : '—';
    $('runtime').textContent = state.running ? 'running ' + clock(performance.now() - state.startedAt) : 'paused';
    var start = Math.log(state.data.vocab.length);
    var delta = avg === null ? 0 : start - avg;
    $('lossDelta').textContent = avg === null ? '' : '−' + delta.toFixed(2) + ' below random';
    $('beat').textContent = avg !== null && avg < start ? 'beats chance by ' + (100 * delta / start).toFixed(0) + '%' : '';
    drawChart(losses, start);
  }

  function drawChart(points, random) {
    var svg = $('chart');
    if (!points.length) { svg.innerHTML = ''; return; }
    var W = 640, H = 176, pad = 4;
    var maxLoss = Math.max(random, 3.7);
    var minLoss = Math.min(0.4, Math.min.apply(null, points.map(function (p) { return p.loss; })) - 0.15);
    var firstStep = points[0].step, lastStep = Math.max(points[points.length - 1].step, firstStep + 40);
    var x = function (s) { return pad + (W - 2 * pad) * (s - firstStep) / (lastStep - firstStep); };
    var y = function (l) { return pad + (H - 2 * pad) * (1 - (l - minLoss) / (maxLoss - minLoss)); };
    var parts = [];
    var step = 1;
    var gridY = [];
    for (var g = Math.ceil(minLoss / step) * step; g <= maxLoss; g += step) gridY.push(g);
    gridY.forEach(function (l) {
      parts.push('<line x1="0" x2="' + W + '" y1="' + y(l).toFixed(1) + '" y2="' + y(l).toFixed(1) + '" stroke="#1b1b21" stroke-width="1"/>');
      parts.push('<text x="2" y="' + (y(l) - 4).toFixed(1) + '" fill="#4b4b56" font-size="9" font-family="ui-monospace, monospace">' + l + '</text>');
    });
    if (random >= minLoss && random <= maxLoss) {
      parts.push('<line x1="0" x2="' + W + '" y1="' + y(random).toFixed(1) + '" y2="' + y(random).toFixed(1) + '" stroke="#4a4a56" stroke-width="1" stroke-dasharray="3 4"/>');
    }
    var line = '';
    points.forEach(function (p, i) {
      line += (i ? 'L' : 'M') + x(p.step).toFixed(1) + ' ' + y(p.loss).toFixed(1);
    });
    parts.push('<path d="' + line + '" fill="none" stroke="#ffa53d" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"/>');
    parts.push('<path d="' + line + 'L' + x(points[points.length - 1].step).toFixed(1) + ' ' + H + 'L' + x(points[0].step).toFixed(1) + ' ' + H + 'Z" fill="rgba(255,165,61,0.08)" stroke="none"/>');
    svg.innerHTML = parts.join('');
  }

  function drawAttention(matrix) {
    var canvas = $('attn');
    var ctx = canvas.getContext('2d');
    var W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#0b0b0e';
    ctx.fillRect(0, 0, W, H);
    if (!matrix || !matrix.n) return;
    var n = matrix.n;
    var padL = 22, padB = 20, padT = 6;
    var size = Math.min(W - padL, H - padB - padT);
    var cell = size / n;
    var top = padT + (H - padB - padT - size) / 2;
    for (var r = 0; r < n; r++) {
      for (var c = 0; c < n; c++) {
        var p = Math.max(0, Math.min(1, matrix.probs[r * n + c] || 0));
        var alpha = Math.pow(p, 0.4);
        var red = Math.round(16 + 239 * alpha);
        var green = Math.round(16 + 149 * alpha);
        var blue = Math.round(20 + 41 * alpha);
        ctx.fillStyle = 'rgb(' + red + ',' + green + ',' + blue + ')';
        ctx.fillRect(padL + c * cell, top + r * cell, Math.max(1, cell - 1), Math.max(1, cell - 1));
      }
    }
    var vocab = state.data ? state.data.vocab : [];
    ctx.fillStyle = '#9a9aa6';
    ctx.font = '11px ui-monospace, monospace';
    ctx.textAlign = 'center';
    for (var k = 0; k < n; k++) {
      var code = matrix.ids && matrix.ids[k] !== undefined ? matrix.ids[k] : -1;
      var label = code >= 0 ? vocab[code] : '';
      if (label === ' ') label = '\u00b7';
      if (label === '\n') label = '\u00b6';
      ctx.fillText(label || '', padL + k * cell + cell / 2, H - 6);
    }
    ctx.textAlign = 'left';
  }

  /* --- controls ------------------------------------------------------- */
  function toggle() {
    state.running = !state.running;
    $('toggle').textContent = state.running ? 'Pause' : 'Resume';
    render();
  }

  function restart() {
    build(state.text, state.source);
  }

  function currentText(choice) {
    if (choice === 'sonnets') return window.CORPUS || '';
    if (choice === 'self') return document.documentElement.outerHTML;
    return $('pasted').value;
  }

  function applyCorpus(choice) {
    var text = currentText(choice);
    if (!text || text.length < 240) {
      if (choice === 'own') { $('pasted').focus(); return; }
    }
    build(text, choice);
  }

  $('toggle').addEventListener('click', toggle);
  $('restart').addEventListener('click', restart);
  $('corpus').addEventListener('change', function (event) {
    var choice = event.target.value;
    $('pasted').classList.toggle('on', choice === 'own');
    if (choice !== 'own') applyCorpus(choice);
    else if ($('pasted').value.length > 240) applyCorpus('own');
  });
  $('pasted').addEventListener('input', function () {
    if ($('pasted').value.length > 240) applyCorpus('own');
  });
  document.addEventListener('keydown', function (event) {
    if (event.target.matches('input, textarea, select')) return;
    if (event.key === ' ') { event.preventDefault(); toggle(); }
    if (event.key === 'r' || event.key === 'R') restart();
  });

  build(window.CORPUS || '', 'sonnets');
  state.running = true;
  $('toggle').textContent = 'Pause';
  requestAnimationFrame(function (now) { state.lastFrame = now - 1000; loop(now); });
})();
