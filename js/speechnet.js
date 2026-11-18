/* SpeechNet v0.3 — tiny int8 MLP speech gate.
   Same recipe as Level 4 ClapNet: plain JS, vendored JSON weights, no libraries,
   nothing leaves the device. Input is 512 ms at 16 kHz. */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.SpeechNet = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  var SR = 16000;
  var WIN = 8192;
  var NFFT = 512;
  var HOP = 512;
  var NMEL = 32;
  var NFR = 16;
  var FEAT = 128;
  var BUDGET = 12;
  var DISPLAY = { speech: 'Voices', music: 'Music', clapping: 'Clapping', other: 'Other', quiet: 'Quiet' };

  function SpeechNet() {
    this.M = null;
    this.L = null;
    this.ready = false;
    this.off = false;
    this.ms = [];
    this.fb = null;
    this.hw = null;
    this.re = null;
    this.im = null;
    this.baseTH = 0.40;
    this.TH = 0.40;
    this.MARGIN = 0.05;
    this.QUIET = 0.004;
  }

  SpeechNet.DISPLAY = DISPLAY;
  SpeechNet.SR = SR;
  SpeechNet.WIN = WIN;
  SpeechNet.BUDGET = BUDGET;

  SpeechNet.prototype._b64 = function (s) {
    var bin = typeof atob === 'function' ? atob(s) : Buffer.from(s, 'base64').toString('binary');
    var o = new Int8Array(bin.length);
    for (var i = 0; i < bin.length; i++) {
      var v = bin.charCodeAt(i);
      o[i] = v > 127 ? v - 256 : v;
    }
    return o;
  };

  SpeechNet.prototype._mel = function () {
    var n = NMEL, nf = NFFT, m = function (f) { return 2595 * Math.log10(1 + f / 700); };
    var im = function (x) { return 700 * (Math.pow(10, x / 2595) - 1); };
    var lo = m(100), hi = m(7800), b = [], fb = [];
    var i, k;
    for (i = 0; i < n + 2; i++) b.push(Math.floor((nf + 1) * im(lo + (hi - lo) * i / (n + 1)) / SR));
    for (i = 0; i < n; i++) {
      var a = b[i], c = b[i + 1], d = b[i + 2];
      if (c <= a) c = a + 1;
      if (d <= c) d = c + 1;
      var r = new Float32Array(nf / 2 + 1);
      for (k = a; k < c && k < r.length; k++) r[k] = (k - a) / Math.max(c - a, 1);
      for (k = c; k < d && k < r.length; k++) r[k] = (d - k) / Math.max(d - c, 1);
      fb.push({ r: r, a: a, d: d });
    }
    return fb;
  };

  SpeechNet.prototype._fft = function (re, im) {
    var n = re.length, i, j, b, t;
    for (i = 1, j = 0; i < n; i++) {
      b = n >> 1;
      for (; j & b; b >>= 1) j ^= b;
      j ^= b;
      if (i < j) {
        t = re[i]; re[i] = re[j]; re[j] = t;
        t = im[i]; im[i] = im[j]; im[j] = t;
      }
    }
    for (var l = 2; l <= n; l <<= 1) {
      var a = -2 * Math.PI / l, h = l / 2;
      for (var k = 0; k < h; k++) {
        var c = Math.cos(a * k), s = Math.sin(a * k);
        for (i = 0; i < n; i += l) {
          var p = i + k + h, xr = re[p] * c - im[p] * s, xi = re[p] * s + im[p] * c;
          re[p] = re[i + k] - xr; im[p] = im[i + k] - xi;
          re[i + k] += xr; im[i + k] += xi;
        }
      }
    }
  };

  SpeechNet.prototype.load = function (M) {
    this.M = M;
    var self = this;
    this.L = M.layers.map(function (l) {
      var q = self._b64(l.w8), w = new Float32Array(q.length);
      for (var i = 0; i < q.length; i++) w[i] = q[i] * l.s;
      return { w: w, r: l.shape[0], c: l.shape[1], b: Float32Array.from(l.b) };
    });
    this.fb = this._mel();
    this.hw = new Float32Array(NFFT);
    for (var i = 0; i < NFFT; i++) this.hw[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (NFFT - 1));
    this.re = new Float64Array(NFFT);
    this.im = new Float64Array(NFFT);
    this.classes = M.classes.slice();
    if (M.th != null && isFinite(M.th)) {
      this.baseTH = +M.th;
      this.TH = +M.th;
    }
    if (M.margin != null && isFinite(M.margin)) this.MARGIN = +M.margin;
    this.ready = true;
    this.off = false;
    return this;
  };

  SpeechNet.prototype.resample = function (x, fs, n) {
    if (Math.abs(fs - SR) < 0.5) {
      var out0 = new Float32Array(n);
      var take = Math.min(n, x.length);
      if (x.subarray) out0.set(x.subarray(Math.max(0, x.length - take), x.length), n - take);
      else {
        for (var i = 0; i < take; i++) out0[n - take + i] = x[x.length - take + i];
      }
      return out0;
    }
    var out = new Float32Array(n), r = fs / SR, K = 12, fc = 7200 / fs, h = [], sum = 0, k;
    for (k = -K; k <= K; k++) {
      var v = k === 0 ? 2 * fc : Math.sin(2 * Math.PI * fc * k) / (Math.PI * k);
      v *= 0.54 + 0.46 * Math.cos(Math.PI * k / K);
      h.push(v); sum += v;
    }
    var srcLen = x.length;
    // map output i=0 to the last n input-resampled samples (tail of the buffer)
    var inputNeed = n * r;
    var inputStart = srcLen - inputNeed;
    for (i = 0; i < n; i++) {
      var p = inputStart + i * r, i0 = Math.floor(p), f = p - i0, a = 0, b = 0;
      for (k = -K; k <= K; k++) {
        var j = i0 + k;
        if (j >= 0 && j < srcLen) a += x[j] * h[k + K];
        if (j + 1 >= 0 && j + 1 < srcLen) b += x[j + 1] * h[k + K];
      }
      out[i] = (a * (1 - f) + b * f) / sum;
    }
    return out;
  };

  SpeechNet.prototype.logmel = function (w) {
    var mx = 1e-6, i, fr, m;
    for (i = 0; i < w.length; i++) {
      var a = w[i] < 0 ? -w[i] : w[i];
      if (a > mx) mx = a;
    }
    var M = [];
    var re = this.re, im = this.im;
    for (fr = 0; fr < NFR; fr++) {
      var off = fr * HOP;
      for (i = 0; i < NFFT; i++) {
        re[i] = ((off + i < w.length) ? w[off + i] : 0) / mx * this.hw[i];
        im[i] = 0;
      }
      this._fft(re, im);
      var row = new Float32Array(NMEL);
      for (m = 0; m < NMEL; m++) {
        var B = this.fb[m], s = 0;
        for (i = B.a; i < B.d && i < 257; i++) if (B.r[i]) s += B.r[i] * (re[i] * re[i] + im[i] * im[i]);
        row[m] = Math.log(s + 1e-6);
      }
      M.push(row);
    }
    return M;
  };

  SpeechNet.prototype.feat = function (w) {
    var M = this.logmel(w);
    var mean = new Float32Array(NMEL);
    var std = new Float32Array(NMEL);
    var mx = new Float32Array(NMEL);
    var dmean = new Float32Array(NMEL);
    var fr, m, i;
    for (m = 0; m < NMEL; m++) {
      mx[m] = -1e9;
      for (fr = 0; fr < NFR; fr++) {
        mean[m] += M[fr][m];
        if (M[fr][m] > mx[m]) mx[m] = M[fr][m];
      }
      mean[m] /= NFR;
      for (fr = 0; fr < NFR; fr++) {
        var d = M[fr][m] - mean[m];
        std[m] += d * d;
      }
      std[m] = Math.sqrt(std[m] / NFR);
      for (fr = 1; fr < NFR; fr++) dmean[m] += M[fr][m] - M[fr - 1][m];
      dmean[m] /= (NFR - 1);
    }
    var f = new Float32Array(FEAT);
    f.set(mean, 0);
    f.set(std, NMEL);
    f.set(mx, NMEL * 2);
    f.set(dmean, NMEL * 3);
    for (i = 0; i < FEAT; i++) f[i] = (f[i] - this.M.mu[i]) / this.M.sd[i];
    return f;
  };

  SpeechNet.prototype.forward = function (f) {
    var h = f, li, j, i, l, o, s;
    for (li = 0; li < this.L.length - 1; li++) {
      l = this.L[li];
      o = new Float32Array(l.c);
      for (j = 0; j < l.c; j++) {
        s = l.b[j];
        for (i = 0; i < l.r; i++) s += h[i] * l.w[i * l.c + j];
        o[j] = s > 0 ? s : 0;
      }
      h = o;
    }
    l = this.L[this.L.length - 1];
    var logits = new Float32Array(l.c);
    var maxv = -1e9;
    for (j = 0; j < l.c; j++) {
      s = l.b[j];
      for (i = 0; i < l.r; i++) s += h[i] * l.w[i * l.c + j];
      logits[j] = s;
      if (s > maxv) maxv = s;
    }
    var scores = {};
    var sum = 0, exps = new Float32Array(l.c);
    for (j = 0; j < l.c; j++) {
      exps[j] = Math.exp(logits[j] - maxv);
      sum += exps[j];
    }
    for (j = 0; j < l.c; j++) scores[this.classes[j]] = exps[j] / sum;
    return scores;
  };

  SpeechNet.prototype.rms = function (w) {
    var s = 0, i;
    for (i = 0; i < w.length; i++) s += w[i] * w[i];
    return Math.sqrt(s / Math.max(1, w.length));
  };

  SpeechNet.prototype.analyze16k = function (w16) {
    var t0 = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
    var energy = this.rms(w16);
    var scores = { speech: 0, music: 0, clapping: 0, other: 0 };
    var label = 'quiet';
    if (energy < this.QUIET) {
      label = 'quiet';
    } else if (this.ready && !this.off) {
      scores = this.forward(this.feat(w16));
      var best = 'other', bestP = -1, k;
      for (k in scores) {
        if (scores[k] > bestP) { bestP = scores[k]; best = k; }
      }
      label = best;
      // Pure held tones have almost no change in log-mel over time; do not call them speech.
      if (label === 'speech' && this._looksLikeTone(w16)) {
        scores.speech = Math.min(scores.speech, 0.2);
        scores.music = Math.max(scores.music, 0.6);
        label = 'music';
      }
    }
    var dt = ((typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now()) - t0;
    this._note(dt);
    var pSpeech = scores.speech || 0;
    var pMusic = scores.music || 0;
    var pClap = scores.clapping || 0;
    var isSpeech = label !== 'quiet' && pSpeech >= this.TH && pSpeech >= pMusic + this.MARGIN && pSpeech >= pClap + this.MARGIN;
    return {
      scores: scores,
      label: label,
      heard: DISPLAY[label] || label,
      pSpeech: pSpeech,
      isSpeech: isSpeech,
      rms: energy,
      ms: dt
    };
  };

  SpeechNet.prototype.analyze = function (samples, sr) {
    var w = this.resample(samples, sr || SR, WIN);
    return this.analyze16k(w);
  };

  SpeechNet.prototype._looksLikeTone = function (w16) {
    var M = this.logmel(w16);
    var m, fr, mean, v, sumStd = 0;
    for (m = 0; m < NMEL; m++) {
      mean = 0;
      for (fr = 0; fr < NFR; fr++) mean += M[fr][m];
      mean /= NFR;
      v = 0;
      for (fr = 0; fr < NFR; fr++) v += (M[fr][m] - mean) * (M[fr][m] - mean);
      sumStd += Math.sqrt(v / NFR);
    }
    var meanStd = sumStd / NMEL;
    return meanStd < 0.18;
  };

  SpeechNet.prototype.setThreshold = function (th) {
    var cap = 0.70, base = this.baseTH != null ? this.baseTH : 0.40;
    if (th == null || !isFinite(th)) {
      this.TH = base;
      return this.TH;
    }
    this.TH = Math.max(base, Math.min(cap, +th));
    return this.TH;
  };

  SpeechNet.roomThresholdFromScores = function (pSpeechList, opts) {
    opts = opts || {};
    var base = opts.baseTh != null ? opts.baseTh : 0.40;
    var cap = opts.cap != null ? opts.cap : 0.70;
    var margin = opts.margin != null ? opts.margin : 0.10;
    var arr = [];
    var i;
    for (i = 0; i < (pSpeechList || []).length; i++) arr.push(+pSpeechList[i] || 0);
    arr.sort(function (a, b) { return a - b; });
    var p95 = 0;
    if (arr.length) {
      var idx = Math.min(arr.length - 1, Math.max(0, Math.ceil(0.95 * arr.length) - 1));
      p95 = arr[idx];
    }
    var speechTh = Math.max(base, Math.min(cap, p95 + margin));
    return { speechTh: speechTh, p95Speech: p95, windows: arr.length, baseTh: base };
  };

  SpeechNet.prototype.adaptFromRoom = function (samples, sr, opts) {
    opts = opts || {};
    sr = sr || SR;
    var base = opts.baseTh != null ? opts.baseTh : (this.baseTH != null ? this.baseTH : 0.40);
    var winSrc = Math.max(1, Math.round(WIN * sr / SR));
    var hopSrc = Math.max(1, Math.round(winSrc / 2));
    var scores = [];
    var rmses = [];
    var i;
    if (!samples || !samples.length) {
      return { speechTh: base, p95Speech: 0, noiseFloor: 0.002, windows: 0 };
    }
    for (i = 0; i + winSrc <= samples.length; i += hopSrc) {
      var slice = samples.subarray ? samples.subarray(i, i + winSrc) : samples.slice(i, i + winSrc);
      var r = this.analyze(slice, sr);
      scores.push(r.pSpeech || 0);
      rmses.push(r.rms || 0);
    }
    var adapted = SpeechNet.roomThresholdFromScores(scores, { baseTh: base, cap: opts.cap, margin: opts.margin });
    rmses.sort(function (a, b) { return a - b; });
    var floor = 0.002;
    if (rmses.length) {
      var fi = Math.min(rmses.length - 1, Math.max(0, Math.floor(0.8 * rmses.length)));
      floor = Math.max(0.0003, rmses[fi]);
    }
    adapted.noiseFloor = floor;
    this.setThreshold(adapted.speechTh);
    return adapted;
  };

  SpeechNet.prototype._note = function (ms) {
    this.ms.push(ms);
    if (this.ms.length > 40) this.ms.shift();
    if (this.ms.length >= 8) {
      var a = this.ms.slice().sort(function (x, y) { return x - y; });
      if (a[a.length >> 1] > BUDGET) this.off = true;
    }
  };

  SpeechNet.prototype.medianMs = function () {
    if (!this.ms.length) return null;
    var a = this.ms.slice().sort(function (x, y) { return x - y; });
    return a[a.length >> 1];
  };

  function loadFromObject(M) {
    var n = new SpeechNet();
    n.load(M);
    return n;
  }

  function loadFromPath(path) {
    var fs = require('fs');
    return loadFromObject(JSON.parse(fs.readFileSync(path, 'utf8')));
  }

  return {
    SpeechNet: SpeechNet,
    load: loadFromObject,
    loadFromPath: typeof require === 'function' ? loadFromPath : null,
    DISPLAY: DISPLAY,
    SR: SR,
    WIN: WIN
  };
});
