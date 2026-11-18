/* Talk Meter v0.1 — in-browser audio features (MFCC, VAD, music cues).
   No audio is stored or sent; only numeric features leave this module. */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.TMAudio = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  var TARGET_SR = 16000;
  var FRAME = 400;
  var HOP = 160;
  var NFFT = 512;
  var N_MEL = 26;
  var N_MFCC = 13;
  var F0_MIN = 70;
  var F0_MAX = 600;

  function resampleLinear(input, inRate, outRate) {
    if (Math.abs(inRate - outRate) < 0.5) return new Float32Array(input);
    var outLen = Math.max(0, Math.floor(input.length * outRate / inRate));
    var out = new Float32Array(outLen);
    var ratio = inRate / outRate;
    var last = input.length - 1;
    for (var i = 0; i < outLen; i++) {
      var x = i * ratio;
      var i0 = Math.floor(x);
      var i1 = i0 + 1;
      var f = x - i0;
      if (i0 < 0) i0 = 0;
      if (i0 > last) i0 = last;
      if (i1 > last) i1 = last;
      out[i] = input[i0] * (1 - f) + input[i1] * f;
    }
    return out;
  }

  function resample(input, inRate, outRate) {
    if (Math.abs(inRate - outRate) < 0.5) return new Float32Array(input);
    if (inRate > outRate * 1.15) {
      var win = Math.max(2, Math.round(inRate / outRate));
      var tmp = new Float32Array(input.length);
      var acc = 0;
      for (var i = 0; i < input.length; i++) {
        acc += input[i];
        if (i >= win) acc -= input[i - win];
        tmp[i] = acc / Math.min(i + 1, win);
      }
      return resampleLinear(tmp, inRate, outRate);
    }
    return resampleLinear(input, inRate, outRate);
  }

  function fft(re, im) {
    var n = re.length;
    var bits = 0;
    while ((1 << bits) < n) bits++;
    var i, j;
    for (i = 0; i < n; i++) {
      var rev = 0, x = i;
      for (j = 0; j < bits; j++) {
        rev = (rev << 1) | (x & 1);
        x >>= 1;
      }
      if (rev > i) {
        var tr = re[i]; re[i] = re[rev]; re[rev] = tr;
        tr = im[i]; im[i] = im[rev]; im[rev] = tr;
      }
    }
    for (var size = 2; size <= n; size <<= 1) {
      var half = size >> 1;
      var ang = -2 * Math.PI / size;
      for (i = 0; i < n; i += size) {
        for (j = 0; j < half; j++) {
          var wr = Math.cos(ang * j);
          var wi = Math.sin(ang * j);
          var kr = re[i + j + half];
          var ki = im[i + j + half];
          var tr2 = wr * kr - wi * ki;
          var ti2 = wr * ki + wi * kr;
          re[i + j + half] = re[i + j] - tr2;
          im[i + j + half] = im[i + j] - ti2;
          re[i + j] += tr2;
          im[i + j] += ti2;
        }
      }
    }
  }

  var HANN = (function () {
    var w = new Float32Array(FRAME);
    for (var i = 0; i < FRAME; i++) {
      w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (FRAME - 1)));
    }
    return w;
  })();

  function magnitudeSpectrum(frame) {
    var re = new Float32Array(NFFT);
    var im = new Float32Array(NFFT);
    var n = Math.min(frame.length, FRAME, NFFT);
    for (var i = 0; i < n; i++) re[i] = frame[i] * HANN[i];
    fft(re, im);
    var mag = new Float32Array(NFFT / 2 + 1);
    for (i = 0; i <= NFFT / 2; i++) {
      mag[i] = Math.sqrt(re[i] * re[i] + im[i] * im[i]);
    }
    return mag;
  }

  function hzToMel(hz) { return 2595 * Math.log10(1 + hz / 700); }
  function melToHz(m) { return 700 * (Math.pow(10, m / 2595) - 1); }

  var MEL = (function () {
    var lowM = hzToMel(20);
    var highM = hzToMel(TARGET_SR / 2);
    var pts = [];
    var i;
    for (i = 0; i < N_MEL + 2; i++) {
      pts.push(melToHz(lowM + (highM - lowM) * i / (N_MEL + 1)));
    }
    var bins = pts.map(function (hz) {
      return Math.floor((NFFT + 1) * hz / TARGET_SR);
    });
    var filters = [];
    for (var m = 0; m < N_MEL; m++) {
      var f = new Float32Array(NFFT / 2 + 1);
      var a = bins[m], b = bins[m + 1], c = bins[m + 2];
      if (b <= a) b = a + 1;
      if (c <= b) c = b + 1;
      var j;
      for (j = a; j < b && j < f.length; j++) f[j] = (j - a) / (b - a);
      for (j = b; j < c && j < f.length; j++) f[j] = (c - j) / (c - b);
      filters.push(f);
    }
    return filters;
  })();

  var DCT = (function () {
    var M = [];
    for (var k = 0; k < N_MFCC; k++) {
      var row = new Float32Array(N_MEL);
      var scale = k === 0 ? Math.sqrt(1 / N_MEL) : Math.sqrt(2 / N_MEL);
      for (var n = 0; n < N_MEL; n++) {
        row[n] = scale * Math.cos(Math.PI * k * (n + 0.5) / N_MEL);
      }
      M.push(row);
    }
    return M;
  })();

  function mfccFromMag(mag) {
    var logMel = new Float32Array(N_MEL);
    for (var m = 0; m < N_MEL; m++) {
      var s = 0, f = MEL[m];
      for (var i = 0; i < f.length; i++) s += f[i] * mag[i];
      logMel[m] = Math.log(s + 1e-10);
    }
    var ceps = new Float32Array(N_MFCC);
    for (var k = 0; k < N_MFCC; k++) {
      var v = 0, row = DCT[k];
      for (var n = 0; n < N_MEL; n++) v += row[n] * logMel[n];
      ceps[k] = v;
    }
    return ceps;
  }

  function rms(frame) {
    var s = 0;
    for (var i = 0; i < frame.length; i++) s += frame[i] * frame[i];
    return Math.sqrt(s / Math.max(1, frame.length));
  }

  function zcr(frame) {
    var c = 0;
    for (var i = 1; i < frame.length; i++) {
      if ((frame[i] >= 0) !== (frame[i - 1] >= 0)) c++;
    }
    return c / frame.length;
  }

  function spectralCentroid(mag, sr) {
    var num = 0, den = 0;
    var binHz = sr / NFFT;
    for (var i = 1; i < mag.length; i++) {
      num += i * binHz * mag[i];
      den += mag[i];
    }
    return den > 1e-12 ? num / den : 0;
  }

  function spectralFlatness(mag) {
    var start = 2;
    var logSum = 0, sum = 0, n = mag.length - start;
    for (var i = start; i < mag.length; i++) {
      var v = mag[i] + 1e-12;
      logSum += Math.log(v);
      sum += v;
    }
    return Math.exp(logSum / n) / (sum / n);
  }

  function spectralFlux(mag, prev) {
    if (!prev) return 0;
    var s = 0;
    for (var i = 0; i < mag.length; i++) {
      var d = mag[i] - prev[i];
      if (d > 0) s += d;
    }
    return s / mag.length;
  }

  function estimateF0(frame, sr) {
    var minLag = Math.floor(sr / F0_MAX);
    var maxLag = Math.min(Math.floor(sr / F0_MIN), frame.length - 2);
    var i, mean = 0;
    for (i = 0; i < frame.length; i++) mean += frame[i];
    mean /= frame.length;
    var e0 = 0;
    for (i = 0; i < frame.length; i++) {
      var x = frame[i] - mean;
      e0 += x * x;
    }
    if (e0 < 1e-12) return { f0: 0, conf: 0 };
    var best = 0, bestLag = 0;
    var acAt = {};
    for (var lag = minLag; lag <= maxLag; lag++) {
      var ac = 0;
      var n = frame.length - lag;
      for (i = 0; i < n; i++) {
        ac += (frame[i] - mean) * (frame[i + lag] - mean);
      }
      ac /= e0;
      acAt[lag] = ac;
      if (ac > best) {
        best = ac;
        bestLag = lag;
      }
    }
    // Prefer a shorter lag (higher F0) when it is nearly as strong — avoids octave-down errors on tones.
    if (bestLag) {
      for (var div = 2; div <= 4; div++) {
        var l = Math.round(bestLag / div);
        if (l < minLag) continue;
        var ac2 = acAt[l];
        if (ac2 == null) continue;
        if (ac2 >= 0.8 * best) {
          bestLag = l;
        }
      }
    }
    return { f0: bestLag ? sr / bestLag : 0, conf: Math.max(0, best) };
  }

  function hpsPeaks(mag, sr, count) {
    var n = mag.length;
    var hps = new Float32Array(n);
    var i;
    for (i = 0; i < n; i++) hps[i] = mag[i];
    for (var h = 2; h <= 3; h++) {
      var lim = Math.floor(n / h);
      for (i = 0; i < lim; i++) hps[i] *= mag[i * h];
    }
    var minBin = Math.floor(F0_MIN * NFFT / sr);
    var maxBin = Math.min(Math.floor(F0_MAX * NFFT / sr), n - 2);
    var peaks = [];
    for (i = minBin + 1; i < maxBin; i++) {
      if (hps[i] > hps[i - 1] && hps[i] >= hps[i + 1] && hps[i] > 0) {
        peaks.push({ bin: i, val: hps[i], hz: i * sr / NFFT });
      }
    }
    peaks.sort(function (a, b) { return b.val - a.val; });
    var kept = [];
    for (i = 0; i < peaks.length && kept.length < count; i++) {
      var p = peaks[i];
      var ok = true;
      for (var k = 0; k < kept.length; k++) {
        var ratio = p.hz / kept[k].hz;
        var nearInt = Math.round(ratio);
        if (nearInt >= 1 && Math.abs(ratio - nearInt) < 0.08) { ok = false; break; }
        ratio = kept[k].hz / p.hz;
        nearInt = Math.round(ratio);
        if (nearInt >= 1 && Math.abs(ratio - nearInt) < 0.08) { ok = false; break; }
      }
      if (ok) kept.push(p);
    }
    return kept;
  }

  function extractFrame(frame, sr, prevMag) {
    var energy = rms(frame);
    var mag = magnitudeSpectrum(frame);
    var ceps = mfccFromMag(mag);
    var pitch = estimateF0(frame, sr || TARGET_SR);
    return {
      rms: energy,
      logRms: Math.log(energy + 1e-8),
      zcr: zcr(frame),
      centroid: spectralCentroid(mag, sr || TARGET_SR),
      flatness: spectralFlatness(mag),
      flux: spectralFlux(mag, prevMag),
      mfcc: ceps,
      f0: pitch.f0,
      f0conf: pitch.conf,
      pitches: hpsPeaks(mag, sr || TARGET_SR, 3).map(function (p) { return p.hz; }),
      _mag: mag
    };
  }

  function embed(feat) {
    var v = new Float32Array(N_MFCC - 1);
    for (var i = 1; i < N_MFCC; i++) v[i - 1] = feat.mfcc[i];
    return v;
  }

  function averageEmbeds(vecs) {
    if (!vecs || !vecs.length) return null;
    var dim = vecs[0].length;
    var m = new Float32Array(dim);
    var i, j;
    for (i = 0; i < vecs.length; i++) {
      for (j = 0; j < dim; j++) m[j] += vecs[i][j];
    }
    for (j = 0; j < dim; j++) m[j] /= vecs.length;
    return m;
  }

  function f0Compatible(feat, profile, semitones) {
    var lim = semitones == null ? 8 : semitones;
    if (!profile || !profile.meanF0 || !feat || feat.f0conf < 0.25 || !feat.f0) return true;
    var st = Math.abs(12 * Math.log(feat.f0 / profile.meanF0) / Math.LN2);
    return st <= lim;
  }

  function f0Semitones(feat, profile) {
    if (!profile || !profile.meanF0 || !feat || !feat.f0) return null;
    return 12 * Math.log(feat.f0 / profile.meanF0) / Math.LN2;
  }

  function isVoiced(feat, noiseFloor, sensitivity) {
    var floor = noiseFloor || 0.002;
    var sens = sensitivity == null ? 0.5 : sensitivity;
    var margin = 2.6 - sens * 1.4;
    var energyOk = feat.rms > floor * margin;
    var pitchOk = feat.f0conf > 0.22 && feat.f0 >= F0_MIN && feat.f0 <= F0_MAX;
    var spectralOk = feat.centroid > 180 && feat.centroid < 5200 && feat.flatness < 0.65;
    var notClicky = feat.zcr < 0.38;
    return energyOk && notClicky && (pitchOk || spectralOk);
  }

  function variance(vals) {
    if (!vals.length) return 0;
    var m = 0, i;
    for (i = 0; i < vals.length; i++) m += vals[i];
    m /= vals.length;
    var s = 0;
    for (i = 0; i < vals.length; i++) s += (vals[i] - m) * (vals[i] - m);
    return s / vals.length;
  }

  function musicScore(recent) {
    if (!recent || recent.length < 8) return 0;
    var voiced = [];
    var i, j;
    for (i = 0; i < recent.length; i++) {
      if (recent[i].rms > 1e-5) voiced.push(recent[i]);
    }
    if (voiced.length < 8) return 0;
    var dim = N_MFCC - 1;
    var means = [];
    for (j = 0; j < dim; j++) means[j] = 0;
    for (i = 0; i < voiced.length; i++) {
      var e = embed(voiced[i]);
      for (j = 0; j < dim; j++) means[j] += e[j];
    }
    for (j = 0; j < dim; j++) means[j] /= voiced.length;
    var varSum = 0;
    for (i = 0; i < voiced.length; i++) {
      e = embed(voiced[i]);
      for (j = 0; j < dim; j++) {
        var d = e[j] - means[j];
        varSum += d * d;
      }
    }
    var mfccVar = varSum / (voiced.length * dim);

    var f0s = [];
    for (i = 0; i < voiced.length; i++) {
      if (voiced[i].f0conf > 0.28 && voiced[i].f0 > 0) f0s.push(voiced[i].f0);
    }
    var f0cv = 1;
    if (f0s.length >= 6) {
      var m = 0;
      for (i = 0; i < f0s.length; i++) m += f0s[i];
      m /= f0s.length;
      f0cv = m > 0 ? Math.sqrt(variance(f0s)) / m : 1;
    }
    var f0stable = f0cv < 0.03 ? 1 : f0cv < 0.08 ? (0.08 - f0cv) / 0.05 : 0;

    var flat = 0;
    for (i = 0; i < voiced.length; i++) flat += voiced[i].flatness;
    flat /= voiced.length;
    var veryTonal = flat < 0.015 ? 1 : flat < 0.04 ? 0.5 : 0;
    var timbreStable = mfccVar < 0.25 ? 1 : mfccVar < 2.0 ? (2.0 - mfccVar) / 1.75 : 0;

    // Moving formants (speech) produce higher MFCC variance than held tones or static pads.
    if (mfccVar > 4) return 0;

    var score = 0;
    if (veryTonal > 0 && timbreStable > 0) {
      score = Math.max(score, 0.5 * veryTonal + 0.5 * timbreStable);
    }
    if (f0stable > 0.5 && timbreStable > 0.5) {
      score = Math.max(score, 0.6 * f0stable + 0.4 * timbreStable);
    }
    if (flat < 0.02 && mfccVar < 2.5) score = Math.max(score, 0.85);
    if (flat < 0.005 && mfccVar < 0.3) score = Math.max(score, 1);
    return Math.max(0, Math.min(1, score));
  }

  function meanStd(vectors) {
    if (!vectors.length) return null;
    var dim = vectors[0].length;
    var mean = new Float32Array(dim);
    var std = new Float32Array(dim);
    var i, j;
    for (i = 0; i < vectors.length; i++) {
      for (j = 0; j < dim; j++) mean[j] += vectors[i][j];
    }
    for (j = 0; j < dim; j++) mean[j] /= vectors.length;
    for (i = 0; i < vectors.length; i++) {
      for (j = 0; j < dim; j++) {
        var d = vectors[i][j] - mean[j];
        std[j] += d * d;
      }
    }
    for (j = 0; j < dim; j++) {
      std[j] = Math.sqrt(std[j] / Math.max(1, vectors.length - 1));
      if (std[j] < 1e-3) std[j] = 1e-3;
    }
    return { mean: mean, std: std, count: vectors.length };
  }

  function mahalanobis(vec, profile) {
    var dim = profile.mean.length;
    var s = 0;
    for (var j = 0; j < dim; j++) {
      var st = profile.std[j] < 1e-3 ? 1e-3 : profile.std[j];
      var d = (vec[j] - profile.mean[j]) / st;
      s += d * d;
    }
    return Math.sqrt(s / dim);
  }

  function cosineDistance(vec, profile) {
    var dim = profile.mean.length;
    var dot = 0, a = 0, b = 0;
    for (var j = 0; j < dim; j++) {
      dot += vec[j] * profile.mean[j];
      a += vec[j] * vec[j];
      b += profile.mean[j] * profile.mean[j];
    }
    var c = dot / (Math.sqrt(a * b) + 1e-12);
    return 1 - c;
  }

  function blendedDistance(vec, profile) {
    return 0.7 * mahalanobis(vec, profile) + 0.3 * cosineDistance(vec, profile) * 8;
  }

  function speakerDistance(vec, profile) {
    if (profile.snapshots && profile.snapshots.length) {
      var best = Infinity;
      var shim = { mean: null, std: profile.std };
      for (var i = 0; i < profile.snapshots.length; i++) {
        shim.mean = profile.snapshots[i];
        var d = blendedDistance(vec, shim);
        if (d < best) best = d;
      }
      return best;
    }
    return blendedDistance(vec, profile);
  }

  function percentile(arr, p) {
    if (!arr || !arr.length) return 0;
    var a = arr.slice().sort(function (x, y) { return x - y; });
    var idx = (a.length - 1) * p;
    var lo = Math.floor(idx), hi = Math.ceil(idx);
    if (lo === hi) return a[lo];
    return a[lo] * (hi - idx) + a[hi] * (idx - lo);
  }

  function analyzeBuffer(samples, sr) {
    var x = resample(samples, sr, TARGET_SR);
    var feats = [];
    var prevMag = null;
    for (var start = 0; start + FRAME <= x.length; start += HOP) {
      var copy = new Float32Array(FRAME);
      copy.set(x.subarray(start, start + FRAME));
      var feat = extractFrame(copy, TARGET_SR, prevMag);
      prevMag = feat._mag;
      feat._mag = undefined;
      feats.push(feat);
    }
    return feats;
  }

  function buildProfile(samples, sr, noiseFloor, sensitivity) {
    var feats = analyzeBuffer(samples, sr);
    var vecs = [];
    var f0s = [];
    var i;
    var sens = sensitivity == null ? 0.5 : sensitivity;
    for (i = 0; i < feats.length; i++) {
      if (isVoiced(feats[i], noiseFloor, sens)) {
        vecs.push(embed(feats[i]));
        if (feats[i].f0conf > 0.25 && feats[i].f0 > 0) f0s.push(feats[i].f0);
      }
    }
    var stats = meanStd(vecs);
    if (!stats) {
      return { ok: false, reason: 'no-voiced', frames: 0, voiced: 0, total: feats.length };
    }
    // Short snapshots so different vowels still match the same speaker.
    var SNAP = 8;
    var snapshots = [];
    for (i = 0; i + SNAP <= vecs.length; i += SNAP) {
      snapshots.push(Array.prototype.slice.call(averageEmbeds(vecs.slice(i, i + SNAP))));
    }
    if (!snapshots.length) snapshots.push(Array.prototype.slice.call(stats.mean));
    var snapProfile = { mean: stats.mean, std: stats.std, snapshots: snapshots };
    var dists = [];
    for (i = 0; i < vecs.length; i++) dists.push(speakerDistance(vecs[i], snapProfile));
    var p90 = percentile(dists, 0.9);
    var threshold = Math.max(1.2, Math.min(3.2, p90 * 1.7 + 0.3));
    var meanF0 = 0;
    if (f0s.length) {
      for (i = 0; i < f0s.length; i++) meanF0 += f0s[i];
      meanF0 /= f0s.length;
    }
    return {
      ok: true,
      mean: Array.prototype.slice.call(stats.mean),
      std: Array.prototype.slice.call(stats.std),
      snapshots: snapshots,
      dim: stats.mean.length,
      frames: stats.count,
      total: feats.length,
      durationSec: samples.length / sr,
      voiced: stats.count,
      threshold: threshold,
      meanF0: meanF0,
      distP90: p90
    };
  }

  function noiseFloorFromSamples(samples, sr) {
    var x = resample(samples, sr, TARGET_SR);
    var vals = [];
    for (var start = 0; start + FRAME <= x.length; start += HOP) {
      vals.push(rms(x.subarray(start, start + FRAME)));
    }
    return Math.max(0.0003, percentile(vals, 0.8));
  }

  function FramePusher() {
    this.buf = new Float32Array(FRAME);
    this.fill = 0;
  }
  FramePusher.prototype.push = function (samples, sr) {
    var x = resample(samples, sr, TARGET_SR);
    var out = [];
    var i = 0;
    while (i < x.length) {
      var need = FRAME - this.fill;
      var take = Math.min(need, x.length - i);
      this.buf.set(x.subarray(i, i + take), this.fill);
      this.fill += take;
      i += take;
      if (this.fill === FRAME) {
        var frame = new Float32Array(FRAME);
        frame.set(this.buf);
        out.push(frame);
        this.buf.copyWithin(0, HOP);
        this.fill = FRAME - HOP;
      }
    }
    return out;
  };
  FramePusher.prototype.reset = function () { this.fill = 0; };

  return {
    TARGET_SR: TARGET_SR,
    FRAME: FRAME,
    HOP: HOP,
    N_MFCC: N_MFCC,
    resample: resample,
    extractFrame: extractFrame,
    embed: embed,
    isVoiced: isVoiced,
    musicScore: musicScore,
    meanStd: meanStd,
    speakerDistance: speakerDistance,
    blendedDistance: blendedDistance,
    averageEmbeds: averageEmbeds,
    f0Compatible: f0Compatible,
    f0Semitones: f0Semitones,
    mahalanobis: mahalanobis,
    cosineDistance: cosineDistance,
    analyzeBuffer: analyzeBuffer,
    buildProfile: buildProfile,
    noiseFloorFromSamples: noiseFloorFromSamples,
    FramePusher: FramePusher,
    rms: rms,
    percentile: percentile,
    variance: variance
  };
});
