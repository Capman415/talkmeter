/* Synthetic speech-like signals and tones for tests. Not used in the live classroom app. */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.TMSynth = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  function mulberry32(seed) {
    var a = seed >>> 0;
    return function () {
      a |= 0;
      a = a + 0x6D2B79F5 | 0;
      var t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

  function formantFilter(x, freq, bw, sr) {
    var r = Math.exp(-Math.PI * bw / sr);
    var theta = 2 * Math.PI * freq / sr;
    var a1 = 2 * r * Math.cos(theta);
    var a2 = -r * r;
    var y1 = 0, y2 = 0;
    var y = new Float32Array(x.length);
    for (var i = 0; i < x.length; i++) {
      var out = x[i] + a1 * y1 + a2 * y2;
      y2 = y1;
      y1 = out;
      y[i] = out;
    }
    return y;
  }

  function normalize(x, peak) {
    peak = peak == null ? 0.25 : peak;
    var m = 0, i;
    for (i = 0; i < x.length; i++) {
      var a = Math.abs(x[i]);
      if (a > m) m = a;
    }
    if (m < 1e-12) return x;
    var g = peak / m;
    for (i = 0; i < x.length; i++) x[i] *= g;
    return x;
  }

  /**
   * Source-filter "speech": jittered glottal pulse train through 3 formants,
   * with slow formant drift and an on/off envelope so it is not a steady tone.
   */
  var VOWELS = {
    a: [730, 1090, 2440],
    i: [270, 2290, 3010],
    u: [300, 870, 2240],
    e: [530, 1840, 2480],
    o: [570, 840, 2410]
  };

  /**
   * Source-filter "speech": jittered glottal pulse train through 3 formants
   * that switch between vowel targets, plus syllable envelopes, so the
   * signal moves like speech instead of a held tone.
   */
  function speechLike(opts) {
    opts = opts || {};
    var sr = opts.sr || 16000;
    var seconds = opts.seconds || 2;
    var f0 = opts.f0 || 110;
    var formantScale = opts.formantScale || 1;
    var vowelNames = opts.vowels || ['a', 'e', 'i', 'o', 'u', 'a', 'e'];
    var seed = opts.seed || 1;
    var rng = mulberry32(seed);
    var n = Math.floor(sr * seconds);
    var src = new Float32Array(n);
    var phase = 0;
    var i;
    for (i = 0; i < n; i++) {
      var t = i / sr;
      var contour = 1 + 0.14 * Math.sin(2 * Math.PI * 2.3 * t + seed) +
                    0.07 * Math.sin(2 * Math.PI * 5.1 * t);
      var instF0 = f0 * contour * (1 + (rng() - 0.5) * 0.04);
      var period = sr / Math.max(60, instF0);
      phase += 1;
      if (phase >= period) phase -= period;
      var frac = phase / period;
      src[i] = frac < 0.14 ? (1 - frac / 0.14) : 0;
    }

    // Build a vowel schedule (new target every 80–180 ms).
    var schedule = [];
    var pos = 0;
    var vi = 0;
    while (pos < n) {
      var dur = Math.floor(sr * (0.08 + rng() * 0.10));
      var name = vowelNames[(vi + seed) % vowelNames.length];
      var f = VOWELS[name].slice();
      for (var k = 0; k < f.length; k++) f[k] *= formantScale * (1 + (rng() - 0.5) * 0.04);
      schedule.push({ start: pos, end: Math.min(n, pos + dur), formants: f });
      pos += dur;
      vi++;
    }

    var y = src;
    var bws = opts.bandwidths || [90, 120, 160];
    for (var fi = 0; fi < 3; fi++) {
      y = formantFollow(y, schedule, fi, bws[fi] || 120, sr);
    }
    var out = new Float32Array(n);
    out[0] = y[0];
    for (i = 1; i < n; i++) out[i] = y[i] - 0.97 * y[i - 1];
    for (i = 0; i < n; i++) {
      var t = i / sr;
      var syll = Math.pow(0.5 + 0.5 * Math.sin(2 * Math.PI * 3.6 * t + seed * 1.7), 1.2);
      syll = 0.25 + 0.75 * syll;
      // brief consonant-like dips
      if (Math.sin(2 * Math.PI * 2.1 * t + seed) > 0.88) syll *= 0.12;
      out[i] = out[i] * syll + (rng() - 0.5) * 0.004;
    }
    if (opts.gain != null) {
      var g = opts.gain;
      for (i = 0; i < n; i++) out[i] *= g;
      return out;
    }
    return normalize(out, opts.peak == null ? 0.28 : opts.peak);
  }

  function formantFollow(x, schedule, formantIndex, bw, sr) {
    var y = new Float32Array(x.length);
    var r = Math.exp(-Math.PI * bw / sr);
    var y1 = 0, y2 = 0;
    var si = 0;
    var cur = schedule[0].formants[formantIndex];
    var target = cur;
    var a1 = 0, a2 = 0, theta;
    function coeffs(freq) {
      theta = 2 * Math.PI * freq / sr;
      a1 = 2 * r * Math.cos(theta);
      a2 = -r * r;
    }
    coeffs(cur);
    for (var i = 0; i < x.length; i++) {
      while (si < schedule.length - 1 && i >= schedule[si].end) si++;
      target = schedule[si].formants[formantIndex];
      cur += (target - cur) * 0.08;
      if (i % 32 === 0) coeffs(cur);
      var out = x[i] + a1 * y1 + a2 * y2;
      y2 = y1;
      y1 = out;
      y[i] = out;
    }
    return y;
  }

  var SPEAKERS = {
    teacher: { f0: 112, formantScale: 0.96, vowels: ['a', 'o', 'e', 'u', 'a', 'o'], seed: 11 },
    childA: { f0: 230, formantScale: 1.28, vowels: ['i', 'e', 'a', 'i', 'e'], seed: 22 },
    childB: { f0: 260, formantScale: 1.36, vowels: ['e', 'i', 'u', 'e', 'i'], seed: 33 },
    childC: { f0: 200, formantScale: 1.20, vowels: ['a', 'i', 'e', 'o', 'i'], seed: 44 }
  };

  function speaker(name, seconds, extra) {
    var base = SPEAKERS[name] || SPEAKERS.teacher;
    var opts = {
      f0: base.f0,
      formantScale: base.formantScale,
      vowels: base.vowels.slice(),
      seed: base.seed,
      seconds: seconds || 2,
      sr: 16000
    };
    if (extra) {
      if (extra.seconds) opts.seconds = extra.seconds;
      if (extra.sr) opts.sr = extra.sr;
      if (extra.peak != null) opts.peak = extra.peak;
      if (extra.seed != null) opts.seed = extra.seed;
      if (extra.gain != null) opts.gain = extra.gain;
    }
    return speechLike(opts);
  }

  function mix(buffers, gains) {
    var n = 0, i, j;
    for (i = 0; i < buffers.length; i++) if (buffers[i].length > n) n = buffers[i].length;
    var out = new Float32Array(n);
    for (i = 0; i < buffers.length; i++) {
      var g = gains && gains[i] != null ? gains[i] : 1;
      var b = buffers[i];
      for (j = 0; j < b.length; j++) out[j] += b[j] * g;
    }
    return normalize(out, 0.32);
  }

  function silence(seconds, sr) {
    return new Float32Array(Math.floor((sr || 16000) * seconds));
  }

  function concat(parts) {
    var n = 0, i, j, o = 0;
    for (i = 0; i < parts.length; i++) n += parts[i].length;
    var out = new Float32Array(n);
    for (i = 0; i < parts.length; i++) {
      out.set(parts[i], o);
      o += parts[i].length;
    }
    return out;
  }

  function sine(freq, seconds, sr, peak) {
    sr = sr || 16000;
    var n = Math.floor(sr * seconds);
    var x = new Float32Array(n);
    peak = peak == null ? 0.22 : peak;
    for (var i = 0; i < n; i++) x[i] = Math.sin(2 * Math.PI * freq * i / sr) * peak;
    return x;
  }

  function chord(freqs, seconds, sr, peak) {
    sr = sr || 16000;
    var n = Math.floor(sr * seconds);
    var x = new Float32Array(n);
    var i, k;
    for (k = 0; k < freqs.length; k++) {
      var f = freqs[k], a = 1 / freqs.length;
      for (i = 0; i < n; i++) x[i] += Math.sin(2 * Math.PI * f * i / sr) * a;
    }
    return normalize(x, peak == null ? 0.22 : peak);
  }

  /** Simple looping "music": held notes with slow chord changes, little formant motion. */
  function musicLoop(seconds, sr) {
    sr = sr || 16000;
    var n = Math.floor(sr * seconds);
    var x = new Float32Array(n);
    var notes = [
      [262, 330, 392],
      [294, 370, 440],
      [262, 349, 415],
      [247, 311, 392]
    ];
    var beat = Math.floor(sr * 0.5);
    for (var i = 0; i < n; i++) {
      var chordIdx = Math.floor(i / beat) % notes.length;
      var freqs = notes[chordIdx];
      var env = 1;
      var pos = i % beat;
      if (pos < sr * 0.02) env = pos / (sr * 0.02);
      if (pos > beat - sr * 0.04) env = Math.max(0, (beat - pos) / (sr * 0.04));
      var s = 0;
      for (var k = 0; k < freqs.length; k++) {
        s += Math.sin(2 * Math.PI * freqs[k] * i / sr);
      }
      x[i] = (s / freqs.length) * 0.2 * env;
    }
    return x;
  }

  function placeAt(baseLen, snippet, offsetSamples) {
    var out = new Float32Array(Math.max(baseLen, offsetSamples + snippet.length));
    out.set(snippet, offsetSamples);
    return out;
  }

  function addAt(dest, snippet, offsetSamples, gain) {
    gain = gain == null ? 1 : gain;
    for (var i = 0; i < snippet.length; i++) {
      var j = i + offsetSamples;
      if (j >= 0 && j < dest.length) dest[j] += snippet[i] * gain;
    }
    return dest;
  }

  function whiteNoise(seconds, sr, peak) {
    sr = sr || 16000;
    var n = Math.floor(sr * seconds);
    var x = new Float32Array(n);
    var rng = mulberry32(99);
    peak = peak == null ? 0.02 : peak;
    for (var i = 0; i < n; i++) x[i] = (rng() * 2 - 1) * peak;
    return x;
  }

  return {
    speechLike: speechLike,
    speaker: speaker,
    SPEAKERS: SPEAKERS,
    mix: mix,
    silence: silence,
    concat: concat,
    sine: sine,
    chord: chord,
    musicLoop: musicLoop,
    placeAt: placeAt,
    addAt: addAt,
    whiteNoise: whiteNoise,
    normalize: normalize
  };
});
