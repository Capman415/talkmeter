#!/usr/bin/env node
'use strict';

var TMAudio = require('../js/audio.js');
var TMSynth = require('../js/synth.js');
var TMDetector = require('../js/detector.js');
var TMStore = require('../js/store.js');

var passed = 0;
var failed = 0;
var results = [];

function approx(a, b, tol) {
  return Math.abs(a - b) <= (tol == null ? 1e-6 : tol);
}

function assert(name, cond, detail) {
  if (cond) {
    passed++;
    results.push({ name: name, ok: true, detail: detail || '' });
    console.log('  PASS  ' + name + (detail ? '  — ' + detail : ''));
  } else {
    failed++;
    results.push({ name: name, ok: false, detail: detail || '' });
    console.log('  FAIL  ' + name + (detail ? '  — ' + detail : ''));
  }
}

console.log('\nTalk Meter v0.1 tests (synthetic audio)\n');

// --- Features ---
console.log('Features');
(function () {
  var sine = TMSynth.sine(440, 0.6, 16000, 0.3);
  var noise = TMSynth.whiteNoise(0.6, 16000, 0.3);
  var featsS = TMAudio.analyzeBuffer(sine, 16000);
  var featsN = TMAudio.analyzeBuffer(noise, 16000);
  var mid = featsS[Math.floor(featsS.length / 2)];
  var midN = featsN[Math.floor(featsN.length / 2)];
  assert('sine has low spectral flatness vs noise', mid.flatness < midN.flatness * 0.4,
    'sine=' + mid.flatness.toFixed(4) + ' noise=' + midN.flatness.toFixed(4));
  assert('sine F0 near 440 Hz', mid.f0 > 400 && mid.f0 < 480 && mid.f0conf > 0.3,
    'f0=' + mid.f0.toFixed(1) + ' conf=' + mid.f0conf.toFixed(2));
  assert('silence VAD is false', (function () {
    var z = TMSynth.silence(0.5, 16000);
    var f = TMAudio.analyzeBuffer(z, 16000);
    var v = 0;
    for (var i = 0; i < f.length; i++) if (TMAudio.isVoiced(f[i], 0.002, 0.5)) v++;
    return v === 0;
  })(), '');
})();

// --- Enrollment / speaker distance ---
console.log('\nVoice profile');
var teacherEnroll = TMSynth.speaker('teacher', 3.0, { seed: 11 });
var profile = TMAudio.buildProfile(teacherEnroll, 16000, 0.001, 0.5);
assert('enrollment finds voiced frames', profile.ok && profile.voiced >= 40,
  'voiced=' + profile.voiced + ' / ' + profile.total);
assert('profile stores numeric mean/std only',
  Array.isArray(profile.mean) && Array.isArray(profile.std) && profile.mean.length === 12 && !profile.samples,
  'dim=' + (profile.mean && profile.mean.length));

var teacherHold = TMSynth.speaker('teacher', 2.0, { seed: 99 }); // same speaker, different seed/contour
var childHold = TMSynth.speaker('childA', 2.0, { seed: 22 });
function medianDist(buf, prof) {
  var feats = TMAudio.analyzeBuffer(buf, 16000);
  var ds = [];
  for (var i = 0; i < feats.length; i++) {
    if (TMAudio.isVoiced(feats[i], 0.001, 0.5)) {
      ds.push(TMAudio.speakerDistance(TMAudio.embed(feats[i]), prof));
    }
  }
  ds.sort(function (a, b) { return a - b; });
  return ds.length ? ds[Math.floor(ds.length / 2)] : 999;
}
var dT = medianDist(teacherHold, profile);
var dC = medianDist(childHold, profile);
assert('same speaker is closer than a different speaker', dT < dC * 0.7 && dT < profile.threshold,
  'dTeacher=' + dT.toFixed(2) + ' dChild=' + dC.toFixed(2) + ' thr=' + profile.threshold.toFixed(2));
assert('different speaker is above the teacher threshold', dC > profile.threshold,
  'dChild=' + dC.toFixed(2) + ' thr=' + profile.threshold.toFixed(2));

// --- Detector: teacher alone does not raise chatter ---
console.log('\nDetector');
function runEngine(profile, parts) {
  // parts: [{buf, sr, startSec}] concatenated on a timeline
  var eng = new TMDetector.TalkMeterEngine({
    profile: profile,
    sensitivity: 0.5,
    noiseFloor: 0.001
  });
  eng.resetSession(0);
  // Build one long buffer with known events, process in chunks of 1600 samples (100ms)
  var sr = 16000;
  var total = 0;
  var i;
  for (i = 0; i < parts.length; i++) {
    var end = (parts[i].at || 0) + parts[i].buf.length / sr;
    if (end > total) total = end;
  }
  var timeline = new Float32Array(Math.ceil(total * sr) + 1);
  for (i = 0; i < parts.length; i++) {
    TMSynth.addAt(timeline, parts[i].buf, Math.floor((parts[i].at || 0) * sr), parts[i].gain == null ? 1 : parts[i].gain);
  }
  var hop = 1600;
  var tMs = 0;
  for (i = 0; i < timeline.length; i += hop) {
    var chunk = timeline.subarray(i, Math.min(i + hop, timeline.length));
    var copy = new Float32Array(chunk.length);
    copy.set(chunk);
    tMs = (i / sr) * 1000;
    eng.processSamples(copy, sr, tMs);
  }
  return eng;
}

var teacherOnly = runEngine(profile, [
  { buf: TMSynth.speaker('teacher', 4.0, { seed: 15 }), at: 0.2 }
]);
var s1 = teacherOnly.getState();
assert('teacher alone: teacher becomes active', s1.teacherTalkSec > 1.5,
  'talkSec=' + s1.teacherTalkSec.toFixed(2) + ' status=' + s1.status);
assert('teacher alone: no chatter starts', s1.chatterStarts === 0 && s1.gauge < 0.15,
  'starts=' + s1.chatterStarts + ' gauge=' + s1.gauge.toFixed(3));

// Teacher, then overlap with child 2s after teacher begins (teacher starts ~0.2)
var tBuf = TMSynth.speaker('teacher', 6.0, { seed: 18 });
var cBuf = TMSynth.speaker('childA', 3.0, { seed: 22 });
var overlapEng = runEngine(profile, [
  { buf: tBuf, at: 0.0, gain: 1 },
  { buf: cBuf, at: 2.0, gain: 1.05 }
]);
var s2 = overlapEng.getState();
assert('teacher + other voice: one chatter stretch (not a flicker)', s2.chatterStarts >= 1 && s2.chatterStarts <= 2,
  'starts=' + s2.chatterStarts + ' delay=' + (s2.usualDelaySec != null ? s2.usualDelaySec.toFixed(2) : 'n/a') +
  ' status=' + s2.status + ' peak=' + s2.peakGauge.toFixed(3));
assert('chatter delay is roughly 2 seconds', s2.usualDelaySec != null && s2.usualDelaySec > 1.2 && s2.usualDelaySec < 3.2,
  'delay=' + (s2.usualDelaySec != null ? s2.usualDelaySec.toFixed(2) : 'n/a'));

// Two separate chatter starts in two teacher utterances
var twoEng = runEngine(profile, [
  { buf: TMSynth.speaker('teacher', 2.2, { seed: 3 }), at: 0.0, gain: 1 },
  { buf: TMSynth.speaker('childA', 0.8, { seed: 4 }), at: 0.9, gain: 1.1 },
  // gap > hangover
  { buf: TMSynth.speaker('teacher', 2.2, { seed: 5 }), at: 4.0, gain: 1 },
  { buf: TMSynth.speaker('childB', 0.8, { seed: 6 }), at: 4.9, gain: 1.1 }
]);
var s3 = twoEng.getState();
assert('two utterances each with chatter: two starts', s3.chatterStarts === 2,
  'starts=' + s3.chatterStarts + ' delays=' + JSON.stringify(s3.delays.map(function (d) { return +d.toFixed(2); })));

// Child only (no teacher) should not count chatter
var childOnly = runEngine(profile, [
  { buf: TMSynth.speaker('childA', 3.0, { seed: 22 }), at: 0.1 }
]);
var s4 = childOnly.getState();
assert('other voice alone does not count as chatter (teacher not talking)', s4.chatterStarts === 0 && s4.gauge < 0.15,
  'starts=' + s4.chatterStarts + ' gauge=' + s4.gauge.toFixed(3) + ' status=' + s4.status);

// --- Music rejection ---
console.log('\nMusic / tone rejection');
(function () {
  var tone = TMSynth.sine(440, 2.0, 16000, 0.25);
  var feats = TMAudio.analyzeBuffer(tone, 16000);
  var score = TMAudio.musicScore(feats);
  assert('steady sine gets a high music score', score >= 0.62, 'score=' + score.toFixed(2));

  var speechFeats = TMAudio.analyzeBuffer(TMSynth.speaker('teacher', 2.0, { seed: 11 }), 16000);
  var sScore = TMAudio.musicScore(speechFeats);
  assert('speech-like signal gets a low music score', sScore < 0.5, 'score=' + sScore.toFixed(2));

  var musicEng = runEngine(profile, [
    { buf: TMSynth.speaker('teacher', 5.0, { seed: 20 }), at: 0.0, gain: 1 },
    { buf: TMSynth.sine(440, 3.0, 16000, 0.22), at: 1.5, gain: 1.0 }
  ]);
  var sm = musicEng.getState();
  assert('steady tone under teacher does not add a chatter start', sm.chatterStarts === 0,
    'starts=' + sm.chatterStarts + ' music=' + sm.musicLikely + ' gauge=' + sm.gauge.toFixed(3));

  var loopEng = runEngine(profile, [
    { buf: TMSynth.speaker('teacher', 5.0, { seed: 21 }), at: 0.0, gain: 1 },
    { buf: TMSynth.musicLoop(3.5, 16000), at: 1.2, gain: 1.0 }
  ]);
  var sl = loopEng.getState();
  assert('simple held-chord loop under teacher does not add a chatter start', sl.chatterStarts === 0,
    'starts=' + sl.chatterStarts + ' gauge=' + sl.gauge.toFixed(3));
})();

// --- Voice level (honest / rough) ---
console.log('\nRough voice level');
(function () {
  var one = runEngine(profile, [
    { buf: TMSynth.speaker('teacher', 4.0, { seed: 8 }), at: 0, gain: 1 },
    { buf: TMSynth.speaker('childA', 2.5, { seed: 22 }), at: 1.0, gain: 1.1 }
  ]);
  var many = runEngine(profile, [
    { buf: TMSynth.speaker('teacher', 4.5, { seed: 8 }), at: 0, gain: 1 },
    { buf: TMSynth.speaker('childA', 3.0, { seed: 22 }), at: 1.0, gain: 0.95 },
    { buf: TMSynth.speaker('childB', 3.0, { seed: 33 }), at: 1.0, gain: 0.95 },
    { buf: TMSynth.speaker('childC', 3.0, { seed: 44 }), at: 1.0, gain: 0.95 }
  ]);
  var l1 = one.getState().voiceTag;
  var l2 = many.getState().voiceTag;
  // This is a rough estimator. Require only that "lots of voices" is not labeled quieter than a single extra voice.
  var rank = { 'quiet': 0, 'just-you': 0, 'one-two': 1, 'a-few': 2, 'lots': 3 };
  assert('more mixed voices is not labeled below a single extra voice',
    (rank[l2] || 0) >= (rank[l1] || 0) && (one.getState().chatterStarts >= 1) && (many.getState().chatterStarts >= 1),
    'one=' + one.getState().voiceLabel + ' many=' + many.getState().voiceLabel);
})();

// --- Summary text ---
console.log('\nSummary + storage');
assert('summary: 9 starts, 8s delay',
  TMDetector.formatSummary({ chatterStarts: 9, usualDelaySec: 8 }) ===
    'Chatter started 9 times, usually about 8 seconds after you began.',
  '');
assert('summary: zero starts',
  TMDetector.formatSummary({ chatterStarts: 0 }) ===
    'Chatter never started while you were talking.',
  '');
assert('summary: 1 start, ~1s',
  TMDetector.formatSummary({ chatterStarts: 1, usualDelaySec: 1.2 }) ===
    'Chatter started 1 time, usually about 1 second after you began.',
  '');

(function () {
  var mem = TMStore._memoryStorage();
  TMStore._setStorage(mem);
  var p = TMAudio.buildProfile(TMSynth.speaker('teacher', 1.5, { seed: 11 }), 16000, 0.001, 0.5);
  p.samples = new Float32Array(100); // attacker-style slip; must be stripped
  p.audio = 'nope';
  TMStore.setProfile(p);
  var loaded = TMStore.load();
  assert('stored profile has mean vector and no audio fields',
    loaded.profile && loaded.profile.mean && loaded.profile.mean.length === 12 &&
      !loaded.profile.samples && !loaded.profile.audio && !TMStore.profileHasAudio(loaded.profile),
    '');
  TMStore.addHistory({
    durationSec: 1800,
    chatterStarts: 9,
    usualDelaySec: 8,
    peakGauge: 0.8,
    peakLevel: 'lots',
    teacherTalkSec: 600,
    endedAt: '2026-11-18T13:30:00.000Z'
  });
  var csv = TMStore.historyCsv();
  assert('CSV has header and one data row, no raw audio',
    csv.split('\n').length === 2 && csv.indexOf('chatter_starts') >= 0 && csv.indexOf('9') >= 0 && csv.indexOf('Float32') < 0,
    csv.replace(/\n/g, ' | '));
  TMStore.clearHistory();
  assert('clear history empties rows', TMStore.load().history.length === 0, '');
})();

// --- Noise floor ---
console.log('\nCalibration');
(function () {
  var quiet = TMSynth.whiteNoise(2.0, 16000, 0.004);
  var floor = TMAudio.noiseFloorFromSamples(quiet, 16000);
  assert('noise floor from quiet noise is in a sensible range', floor > 0.0003 && floor < 0.02,
    'floor=' + floor.toFixed(5));
})();

console.log('\n' + passed + ' passed, ' + failed + ' failed\n');
process.exit(failed ? 1 : 0);
