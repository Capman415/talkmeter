/* Live talk / chatter detector. Operates on features only — no audio is kept. */
(function (root, factory) {
  var dep = function () {
    return typeof TMAudio !== 'undefined' ? TMAudio : require('./audio.js');
  };
  if (typeof module === 'object' && module.exports) module.exports = factory(dep());
  else root.TMDetector = factory(dep());
})(typeof self !== 'undefined' ? self : this, function (TMAudio) {
  'use strict';

  var RECENT_MAX = 120; // 1.2s at 10ms hop
  var TEACHER_HANG_MS = 800;
  var LEVEL_WINDOW = 100;

  function median(arr) {
    if (!arr || !arr.length) return null;
    var a = arr.slice().sort(function (x, y) { return x - y; });
    var mid = Math.floor(a.length / 2);
    return a.length % 2 ? a[mid] : (a[mid - 1] + a[mid]) / 2;
  }

  function formatSummary(s) {
    var n = s.chatterStarts || 0;
    if (n === 0) {
      return 'Chatter never started while you were talking.';
    }
    var times = n === 1 ? '1 time' : n + ' times';
    if (s.usualDelaySec == null) {
      return 'Chatter started ' + times + ' while you were talking.';
    }
    var d = Math.round(s.usualDelaySec);
    var delay;
    if (d <= 1) delay = 'usually about 1 second after you began';
    else delay = 'usually about ' + d + ' seconds after you began';
    return 'Chatter started ' + times + ', ' + delay + '.';
  }

  function extraPitch(feat, teacherF0) {
    if (!feat.pitches || !feat.pitches.length || !teacherF0) return false;
    for (var i = 0; i < feat.pitches.length; i++) {
      var hz = feat.pitches[i];
      var ratio = hz / teacherF0;
      var near = Math.round(ratio);
      if (near < 1) near = 1;
      var harmonic = Math.abs(ratio - near) < 0.08 || Math.abs(teacherF0 / hz - Math.round(teacherF0 / hz)) < 0.08;
      if (!harmonic && Math.abs(hz - teacherF0) / teacherF0 > 0.12) return true;
    }
    return false;
  }

  function roughVoiceLevel(recent) {
    var i, f0s = [];
    var busy = 0, n = 0, harm = 0, harmN = 0;
    var start = Math.max(0, recent.length - LEVEL_WINDOW);
    for (i = start; i < recent.length; i++) {
      var f = recent[i];
      if (!f.voiced || f.music || f.kind === 'teacher' || f.kind === 'quiet') continue;
      n++;
      if (f.f0conf > 0.25 && f.f0 > 0) f0s.push(Math.round(12 * Math.log2(f.f0 / 110) * 2) / 2);
      if (f.f0conf < 0.35 && f.rms > 0) busy++;
      if (f.f0conf > 0) { harm += f.f0conf; harmN++; }
    }
    if (n < 4) return { label: '1–2 voices', tag: 'one-two', unique: 0, rough: true };
    var counts = {};
    for (i = 0; i < f0s.length; i++) {
      var k = String(f0s[i]);
      counts[k] = (counts[k] || 0) + 1;
    }
    var unique = 0;
    for (k in counts) {
      if (counts[k] >= 3) unique++;
    }
    var meanHarm = harmN ? harm / harmN : 1;
    var noisyCrowd = (busy / n > 0.55 && meanHarm < 0.4);
    var label, tag;
    if (noisyCrowd || unique >= 5) { label = 'lots'; tag = 'lots'; }
    else if (unique >= 3) { label = 'a few'; tag = 'a-few'; }
    else { label = '1–2 voices'; tag = 'one-two'; }
    return { label: label, tag: tag, unique: unique, rough: true };
  }

  function TalkMeterEngine(opts) {
    opts = opts || {};
    this.profile = opts.profile || null;
    this.sensitivity = opts.sensitivity == null ? 0.5 : +opts.sensitivity;
    this.noiseFloor = opts.noiseFloor == null ? 0.002 : +opts.noiseFloor;
    this.pusher = new TMAudio.FramePusher();
    this.prevMag = null;
    this.recent = [];
    this.nowMs = 0;
    this.teacherActive = false;
    this.teacherLastMs = -1e12;
    this.teacherStartMs = null;
    this.chatterActive = false;
    this.otherRunMs = 0;
    this.quietRunMs = 0;
    this.chatterStarts = 0;
    this.delays = [];
    this.utteranceHadChatter = false;
    this.gauge = 0;
    this.peakGauge = 0;
    this.voiceInfo = { label: 'quiet', tag: 'quiet', unique: 0, rough: true };
    this.sessionStartMs = null;
    this.teacherTalkMs = 0;
    this.lastTickMs = null;
    this.musicLikely = false;
    this.status = 'idle';
    this.debug = {};
    this.samplesProcessed = 0;
  }

  TalkMeterEngine.prototype.setProfile = function (p) { this.profile = p; };
  TalkMeterEngine.prototype.setSensitivity = function (s) { this.sensitivity = +s; };
  TalkMeterEngine.prototype.setNoiseFloor = function (n) { this.noiseFloor = +n; };

  TalkMeterEngine.prototype.resetSession = function (nowMs) {
    this.nowMs = nowMs || 0;
    this.teacherActive = false;
    this.teacherLastMs = -1e12;
    this.teacherStartMs = null;
    this.chatterActive = false;
    this.otherRunMs = 0;
    this.quietRunMs = 0;
    this.chatterStarts = 0;
    this.delays = [];
    this.utteranceHadChatter = false;
    this.gauge = 0;
    this.peakGauge = 0;
    this.voiceInfo = { label: 'quiet', tag: 'quiet', unique: 0, rough: true };
    this.sessionStartMs = this.nowMs;
    this.teacherTalkMs = 0;
    this.lastTickMs = this.nowMs;
    this.musicLikely = false;
    this.status = 'listening';
    this.recent = [];
    this.prevMag = null;
    this.pusher.reset();
    this.samplesProcessed = 0;
    this.embHist = [];
    this.teacherRunMs = 0;
    this.chatterOffMs = 0;
  };

  TalkMeterEngine.prototype._debounceOn = function () {
    return 280 - this.sensitivity * 140; // 280ms … 140ms
  };
  TalkMeterEngine.prototype._debounceOff = function () {
    return 260 + (1 - this.sensitivity) * 80;
  };

  TalkMeterEngine.prototype.processSamples = function (samples, sr, nowMs) {
    var frames = this.pusher.push(samples, sr);
    var hopMs = (TMAudio.HOP / TMAudio.TARGET_SR) * 1000;
    var t = nowMs != null ? nowMs : this.nowMs;
    for (var i = 0; i < frames.length; i++) {
      t += hopMs;
      this._processFrame(frames[i], t);
    }
    this.nowMs = t;
    this.samplesProcessed += samples.length;
    return this.getState();
  };

  TalkMeterEngine.prototype._processFrame = function (frame, nowMs) {
    var dt = this.lastTickMs == null ? 10 : Math.max(1, nowMs - this.lastTickMs);
    this.lastTickMs = nowMs;
    this.nowMs = nowMs;

    var feat = TMAudio.extractFrame(frame, TMAudio.TARGET_SR, this.prevMag);
    this.prevMag = feat._mag;
    feat._mag = undefined;

    var voiced = TMAudio.isVoiced(feat, this.noiseFloor, this.sensitivity);
    feat.voiced = voiced;

    this.recent.push(feat);
    if (this.recent.length > RECENT_MAX) this.recent.shift();

    var mScore = TMAudio.musicScore(this.recent);
    this.musicLikely = mScore >= 0.62;
    feat.music = this.musicLikely && mScore >= 0.62;

    var dist = null;
    var kind = 'quiet';
    if (!voiced) {
      kind = 'quiet';
      this.embHist = [];
    } else if (this.profile && this.profile.mean) {
      var emb = TMAudio.embed(feat);
      this.embHist = this.embHist || [];
      this.embHist.push(emb);
      if (this.embHist.length > 5) this.embHist.shift();
      var used = this.embHist.length >= 2 ? TMAudio.averageEmbeds(this.embHist) : emb;
      dist = TMAudio.speakerDistance(used, this.profile);
      var thr = this.profile.threshold != null ? this.profile.threshold : 2.6;
      var otherThr = thr * (1.15 + (0.5 - this.sensitivity) * 0.15);
      var f0ok = TMAudio.f0Compatible(feat, this.profile, 8);
      var tonalFrame = feat.flatness < 0.012;
      // Close timbre match wins even if F0 jumps an octave; pitch is used to reject far voices.
      if (dist <= thr) kind = 'teacher';
      else if (feat.music || tonalFrame) kind = 'music';
      else if (!f0ok || dist >= otherThr) kind = 'other';
      else kind = 'uncertain';
    } else if (feat.music || feat.flatness < 0.012) {
      kind = 'music';
    } else {
      kind = 'other';
    }
    feat.kind = kind;
    feat.dist = dist;

    if (kind === 'teacher') {
      this.teacherRunMs = (this.teacherRunMs || 0) + dt;
      if (this.teacherRunMs >= 70) this.teacherLastMs = nowMs;
    } else {
      this.teacherRunMs = 0;
    }

    var wasTeacher = this.teacherActive;
    this.teacherActive = (nowMs - this.teacherLastMs) <= TEACHER_HANG_MS;
    if (this.teacherActive && !wasTeacher) {
      this.teacherStartMs = nowMs;
      this.utteranceHadChatter = false;
    }
    if (!this.teacherActive) {
      this.teacherStartMs = null;
      this.utteranceHadChatter = false;
    }
    if (this.teacherActive) this.teacherTalkMs += dt;

    var overlapCue = false;
    if (this.teacherActive && voiced && !feat.music) {
      var f0ref = (kind === 'teacher' && feat.f0) ? feat.f0 : (this.profile && this.profile.meanF0);
      if (f0ref && extraPitch(feat, f0ref)) overlapCue = true;
    }

    var otherNow = false;
    if (this.teacherActive && voiced && kind !== 'music') {
      // A teacher-matched frame is never chatter by itself, even if a second pitch slips through.
      if (kind === 'other') otherNow = true;
      else if (kind === 'uncertain' && (this.sensitivity >= 0.7 || overlapCue)) otherNow = true;
    }

    // Overlap often interleaves teacher + other frames, so count "other"
    // in a short window instead of requiring a long consecutive run.
    var win = 30;
    var otherCount = 0;
    var i0 = Math.max(0, this.recent.length - win);
    var ri;
    for (ri = i0; ri < this.recent.length; ri++) {
      var rk = this.recent[ri].kind;
      if (rk === 'other') otherCount += 1;
      else if (rk === 'uncertain' && (this.sensitivity >= 0.7 || this.recent[ri].overlap)) otherCount += 0.6;
    }
    var onCount = 8 - Math.round(this.sensitivity * 3); // 8 at 0, ~7 at 0.5, 5 at 1
    if (onCount < 5) onCount = 5;
    var offCount = 2;

    var wasChatter = this.chatterActive;
    var offHoldMs = 550 + (1 - this.sensitivity) * 200;
    if (!this.teacherActive) {
      this.chatterActive = false;
      this.otherRunMs = 0;
      this.quietRunMs = 0;
      this.chatterOffMs = 0;
    } else if (!wasChatter && otherCount >= onCount) {
      this.chatterActive = true;
      this.chatterOffMs = 0;
    } else if (wasChatter && otherCount <= offCount) {
      this.chatterOffMs = (this.chatterOffMs || 0) + dt;
      if (this.chatterOffMs >= offHoldMs) this.chatterActive = false;
    } else if (wasChatter) {
      this.chatterOffMs = 0;
    }
    this.otherRunMs = otherCount * 10;
    this.quietRunMs = otherNow ? 0 : this.quietRunMs + dt;

    if (this.chatterActive && !wasChatter) {
      this.chatterStarts += 1;
      if (!this.utteranceHadChatter && this.teacherStartMs != null) {
        var delay = (nowMs - this.teacherStartMs) / 1000;
        if (delay >= 0) this.delays.push(delay);
        this.utteranceHadChatter = true;
      }
    }

    if (this.chatterActive) {
      this.voiceInfo = roughVoiceLevel(this.recent);
    } else if (!this.teacherActive) {
      this.voiceInfo = { label: 'quiet', tag: 'quiet', unique: 0, rough: true };
    } else {
      this.voiceInfo = { label: 'just you', tag: 'just-you', unique: 1, rough: true };
    }

    var target = 0;
    if (this.chatterActive) {
      if (this.voiceInfo.tag === 'lots') target = 0.92;
      else if (this.voiceInfo.tag === 'a-few') target = 0.64;
      else target = 0.38;
      var boost = Math.min(0.15, Math.max(0, (feat.rms / 0.08) * 0.08));
      target = Math.min(1, target + boost);
    }
    var alpha = target > this.gauge ? 0.18 : 0.07;
    this.gauge += (target - this.gauge) * alpha;
    if (this.gauge < 0.008 && target === 0) this.gauge = 0;
    if (this.gauge > this.peakGauge) this.peakGauge = this.gauge;

    if (!this.teacherActive) this.status = this.musicLikely ? 'music' : (voiced ? 'other-only' : 'listening');
    else if (this.chatterActive) this.status = 'chatter';
    else this.status = 'teacher';

    this.debug = {
      kind: kind,
      dist: dist,
      voiced: voiced,
      musicScore: mScore,
      music: feat.music,
      overlap: overlapCue,
      rms: feat.rms,
      f0: feat.f0,
      f0conf: feat.f0conf
    };
  };

  TalkMeterEngine.prototype.getState = function () {
    return {
      gauge: this.gauge,
      peakGauge: this.peakGauge,
      teacherActive: this.teacherActive,
      chatterActive: this.chatterActive,
      chatterStarts: this.chatterStarts,
      delays: this.delays.slice(),
      usualDelaySec: median(this.delays),
      voiceLabel: this.voiceInfo.label,
      voiceTag: this.voiceInfo.tag,
      voiceRough: true,
      status: this.status,
      musicLikely: this.musicLikely,
      sessionSec: this.sessionStartMs == null ? 0 : (this.nowMs - this.sessionStartMs) / 1000,
      teacherTalkSec: this.teacherTalkMs / 1000,
      debug: this.debug
    };
  };

  TalkMeterEngine.prototype.endSession = function (endedAt) {
    var state = this.getState();
    var summary = {
      durationSec: state.sessionSec,
      chatterStarts: state.chatterStarts,
      delays: state.delays,
      usualDelaySec: state.usualDelaySec,
      peakGauge: state.peakGauge,
      peakLevel: state.peakGauge >= 0.75 ? 'lots' : state.peakGauge >= 0.5 ? 'a few' : state.peakGauge >= 0.2 ? '1–2 voices' : 'quiet',
      teacherTalkSec: state.teacherTalkSec,
      endedAt: endedAt || new Date().toISOString()
    };
    summary.text = formatSummary(summary);
    this.status = 'ended';
    return summary;
  };

  return {
    TalkMeterEngine: TalkMeterEngine,
    formatSummary: formatSummary,
    median: median,
    roughVoiceLevel: roughVoiceLevel
  };
});
