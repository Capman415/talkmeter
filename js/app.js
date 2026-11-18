/* Talk Meter v0.2 — UI, mic, enrollment, SpeechNet gate, live session. Audio never leaves the page. */
(function () {
  'use strict';

  var ENROLL_SEC = 10;
  var CAL_SEC = 3;
  var SIZES = ['compact', 'medium', 'large', 'full'];
  var POSITIONS = ['bottom-right', 'bottom-left', 'top-right', 'top-left'];

  var state = {
    screen: 'setup',
    stream: null,
    ctx: null,
    processor: null,
    source: null,
    engine: null,
    sessionOrigin: 0,
    raf: 0,
    lastLive: null,
    lastSummary: null,
    collecting: null,
    demo: false,
    speechNet: null
  };

  function $(id) { return document.getElementById(id); }

  function sensLabel(v) {
    if (v < 0.25) return 'Low';
    if (v < 0.4) return 'Low–medium';
    if (v < 0.6) return 'Medium';
    if (v < 0.8) return 'Medium–high';
    return 'High';
  }

  function formatDelay(sec) {
    if (sec == null) return '—';
    var d = Math.round(sec);
    if (d <= 1) return 'about 1 s';
    return 'about ' + d + ' s';
  }

  function statusText(s) {
    if (!s) return 'Listening';
    if (s.status === 'chatter') return 'Chatter!';
    if (s.status === 'teacher') return "You're talking";
    if (s.status === 'music') return 'Steady sound ignored';
    if (s.status === 'other-only') return 'Voices (you’re quiet)';
    return 'Listening';
  }

  function applyChrome() {
    var data = TMStore.load();
    var size = data.settings.size || 'compact';
    var pos = data.settings.position || 'bottom-right';
    var character = TMCharacters.isAvailable(data.settings.character) ? data.settings.character : 'gauge';
    var hideHear = data.settings.showHear === false;
    document.body.className = 'screen-' + state.screen + ' size-' + size + ' pos-' + pos + (hideHear ? ' hide-hear' : '');
    document.body.setAttribute('data-character', character);
    var stage = $('character-stage');
    if (stage) TMCharacters.mount(stage, character);
  }

  function showScreen(name) {
    state.screen = name;
    applyChrome();
  }

  function fillCharacterSelect() {
    var sel = $('character-select');
    if (!sel) return;
    var data = TMStore.load();
    sel.innerHTML = '';
    TMCharacters.catalog.forEach(function (c) {
      var opt = document.createElement('option');
      opt.value = c.id;
      opt.textContent = c.label;
      opt.disabled = !c.available;
      if (c.id === (data.settings.character || 'gauge')) opt.selected = true;
      sel.appendChild(opt);
    });
  }

  function updateEnrollPill() {
    var data = TMStore.load();
    var pill = $('enroll-pill');
    var re = $('btn-reenroll');
    var start = $('btn-start');
    if (data.profile && data.profile.mean) {
      pill.textContent = 'Voice profile saved on this browser';
      pill.classList.remove('off');
      if (re) re.hidden = false;
      if (start) start.disabled = false;
    } else {
      pill.textContent = 'Not enrolled yet';
      pill.classList.add('off');
      if (re) re.hidden = true;
      if (start) start.disabled = true;
    }
  }

  function updateSensLabels() {
    var data = TMStore.load();
    var v = data.settings.sensitivity;
    if (v == null) v = 0.5;
    $('sensitivity').value = String(Math.round(v * 100));
    $('sensitivity-live').value = String(Math.round(v * 100));
    $('sens-label').textContent = sensLabel(v);
  }

  function setStatus(id, msg, isErr) {
    var el = $(id);
    if (!el) return;
    el.textContent = msg || '';
    el.className = 'status-line' + (isErr ? ' err' : '');
  }

  function setProgress(id, frac, show) {
    var el = $(id);
    if (!el) return;
    el.hidden = !show;
    var bar = el.querySelector('span');
    if (bar) bar.style.width = Math.max(0, Math.min(100, frac * 100)).toFixed(1) + '%';
  }

  async function listMics(selectedId) {
    var sel = $('mic-select');
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
      setStatus('mic-status', 'This browser can’t list microphones.', true);
      return;
    }
    var devices = await navigator.mediaDevices.enumerateDevices();
    var mics = devices.filter(function (d) { return d.kind === 'audioinput'; });
    var current = selectedId != null ? selectedId : TMStore.load().settings.micId;
    sel.innerHTML = '';
    var def = document.createElement('option');
    def.value = '';
    def.textContent = 'Default microphone';
    sel.appendChild(def);
    mics.forEach(function (d, i) {
      var opt = document.createElement('option');
      opt.value = d.deviceId;
      opt.textContent = d.label || ('Microphone ' + (i + 1));
      if (d.deviceId && d.deviceId === current) opt.selected = true;
      sel.appendChild(opt);
    });
    if (!current) sel.value = '';
    var named = mics.filter(function (d) { return d.label; }).length;
    if (!named && mics.length) {
      setStatus('mic-status', 'Allow the microphone once so this list can show names.');
    } else {
      setStatus('mic-status', mics.length ? (mics.length + ' microphone(s) found.') : 'No microphones found.');
    }
  }

  function audioConstraints(deviceId) {
    var audio = {
      echoCancellation: true,
      noiseSuppression: false,
      autoGainControl: false
    };
    if (deviceId) audio.deviceId = { exact: deviceId };
    return { audio: audio };
  }

  async function getMicStream(deviceId) {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error('This browser does not support microphones.');
    }
    return navigator.mediaDevices.getUserMedia(audioConstraints(deviceId));
  }

  function stopStream(stream) {
    if (!stream) return;
    stream.getTracks().forEach(function (t) { t.stop(); });
  }

  function closeCapture() {
    try { if (state.processor) state.processor.disconnect(); } catch (e) {}
    try { if (state.source) state.source.disconnect(); } catch (e) {}
    try { if (state.ctx && state.ctx.state !== 'closed') state.ctx.close(); } catch (e) {}
    stopStream(state.stream);
    state.processor = null;
    state.source = null;
    state.ctx = null;
    state.stream = null;
    state.collecting = null;
  }

  function attachProcessor(stream, onAudio) {
    var AC = window.AudioContext || window.webkitAudioContext;
    var ctx = new AC();
    var source = ctx.createMediaStreamSource(stream);
    var processor = ctx.createScriptProcessor(2048, 1, 1);
    var mute = ctx.createGain();
    mute.gain.value = 0;
    processor.onaudioprocess = function (ev) {
      var input = ev.inputBuffer.getChannelData(0);
      var copy = new Float32Array(input.length);
      copy.set(input);
      onAudio(copy, ctx.sampleRate);
    };
    source.connect(processor);
    processor.connect(mute);
    mute.connect(ctx.destination);
    state.ctx = ctx;
    state.source = source;
    state.processor = processor;
    state.stream = stream;
    if (ctx.state === 'suspended' && ctx.resume) ctx.resume();
    return ctx;
  }

  async function collectSeconds(seconds, progressId, deviceId) {
    var stream = await getMicStream(deviceId || TMStore.load().settings.micId || '');
    return new Promise(function (resolve, reject) {
      var chunks = [];
      var sr = 0;
      var started = performance.now();
      try {
        attachProcessor(stream, function (buf, sampleRate) {
          sr = sampleRate;
          chunks.push(buf);
          var t = (performance.now() - started) / 1000;
          setProgress(progressId, Math.min(1, t / seconds), true);
        });
      } catch (err) {
        stopStream(stream);
        reject(err);
        return;
      }
      setTimeout(function () {
        closeCapture();
        var n = 0, i;
        for (i = 0; i < chunks.length; i++) n += chunks[i].length;
        var samples = new Float32Array(n);
        var o = 0;
        for (i = 0; i < chunks.length; i++) {
          samples.set(chunks[i], o);
          o += chunks[i].length;
        }
        setProgress(progressId, 1, true);
        resolve({ samples: samples, sr: sr || 48000 });
      }, seconds * 1000 + 80);
    });
  }

  async function onCalibrate() {
    setStatus('cal-status', 'Listening to the room… stay quiet.');
    setProgress('cal-progress', 0, true);
    try {
      var rec = await collectSeconds(CAL_SEC, 'cal-progress', $('mic-select').value);
      var floor = TMAudio.noiseFloorFromSamples(rec.samples, rec.sr);
      TMStore.updateSettings({
        noiseFloor: floor,
        calibratedAt: new Date().toISOString()
      });
      setStatus('cal-status', 'Room noise saved. Floor is ' + floor.toFixed(4) + '.');
    } catch (err) {
      setProgress('cal-progress', 0, false);
      setStatus('cal-status', (err && err.message) || 'Could not use the microphone.', true);
    }
  }

  async function onEnroll() {
    setStatus('enroll-status', 'Recording… read the script in your normal voice.');
    setProgress('enroll-progress', 0, true);
    try {
      var rec = await collectSeconds(ENROLL_SEC, 'enroll-progress', $('mic-select').value);
      var data = TMStore.load();
      var profile = TMAudio.buildProfile(rec.samples, rec.sr, data.settings.noiseFloor || 0.002, data.settings.sensitivity);
      // Drop any raw samples immediately.
      rec.samples = null;
      if (!profile.ok || profile.voiced < 80) {
        setStatus('enroll-status', 'I didn’t catch enough of your voice. Try again a bit closer to the mic, and talk for the full 10 seconds.', true);
        return;
      }
      profile.enrolledAt = new Date().toISOString();
      TMStore.setProfile(profile);
      updateEnrollPill();
      setStatus('enroll-status', 'Saved a numeric profile from ' + profile.voiced + ' voiced frames. Nothing was recorded.');
    } catch (err) {
      setProgress('enroll-progress', 0, false);
      setStatus('enroll-status', (err && err.message) || 'Could not use the microphone.', true);
    }
  }

  function paintLive(s) {
    state.lastLive = s;
    TMCharacters.update(s);
    $('stat-starts').textContent = String(s.chatterStarts || 0);
    var voice = s.chatterActive ? (s.voiceLabel || '1–2 voices') : (s.teacherActive ? 'just you' : 'quiet');
    $('stat-voices').textContent = voice;
    $('stat-delay').textContent = formatDelay(s.usualDelaySec);
    var pill = $('status-pill');
    pill.textContent = statusText(s);
    pill.setAttribute('data-status', s.status || 'listening');
    var hear = $('hear-pill');
    if (hear) {
      hear.textContent = s.speechHeard || 'Quiet';
      hear.setAttribute('data-hear', s.speechLabel || 'quiet');
    }
  }

  function tickLive() {
    if (state.screen !== 'live' || !state.engine) return;
    paintLive(state.engine.getState());
    state.raf = requestAnimationFrame(tickLive);
  }

  async function startClass() {
    var data = TMStore.load();
    if (!data.profile || !data.profile.mean) {
      setStatus('setup-status', 'Record your voice first.', true);
      return;
    }
    if (state.demo) {
      state.engine = new TMDetector.TalkMeterEngine({
        profile: data.profile,
        sensitivity: data.settings.sensitivity,
        noiseFloor: data.settings.noiseFloor || 0.002,
        speechNet: state.speechNet
      });
      state.engine.resetSession(0);
      showScreen('live');
      paintLive(state.engine.getState());
      return;
    }
    try {
      var stream = await getMicStream($('mic-select').value || data.settings.micId || '');
      await listMics($('mic-select').value);
      state.engine = new TMDetector.TalkMeterEngine({
        profile: data.profile,
        sensitivity: data.settings.sensitivity,
        noiseFloor: data.settings.noiseFloor || 0.002,
        speechNet: state.speechNet
      });
      state.sessionOrigin = performance.now();
      state.engine.resetSession(0);
      attachProcessor(stream, function (buf, sr) {
        if (!state.engine || state.screen !== 'live') return;
        state.engine.processSamples(buf, sr, performance.now() - state.sessionOrigin);
      });
      showScreen('live');
      tickLive();
    } catch (err) {
      setStatus('setup-status', (err && err.message) || 'Could not open the microphone.', true);
    }
  }

  function endClass() {
    if (state.raf) cancelAnimationFrame(state.raf);
    state.raf = 0;
    var summary;
    if (state.engine) summary = state.engine.endSession();
    else summary = TMDetector.formatSummary ? { text: TMDetector.formatSummary({ chatterStarts: 0 }), chatterStarts: 0, durationSec: 0 } : { text: '', chatterStarts: 0 };
    closeCapture();
    if (summary && !state.demo) TMStore.addHistory(summary);
    state.lastSummary = summary;
    renderSummary(summary, TMStore.load().history);
    showScreen('summary');
  }

  function renderSummary(summary, history) {
    $('summary-text').textContent = (summary && summary.text) || TMDetector.formatSummary(summary || { chatterStarts: 0 });
    var d = $('summary-details');
    var dur = summary ? summary.durationSec : 0;
    var mins = Math.floor(dur / 60);
    var secs = Math.round(dur % 60);
    var parts = [
      'Length: ' + mins + ' min ' + secs + ' s',
      'Chatter starts: ' + ((summary && summary.chatterStarts) || 0),
      'Usual delay: ' + (summary && summary.usualDelaySec != null ? formatDelay(summary.usualDelaySec) : '—'),
      'You were talking about ' + Math.round((summary && summary.teacherTalkSec) || 0) + ' s'
    ];
    d.innerHTML = parts.map(function (p) { return '<div class="status-line">' + p + '</div>'; }).join('');

    var rows = history || [];
    var tb = $('hist-table').querySelector('tbody');
    tb.innerHTML = '';
    var bars = $('trend-bars');
    bars.innerHTML = '';
    var max = 1;
    rows.forEach(function (r) { if (r.chatterStarts > max) max = r.chatterStarts; });
    rows.forEach(function (r) {
      var tr = document.createElement('tr');
      var when = '';
      try { when = new Date(r.endedAt).toLocaleString(); } catch (e) { when = r.endedAt || ''; }
      var delay = r.usualDelaySec == null ? '—' : formatDelay(r.usualDelaySec);
      var mm = Math.floor((r.durationSec || 0) / 60);
      var ss = Math.round((r.durationSec || 0) % 60);
      var ssT = (ss < 10 ? '0' : '') + ss;
      tr.innerHTML = '<td>' + when + '</td><td>' + r.chatterStarts + '</td><td>' + delay + '</td><td>' + mm + ':' + ssT + '</td>';
      tb.appendChild(tr);
      var i = document.createElement('i');
      i.style.height = (Math.max(4, (r.chatterStarts / max) * 64)) + 'px';
      i.title = r.chatterStarts + ' starts';
      bars.appendChild(i);
    });
    if (!rows.length) {
      var tr = document.createElement('tr');
      tr.innerHTML = '<td colspan="4">No saved classes yet.</td>';
      tb.appendChild(tr);
    }
  }

  function exportCsv() {
    var csv = TMStore.historyCsv();
    var blob = new Blob([csv], { type: 'text/csv' });
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'talk-meter-history.csv';
    a.click();
    setTimeout(function () { URL.revokeObjectURL(a.href); }, 1000);
  }

  function cycle(arr, cur) {
    var i = arr.indexOf(cur);
    return arr[(i + 1) % arr.length];
  }

  function toggleFullscreen() {
    var d = document.documentElement;
    if (!document.fullscreenElement) {
      var req = d.requestFullscreen || d.webkitRequestFullscreen;
      if (req) req.call(d).catch(function () {});
    } else if (document.exitFullscreen) {
      document.exitFullscreen();
    }
  }

  function bind() {
    $('btn-mic-test').addEventListener('click', function () {
      getMicStream($('mic-select').value).then(function (stream) {
        stopStream(stream);
        return listMics($('mic-select').value);
      }).catch(function (err) {
        setStatus('mic-status', (err && err.message) || 'Microphone permission was not granted.', true);
      });
    });
    $('mic-select').addEventListener('change', function () {
      TMStore.updateSettings({ micId: $('mic-select').value });
    });
    $('btn-calibrate').addEventListener('click', onCalibrate);
    $('btn-enroll').addEventListener('click', onEnroll);
    $('btn-reenroll').addEventListener('click', function () {
      if (confirm('Replace the voice profile saved on this browser? The old numbers will be deleted. No recordings are kept.')) {
        onEnroll();
      }
    });
    function onSens(ev) {
      var v = ev.target.value / 100;
      TMStore.updateSettings({ sensitivity: v });
      updateSensLabels();
      if (state.engine) state.engine.setSensitivity(v);
    }
    $('sensitivity').addEventListener('input', onSens);
    $('sensitivity-live').addEventListener('input', onSens);
    var hearBox = $('show-hear');
    if (hearBox) {
      hearBox.checked = TMStore.load().settings.showHear !== false;
      hearBox.addEventListener('change', function () {
        TMStore.updateSettings({ showHear: !!hearBox.checked });
        applyChrome();
      });
    }
    $('character-select').addEventListener('change', function () {
      var id = $('character-select').value;
      if (!TMCharacters.isAvailable(id)) {
        $('character-select').value = 'gauge';
        return;
      }
      TMStore.updateSettings({ character: id });
      applyChrome();
    });
    $('btn-start').addEventListener('click', startClass);
    $('btn-end').addEventListener('click', endClass);
    $('btn-new-class').addEventListener('click', startClass);
    $('btn-export').addEventListener('click', exportCsv);
    $('btn-clear-hist').addEventListener('click', function () {
      if (confirm('Clear the saved class trend on this browser?')) {
        TMStore.clearHistory();
        renderSummary(state.lastSummary, []);
      }
    });
    $('btn-back-setup').addEventListener('click', function () {
      showScreen('setup');
    });
    $('btn-settings').addEventListener('click', function () {
      if (state.engine && !confirm('Leave this class and go back to setup? The current class will be ended and saved on this browser.')) return;
      if (state.engine) endClass();
      showScreen('setup');
    });
    $('btn-size').addEventListener('click', function () {
      var next = cycle(SIZES, TMStore.load().settings.size || 'compact');
      TMStore.updateSettings({ size: next });
      applyChrome();
    });
    $('btn-corner').addEventListener('click', function () {
      var next = cycle(POSITIONS, TMStore.load().settings.position || 'bottom-right');
      TMStore.updateSettings({ position: next });
      applyChrome();
    });
    $('btn-fullscreen').addEventListener('click', toggleFullscreen);
    $('btn-live-fs').addEventListener('click', toggleFullscreen);
    document.addEventListener('keydown', function (e) {
      if (e.key === 'f' || e.key === 'F') {
        if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.isContentEditable)) return;
        toggleFullscreen();
      }
    });
  }

  function demoProfile() {
    return {
      ok: true,
      mean: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      std: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
      snapshots: [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
      dim: 12,
      frames: 200,
      voiced: 200,
      total: 200,
      durationSec: 10,
      threshold: 1.8,
      meanF0: 120,
      enrolledAt: '2026-11-18T13:00:00.000Z'
    };
  }

  /* URL shot/demo mode for screenshots and review — no microphone used. */
  function applyShot(name) {
    state.demo = true;
    var bd = $('backdrop');
    var hist = [
      { endedAt: '2026-11-12T14:05:00.000Z', durationSec: 1680, chatterStarts: 6, usualDelaySec: 7, peakGauge: 0.6, peakLevel: 'a few', teacherTalkSec: 500 },
      { endedAt: '2026-11-13T14:10:00.000Z', durationSec: 1740, chatterStarts: 9, usualDelaySec: 8, peakGauge: 0.82, peakLevel: 'lots', teacherTalkSec: 540 },
      { endedAt: '2026-11-17T14:00:00.000Z', durationSec: 1800, chatterStarts: 4, usualDelaySec: 11, peakGauge: 0.4, peakLevel: '1–2 voices', teacherTalkSec: 610 }
    ];
    if (name === 'setup') {
      TMStore.setProfile(null);
      TMStore.clearHistory();
      updateEnrollPill();
      showScreen('setup');
      return;
    }
    if (name === 'setup-enrolled') {
      TMStore.setProfile(demoProfile());
      updateEnrollPill();
      setStatus('enroll-status', 'Saved a numeric profile from 200 voiced frames. Nothing was recorded.');
      setStatus('cal-status', 'Room noise saved.');
      showScreen('setup');
      return;
    }
    TMStore.setProfile(demoProfile());
    var live = {
      gauge: 0,
      peakGauge: 0,
      teacherActive: false,
      chatterActive: false,
      chatterStarts: 0,
      delays: [],
      usualDelaySec: null,
      voiceLabel: 'quiet',
      voiceTag: 'quiet',
      status: 'listening'
    };
    if (name === 'compact-quiet') {
      bd.hidden = false;
      bd.className = 'board';
      TMStore.updateSettings({ size: 'compact', position: 'bottom-right' });
      showScreen('live');
      paintLive(live);
      return;
    }
    if (name === 'compact-few') {
      bd.hidden = false;
      bd.className = 'board';
      TMStore.updateSettings({ size: 'compact', position: 'bottom-right' });
      showScreen('live');
      paintLive({
        gauge: 0.58, peakGauge: 0.58, teacherActive: true, chatterActive: true,
        chatterStarts: 4, delays: [7, 8, 6, 9], usualDelaySec: 7.5,
        voiceLabel: 'a few', voiceTag: 'a-few', status: 'chatter',
        speechHeard: 'Voices', speechLabel: 'speech', speechGate: true
      });
      return;
    }
    if (name === 'compact-lots') {
      bd.hidden = false;
      bd.className = 'board';
      TMStore.updateSettings({ size: 'compact', position: 'bottom-right' });
      showScreen('live');
      paintLive({
        gauge: 0.92, peakGauge: 0.92, teacherActive: true, chatterActive: true,
        chatterStarts: 9, delays: [5, 8, 8, 10], usualDelaySec: 8,
        voiceLabel: 'lots', voiceTag: 'lots', status: 'chatter',
        speechHeard: 'Voices', speechLabel: 'speech', speechGate: true
      });
      return;
    }
    if (name === 'compact-music') {
      bd.hidden = false;
      bd.className = 'board';
      TMStore.updateSettings({ size: 'compact', position: 'bottom-right', showHear: true });
      showScreen('live');
      paintLive({
        gauge: 0, peakGauge: 0, teacherActive: true, chatterActive: false,
        chatterStarts: 0, delays: [], usualDelaySec: null,
        voiceLabel: 'just you', voiceTag: 'just-you', status: 'teacher',
        speechHeard: 'Music', speechLabel: 'music', speechGate: false
      });
      return;
    }
    if (name === 'compact-clapping') {
      bd.hidden = false;
      bd.className = 'board';
      TMStore.updateSettings({ size: 'compact', position: 'bottom-right', showHear: true });
      showScreen('live');
      paintLive({
        gauge: 0, peakGauge: 0, teacherActive: true, chatterActive: false,
        chatterStarts: 0, delays: [], usualDelaySec: null,
        voiceLabel: 'just you', voiceTag: 'just-you', status: 'teacher',
        speechHeard: 'Clapping', speechLabel: 'clapping', speechGate: false
      });
      return;
    }
    if (name === 'summary') {
      var summary = {
        durationSec: 1800,
        chatterStarts: 9,
        delays: [5, 8, 7, 10, 8, 9, 6, 8, 11],
        usualDelaySec: 8,
        peakGauge: 0.82,
        peakLevel: 'lots',
        teacherTalkSec: 612,
        endedAt: '2026-11-18T13:30:00.000Z',
        text: TMDetector.formatSummary({ chatterStarts: 9, usualDelaySec: 8 })
      };
      state.lastSummary = summary;
      renderSummary(summary, hist.concat([summary]));
      showScreen('summary');
    }
  }

  function loadSpeechNet() {
    return fetch('assets/speechnet.json?v=0.2.0', { cache: 'no-store' }).then(function (r) {
      if (!r.ok) throw new Error('SpeechNet missing');
      return r.json();
    }).then(function (M) {
      state.speechNet = SpeechNet.load(M);
      if (state.engine) state.engine.setSpeechNet(state.speechNet);
    }).catch(function (err) {
      console.warn('SpeechNet failed to load', err);
    });
  }

  function init() {
    fillCharacterSelect();
    updateEnrollPill();
    updateSensLabels();
    applyChrome();
    bind();
    loadSpeechNet();
    listMics().catch(function () {});
    var params = new URLSearchParams(location.search);
    var shot = params.get('shot');
    if (shot) applyShot(shot);
    var cal = TMStore.load().settings.calibratedAt;
    if (cal && !shot) setStatus('cal-status', 'Room noise was calibrated earlier on this browser.');
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();

  if ('serviceWorker' in navigator && location.protocol !== 'file:') {
    navigator.serviceWorker.register('sw.js').catch(function () {});
  }
})();
