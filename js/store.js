/* localStorage only: numeric voice profile, settings, and class history. Never audio. */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.TMStore = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  var KEY = 'talkmeter_v1';
  var HISTORY_MAX = 40;

  function memoryStorage() {
    var map = {};
    return {
      getItem: function (k) { return Object.prototype.hasOwnProperty.call(map, k) ? map[k] : null; },
      setItem: function (k, v) { map[k] = String(v); },
      removeItem: function (k) { delete map[k]; }
    };
  }

  var storage;
  try {
    if (typeof localStorage !== 'undefined') {
      localStorage.setItem('__tm_test', '1');
      localStorage.removeItem('__tm_test');
      storage = localStorage;
    }
  } catch (e) {
    storage = memoryStorage();
  }
  if (!storage) storage = memoryStorage();

  function defaults() {
    return {
      version: 1,
      profile: null,
      settings: {
        sensitivity: 0.5,
        noiseFloor: null,
        calibratedAt: null,
        micId: '',
        position: 'bottom-right',
        size: 'compact',
        character: 'gauge',
        showHear: true,
        roomSpeechTh: null,
        roomP95Speech: null,
        roomLearnedAt: null,
        roomWindows: null
      },
      history: []
    };
  }

  function load() {
    try {
      var raw = storage.getItem(KEY);
      if (!raw) return defaults();
      var data = JSON.parse(raw);
      var base = defaults();
      if (data && typeof data === 'object') {
        base.profile = data.profile || null;
        if (data.settings) {
          var s = data.settings;
          for (var k in base.settings) {
            if (Object.prototype.hasOwnProperty.call(s, k)) base.settings[k] = s[k];
          }
        }
        base.history = Array.isArray(data.history) ? data.history : [];
      }
      return base;
    } catch (e) {
      return defaults();
    }
  }

  function save(data) {
    storage.setItem(KEY, JSON.stringify(data));
    return data;
  }

  function get() { return load(); }

  function setProfile(profile) {
    var data = load();
    if (profile) {
      var copy = {};
      var keys = ['ok', 'mean', 'std', 'snapshots', 'dim', 'frames', 'voiced', 'total', 'durationSec', 'threshold', 'meanF0', 'distP90', 'enrolledAt'];
      for (var i = 0; i < keys.length; i++) {
        if (profile[keys[i]] != null) copy[keys[i]] = profile[keys[i]];
      }
      if (!copy.enrolledAt) copy.enrolledAt = new Date().toISOString();
      // Belt and suspenders: never persist sample buffers if a caller slipped them in.
      delete copy.samples;
      delete copy.audio;
      delete copy.buffer;
      data.profile = copy;
    } else {
      data.profile = null;
    }
    return save(data);
  }

  function updateSettings(partial) {
    var data = load();
    for (var k in partial) {
      if (Object.prototype.hasOwnProperty.call(partial, k)) data.settings[k] = partial[k];
    }
    return save(data);
  }

  function addHistory(entry) {
    var data = load();
    var row = {
      id: entry.id || ('c' + Date.now()),
      endedAt: entry.endedAt || new Date().toISOString(),
      durationSec: entry.durationSec || 0,
      chatterStarts: entry.chatterStarts || 0,
      usualDelaySec: entry.usualDelaySec == null ? null : entry.usualDelaySec,
      peakGauge: entry.peakGauge || 0,
      teacherTalkSec: entry.teacherTalkSec || 0,
      peakLevel: entry.peakLevel || ''
    };
    data.history.push(row);
    if (data.history.length > HISTORY_MAX) {
      data.history = data.history.slice(data.history.length - HISTORY_MAX);
    }
    return save(data);
  }

  function clearHistory() {
    var data = load();
    data.history = [];
    return save(data);
  }

  function clearAll() {
    storage.removeItem(KEY);
    return defaults();
  }

  function csvEscape(v) {
    if (v == null) return '';
    var s = String(v);
    if (/[",\n]/.test(s)) return '"' + s.replace(/"/g, '""') + '"';
    return s;
  }

  function historyCsv(history) {
    var rows = history || load().history;
    var header = ['ended_at_local', 'duration_sec', 'chatter_starts', 'usual_delay_sec', 'peak_gauge', 'peak_level', 'teacher_talk_sec'];
    var lines = [header.join(',')];
    for (var i = 0; i < rows.length; i++) {
      var r = rows[i];
      var local = '';
      try {
        local = new Date(r.endedAt).toLocaleString();
      } catch (e) {
        local = r.endedAt || '';
      }
      lines.push([
        csvEscape(local),
        csvEscape(Math.round(r.durationSec * 10) / 10),
        csvEscape(r.chatterStarts),
        csvEscape(r.usualDelaySec == null ? '' : Math.round(r.usualDelaySec * 10) / 10),
        csvEscape(Math.round((r.peakGauge || 0) * 100)),
        csvEscape(r.peakLevel || ''),
        csvEscape(Math.round((r.teacherTalkSec || 0) * 10) / 10)
      ].join(','));
    }
    return lines.join('\n');
  }

  function profileHasAudio(profile) {
    if (!profile) return false;
    return !!(profile.samples || profile.audio || profile.buffer);
  }

  return {
    KEY: KEY,
    load: load,
    get: get,
    save: save,
    setProfile: setProfile,
    updateSettings: updateSettings,
    addHistory: addHistory,
    clearHistory: clearHistory,
    clearAll: clearAll,
    historyCsv: historyCsv,
    profileHasAudio: profileHasAudio,
    _setStorage: function (s) { storage = s; },
    _memoryStorage: memoryStorage
  };
});
