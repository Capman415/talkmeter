/* Character display hook.
   v0.1 ships the riveted meter gauge only.
   Next phase can register another adapter (e.g. "cat") without rewriting the app. */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.TMCharacters = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  var adapters = {};
  var current = 'gauge';
  var stage = null;

  var catalog = [
    { id: 'gauge', label: 'Meter gauge', available: true },
    // Next phase: cartoon cat that gets sicker with chatter, hospital bed,
    // and a kid-appropriate gravestone "class mark" with a revive button.
    { id: 'cat', label: 'Cat (coming next)', available: false }
  ];

  function register(name, adapter) {
    adapters[name] = adapter;
  }

  register('gauge', {
    mount: function (el) {
      var gauge = el.querySelector('[data-character-panel="gauge"]');
      if (gauge) gauge.hidden = false;
      var others = el.querySelectorAll('[data-character-panel]');
      for (var i = 0; i < others.length; i++) {
        if (others[i].getAttribute('data-character-panel') !== 'gauge') others[i].hidden = true;
      }
    },
    update: function (el, state) {
      var fill = el.querySelector('#gauge-fill');
      var read = el.querySelector('#gauge-readout');
      var tube = el.querySelector('.gauge-tube');
      var g = Math.max(0, Math.min(1, state.gauge || 0));
      if (fill) fill.style.height = (g * 100).toFixed(1) + '%';
      if (read) read.textContent = Math.round(g * 100) + '%';
      var level = 'quiet';
      if (g >= 0.75) level = 'lots';
      else if (g >= 0.45) level = 'few';
      else if (g >= 0.15) level = 'some';
      if (tube) tube.setAttribute('data-level', level);
      if (el) el.setAttribute('data-gauge-level', level);
    }
  });

  function mount(el, name) {
    stage = el || stage;
    current = name || current || 'gauge';
    if (!adapters[current] || (current !== 'gauge' && !isAvailable(current))) current = 'gauge';
    if (stage) {
      stage.setAttribute('data-character', current);
      if (adapters[current] && adapters[current].mount) adapters[current].mount(stage);
    }
    return current;
  }

  function update(state) {
    if (!stage) return;
    var ad = adapters[current];
    if (ad && ad.update) ad.update(stage, state || {});
  }

  function isAvailable(id) {
    for (var i = 0; i < catalog.length; i++) {
      if (catalog[i].id === id) return !!catalog[i].available;
    }
    return false;
  }

  function set(name) {
    return mount(stage, name);
  }

  return {
    defaultName: 'gauge',
    catalog: catalog,
    register: register,
    mount: mount,
    set: set,
    update: update,
    current: function () { return current; },
    isAvailable: isAvailable
  };
});
