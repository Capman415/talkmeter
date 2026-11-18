/* Talk Meter service worker — caches the app so it keeps working offline after the first load. */
var CACHE = 'talkmeter-v0.3.0';
var ASSETS = [
  './',
  './index.html',
  './version.json',
  './css/style.css',
  './js/audio.js',
  './js/speechnet.js',
  './js/detector.js',
  './js/store.js',
  './js/characters.js',
  './js/app.js',
  './assets/favicon.svg',
  './assets/speechnet.json',
  './assets/fonts/LilitaOne-Regular.ttf',
  './assets/fonts/RammettoOne-Regular.ttf'
];

self.addEventListener('install', function (event) {
  event.waitUntil(
    caches.open(CACHE).then(function (cache) {
      return cache.addAll(ASSETS);
    }).then(function () { return self.skipWaiting(); })
  );
});

self.addEventListener('activate', function (event) {
  event.waitUntil(
    caches.keys().then(function (keys) {
      return Promise.all(keys.filter(function (k) { return k !== CACHE; }).map(function (k) { return caches.delete(k); }));
    }).then(function () { return self.clients.claim(); })
  );
});

self.addEventListener('fetch', function (event) {
  var req = event.request;
  if (req.method !== 'GET') return;
  event.respondWith(
    caches.match(req, { ignoreSearch: true }).then(function (hit) {
      if (hit) return hit;
      return fetch(req).then(function (res) {
        return res;
      }).catch(function () {
        if (req.mode === 'navigate') return caches.match('./index.html');
        throw new Error('offline');
      });
    })
  );
});
