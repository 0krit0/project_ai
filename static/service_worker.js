// CarDamage AI - Service Worker (Basic PWA)

self.addEventListener('install', event => {
  console.log('[Service Worker] Installed');
  self.skipWaiting();
});

self.addEventListener('activate', event => {
  console.log('[Service Worker] Activated');
});

self.addEventListener('fetch', event => {
  // network-first strategy
  event.respondWith(
    fetch(event.request).catch(() => {
      return new Response('Offline', {
        status: 503,
        statusText: 'Offline'
      });
    })
  );
});
