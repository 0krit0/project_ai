// CarDamage AI - Service Worker (Basic PWA)

self.addEventListener('install', event => {
  console.log('[Service Worker] Installed');
  self.skipWaiting();
});

self.addEventListener('activate', event => {
  console.log('[Service Worker] Activated');
});

self.addEventListener('fetch', event => {
  // ใช้ network-first (เหมาะกับเว็บที่ต้องอัปโหลดรูป)
  event.respondWith(
    fetch(event.request).catch(() => {
      return new Response('Offline', {
        status: 503,
        statusText: 'Offline'
      });
    })
  );
});
