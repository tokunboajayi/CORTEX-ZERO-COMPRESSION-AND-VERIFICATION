const CACHE_NAME = 'halt-nn-v1';
const STATIC_ASSETS = [
    '/',
    '/static/index.html',
    '/static/manifest.json'
];

// Install - cache static assets
self.addEventListener('install', event => {
    event.waitUntil(
        caches.open(CACHE_NAME).then(cache => {
            console.log('[SW] Caching static assets');
            return cache.addAll(STATIC_ASSETS);
        })
    );
    self.skipWaiting();
});

// Activate - clean old caches
self.addEventListener('activate', event => {
    event.waitUntil(
        caches.keys().then(keys => {
            return Promise.all(
                keys.filter(key => key !== CACHE_NAME)
                    .map(key => caches.delete(key))
            );
        })
    );
    self.clients.claim();
});

// Fetch - network first, cache fallback
self.addEventListener('fetch', event => {
    // Skip non-GET requests
    if (event.request.method !== 'GET') return;

    // Skip API requests (always fetch fresh)
    if (event.request.url.includes('/query') ||
        event.request.url.includes('/evidence') ||
        event.request.url.includes('/ws/')) {
        return;
    }

    event.respondWith(
        fetch(event.request)
            .then(response => {
                // Clone and cache successful responses
                if (response.ok) {
                    const clone = response.clone();
                    caches.open(CACHE_NAME).then(cache => {
                        cache.put(event.request, clone);
                    });
                }
                return response;
            })
            .catch(() => {
                // Return cached version if offline
                return caches.match(event.request);
            })
    );
});

// Background sync for offline queries
self.addEventListener('sync', event => {
    if (event.tag === 'sync-queries') {
        event.waitUntil(syncOfflineQueries());
    }
});

async function syncOfflineQueries() {
    // Get queued queries from IndexedDB
    const db = await openDB();
    const queries = await db.getAll('offline-queries');

    for (const query of queries) {
        try {
            await fetch('/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(query)
            });
            await db.delete('offline-queries', query.id);
        } catch (e) {
            console.log('[SW] Sync failed, will retry');
        }
    }
}

// Push notifications
self.addEventListener('push', event => {
    const data = event.data?.json() || {};
    const options = {
        body: data.body || 'New verification result available',
        icon: '/static/icon-192.png',
        badge: '/static/icon-192.png',
        vibrate: [100, 50, 100],
        data: { url: data.url || '/' }
    };

    event.waitUntil(
        self.registration.showNotification(data.title || 'HALT-NN', options)
    );
});

self.addEventListener('notificationclick', event => {
    event.notification.close();
    event.waitUntil(
        clients.openWindow(event.notification.data.url)
    );
});

console.log('[SW] Service Worker loaded');
