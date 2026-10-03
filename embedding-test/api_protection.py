"""API access controls and bounded, per-process request limiting."""
import hmac
import math
import os
import time
from collections import OrderedDict, deque
from starlette.responses import JSONResponse


class APIProtectionMiddleware:
    def __init__(self, app):
        self.app = app
        self.api_key = os.getenv('API_ACCESS_KEY', '')
        self.admin_key = os.getenv('INGEST_API_KEY', '')
        self.limit = int(os.getenv('API_RATE_LIMIT', '20'))
        self.window = int(os.getenv('API_RATE_WINDOW_SECONDS', '60'))
        self.max_clients = 10000
        self.clients = OrderedDict()
        if self.limit < 1 or self.window < 1:
            raise ValueError('API rate limit and window must be positive')

    async def __call__(self, scope, receive, send):
        path = scope.get('path', '')
        if scope['type'] != 'http' or not path.startswith('/api/v1/'):
            return await self.app(scope, receive, send)
        # Health remains available to infrastructure probes; preflight does no work.
        if path == '/api/v1/health' or scope['method'] == 'OPTIONS':
            return await self.app(scope, receive, send)
        headers = dict(scope.get('headers', []))
        supplied = headers.get(b'x-api-key', b'').decode('latin-1')
        expected = self.admin_key if path.startswith('/api/v1/ingest/') else self.api_key
        if path.startswith('/api/v1/ingest/') and not expected:
            return await JSONResponse({'detail': 'Manual ingestion is disabled; configure INGEST_API_KEY.'}, status_code=503)(scope, receive, send)
        # Use the transport peer, never arbitrary client-supplied forwarding headers.
        client = (scope.get('client') or ('unknown', 0))[0]
        now = time.monotonic()
        while self.clients:
            oldest = next(iter(self.clients))
            if self.clients[oldest][-1] > now - self.window:
                break
            self.clients.popitem(last=False)
        if client not in self.clients:
            if len(self.clients) >= self.max_clients:
                return await JSONResponse({'detail': 'Request limiter capacity reached.'}, status_code=429, headers={'Retry-After': str(self.window)})(scope, receive, send)
            self.clients[client] = deque()
        events = self.clients[client]
        while events and events[0] <= now - self.window:
            events.popleft()
        self.clients.move_to_end(client)
        if len(events) >= self.limit:
            retry = max(1, math.ceil(events[0] + self.window - now))
            return await JSONResponse({'detail': 'Too many requests.'}, status_code=429, headers={'Retry-After': str(retry)})(scope, receive, send)
        events.append(now)
        if expected and not hmac.compare_digest(supplied.encode(), expected.encode()):
            return await JSONResponse({'detail': 'Invalid or missing X-API-Key.'}, status_code=401)(scope, receive, send)
        # Reject oversized payloads before model inference. Count streamed bodies too.
        body = bytearray()
        while True:
            message = await receive()
            if message['type'] == 'http.disconnect':
                return
            body.extend(message.get('body', b''))
            if len(body) > 65536:
                return await JSONResponse({'detail': 'Request body too large.'}, status_code=413)(scope, receive, send)
            if not message.get('more_body', False):
                break
        delivered = False
        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {'type': 'http.request', 'body': bytes(body), 'more_body': False}
            return await receive()
        return await self.app(scope, replay, send)
