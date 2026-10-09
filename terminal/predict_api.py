"""predict.fun REST + WebSocket client.

REST: every call carries ``x-api-key``; wallet calls also ``Authorization: Bearer <JWT>``.
JWT: GET /v1/auth/message -> sign -> POST /v1/auth {signer, message, signature}.
WS:  wss://ws.predict.fun/ws?apiKey=..., {"method":"subscribe","requestId":n,"params":[topic]},
     server heartbeats must be echoed: {"method":"heartbeat","data":<same value>}.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from typing import Any, Awaitable, Callable

import aiohttp

log = logging.getLogger("predict")


class PredictError(Exception):
    def __init__(self, status: int, body: Any) -> None:
        self.status = status
        self.body = body
        msg = body.get("message") or body.get("error") if isinstance(body, dict) else str(body)[:300]
        super().__init__(f"HTTP {status}: {msg}")


class PredictRest:
    def __init__(self, base_url: str, api_key: str, signer_address: str,
                 sign_message: Callable[[str], str]) -> None:
        self.base = base_url.rstrip("/")
        self.api_key = api_key
        self.signer_address = signer_address
        self._sign_message = sign_message
        self.session: aiohttp.ClientSession | None = None
        self.jwt: str | None = None
        self.jwt_expires = 0.0
        self._auth_lock = asyncio.Lock()
        self._calls: deque[float] = deque(maxlen=2000)

    @property
    def rpm(self) -> int:
        """REST requests in the last 60 s (predict.fun limit: 240/min per API key)."""
        cutoff = time.time() - 60
        return sum(1 for t in self._calls if t >= cutoff)

    async def start(self) -> None:
        conn = aiohttp.TCPConnector(limit=16, ttl_dns_cache=3600, keepalive_timeout=120)
        self.session = aiohttp.ClientSession(connector=conn, timeout=aiohttp.ClientTimeout(total=10),
                                             json_serialize=json.dumps)

    async def close(self) -> None:
        if self.session is not None:
            await self.session.close()

    def _headers(self, auth: bool) -> dict[str, str]:
        h = {"x-api-key": self.api_key} if self.api_key else {}
        if auth and self.jwt:
            h["Authorization"] = f"Bearer {self.jwt}"
        return h

    async def _req(self, method: str, path: str, *, params: dict | None = None, body: dict | None = None,
                   auth: bool = False, retry_auth: bool = True) -> Any:
        assert self.session is not None
        if auth:
            await self.ensure_jwt()
        url = f"{self.base}{path}"
        self._calls.append(time.time())
        async with self.session.request(method, url, params=params, json=body, headers=self._headers(auth)) as r:
            text = await r.text()
            try:
                data = json.loads(text) if text else None
            except json.JSONDecodeError:
                data = text
            if r.status == 401 and auth and retry_auth:
                self.jwt = None
                return await self._req(method, path, params=params, body=body, auth=auth, retry_auth=False)
            if r.status >= 400 or (isinstance(data, dict) and data.get("success") is False):
                raise PredictError(r.status, data)
            return data

    # ---- auth ------------------------------------------------------------
    async def ensure_jwt(self) -> str:
        if self.jwt and time.time() < self.jwt_expires:
            return self.jwt
        async with self._auth_lock:
            if self.jwt and time.time() < self.jwt_expires:
                return self.jwt
            msg = await self._req("GET", "/v1/auth/message")
            message = (msg or {}).get("data", {}).get("message")
            if not message:
                raise PredictError(0, {"message": "auth message missing"})
            signature = self._sign_message(message)
            res = await self._req("POST", "/v1/auth", body={"signer": self.signer_address, "message": message,
                                                             "signature": signature})
            token = (res or {}).get("data", {}).get("token")
            if not token:
                raise PredictError(0, {"message": "auth token missing"})
            self.jwt = token
            self.jwt_expires = time.time() + 23 * 3600
            log.info("JWT ok for %s", self.signer_address)
            return token

    # ---- market data -----------------------------------------------------
    async def category(self, slug: str) -> dict[str, Any] | None:
        try:
            res = await self._req("GET", f"/v1/categories/{slug}")
        except PredictError as e:
            if e.status == 404:
                return None
            raise
        return (res or {}).get("data")

    async def orderbook(self, market_id: int) -> dict[str, Any]:
        res = await self._req("GET", f"/v1/markets/{market_id}/orderbook")
        return (res or {}).get("data") or {}

    # ---- account ---------------------------------------------------------
    async def account(self) -> dict[str, Any]:
        res = await self._req("GET", "/v1/account", auth=True)
        return (res or {}).get("data") or {}

    async def positions(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        after: str | None = None
        for _ in range(5):
            params = {"first": "100"}
            if after:
                params["after"] = after
            res = await self._req("GET", "/v1/positions", params=params, auth=True)
            out.extend((res or {}).get("data") or [])
            after = (res or {}).get("cursor")
            if not after:
                break
        return out

    async def open_orders(self) -> list[dict[str, Any]]:
        res = await self._req("GET", "/v1/orders", params={"status": "OPEN"}, auth=True)
        return (res or {}).get("data") or []

    async def create_order(self, data: dict[str, Any]) -> dict[str, Any]:
        res = await self._req("POST", "/v1/orders", body={"data": data}, auth=True)
        return (res or {}).get("data") or {}

    async def remove_orders(self, hashes: list[str], ids: list[str] | None = None) -> dict[str, Any]:
        """Remove by hash; if that path is refused, fall back to POST /v1/orders/remove {ids}."""
        try:
            return await self._req("POST", "/v1/orders/remove-by-hash", body={"data": {"hashes": hashes}}, auth=True)
        except PredictError as exc:
            if exc.status not in (400, 404, 405) or not ids:
                raise
        res = await self._req("POST", "/v1/orders/remove", body={"data": {"ids": ids}}, auth=True)
        return {"removed": ids, "rejected": [], "noop": [], "raw": res}


Handler = Callable[[str, dict[str, Any]], None]


class PredictWS:
    """Single socket, auto-reconnect, resubscribes every topic after reconnect."""

    def __init__(self, url: str, api_key: str, on_message: Handler,
                 wallet_topic: Callable[[], Awaitable[str | None]] | None = None,
                 on_status: Callable[[bool], None] | None = None) -> None:
        self.url = f"{url}?apiKey={api_key}" if api_key else url
        self.headers = {"x-api-key": api_key} if api_key else {}
        self.on_message = on_message
        self.wallet_topic = wallet_topic
        self.on_status = on_status
        self.topics: set[str] = set()
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._req_id = 0
        self._stop = False
        self.connected = False
        self.last_msg = 0.0

    async def _send(self, payload: dict[str, Any]) -> None:
        if self._ws is not None and not self._ws.closed:
            await self._ws.send_str(json.dumps(payload))

    async def _subscribe_raw(self, topic: str, method: str = "subscribe") -> None:
        self._req_id += 1
        await self._send({"method": method, "requestId": self._req_id, "params": [topic]})

    async def subscribe(self, topic: str) -> None:
        if topic in self.topics:
            return
        self.topics.add(topic)
        await self._subscribe_raw(topic)

    async def unsubscribe(self, topic: str) -> None:
        if topic not in self.topics:
            return
        self.topics.discard(topic)
        await self._subscribe_raw(topic, "unsubscribe")

    def stop(self) -> None:
        self._stop = True

    def _set_status(self, up: bool) -> None:
        self.connected = up
        if self.on_status:
            self.on_status(up)

    async def run(self) -> None:
        backoff = 1.0
        while not self._stop:
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.ws_connect(self.url, headers=self.headers, heartbeat=None, max_msg_size=8 * 2**20) as ws:
                        self._ws = ws
                        self._set_status(True)
                        backoff = 1.0
                        for t in list(self.topics):
                            await self._subscribe_raw(t)
                        if self.wallet_topic is not None:
                            try:
                                wt = await self.wallet_topic()
                                if wt:
                                    await self._subscribe_raw(wt)
                            except Exception as exc:  # noqa: BLE001
                                log.warning("wallet events subscribe failed: %s", exc)
                        async for msg in ws:
                            if msg.type != aiohttp.WSMsgType.TEXT:
                                if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                    break
                                continue
                            self.last_msg = time.time()
                            try:
                                data = json.loads(msg.data)
                            except json.JSONDecodeError:
                                continue
                            await self._dispatch(data)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.warning("predict ws: %s", exc)
            self._ws = None
            self._set_status(False)
            if self._stop:
                break
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 15)

    async def _dispatch(self, data: dict[str, Any]) -> None:
        if not isinstance(data, dict):
            return
        if data.get("type") == "R":
            if data.get("success") is False:
                log.warning("predict ws request %s rejected: %s", data.get("requestId"), data.get("error"))
            return
        topic = str(data.get("topic") or "")
        if topic == "heartbeat":
            await self._send({"method": "heartbeat", "data": data.get("data")})
            return
        if topic:
            payload = data.get("data")
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except json.JSONDecodeError:
                    return
            if isinstance(payload, dict):
                self.on_message(topic, payload)
