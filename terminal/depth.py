"""Binance Spot BTCUSDT local order book (REST snapshot + diff stream) for the
order-profile / walls overlay, plus an offline demo book."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import time
from collections import deque
from typing import Any, Callable

import aiohttp

log = logging.getLogger("depth")


class SpotDepthBook:
    """Standard Binance sync: buffer diffs until a snapshot; drop u <= lastUpdateId; the first
    applied event must satisfy U <= lastUpdateId+1 <= u; then U == previous u + 1, else resync."""

    def __init__(self, max_buffer: int = 5000) -> None:
        self._max_buffer = max_buffer
        self.snapshots = 0
        self.gaps = 0
        self.reset()

    def reset(self) -> None:
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self.last_u = -1
        self.synced = False
        self._buffer: deque[dict[str, Any]] = deque(maxlen=self._max_buffer)
        self.last_event = 0.0

    @staticmethod
    def _apply_side(book: dict[float, float], levels: Any) -> None:
        for lv in levels or ():
            p, q = float(lv[0]), float(lv[1])
            if q <= 0.0:
                book.pop(p, None)
            else:
                book[p] = q

    def _apply_live(self, msg: dict[str, Any]) -> bool:
        first, last = int(msg["U"]), int(msg["u"])
        if last <= self.last_u:
            return False
        if first > self.last_u + 1:
            self.gaps += 1
            self.synced = False
            self._buffer.clear()
            self._buffer.append(msg)
            return False
        self._apply_side(self.bids, msg.get("b"))
        self._apply_side(self.asks, msg.get("a"))
        self.last_u = last
        self.last_event = time.time()
        return True

    def on_event(self, msg: dict[str, Any]) -> bool:
        if not self.synced:
            self._buffer.append(msg)
            return False
        return self._apply_live(msg)

    def apply_snapshot(self, snap: dict[str, Any]) -> bool:
        lid = int(snap["lastUpdateId"])
        pending = [m for m in self._buffer if int(m.get("u", -1)) > lid]
        if pending and int(pending[0].get("U", 0)) > lid + 1:
            return False                       # snapshot older than the buffer: fetch again
        self.bids = {float(p): float(q) for p, q, *_ in snap.get("bids") or () if float(q) > 0}
        self.asks = {float(p): float(q) for p, q, *_ in snap.get("asks") or () if float(q) > 0}
        self.last_u = lid
        self.synced = True
        self.last_event = time.time()
        self._buffer.clear()
        for m in pending:
            self._apply_live(m)
            if not self.synced:
                return False
        self.snapshots += 1
        return True

    def prune(self, mid: float, frac: float = 0.03) -> None:
        lo, hi = mid * (1 - frac), mid * (1 + frac)
        self.bids = {p: q for p, q in self.bids.items() if p >= lo}
        self.asks = {p: q for p, q in self.asks.items() if p <= hi}


def aggregate(bids: dict[float, float], asks: dict[float, float], range_usd: float,
              step: float = 1.0) -> dict[str, Any] | None:
    """$-step buckets within range_usd of the mid. Bids floor, asks ceil (a bucket never
    crosses the spread)."""
    if not bids or not asks:
        return None
    bb, ba = max(bids), min(asks)
    mid = (bb + ba) / 2
    lo, hi = mid - range_usd, mid + range_usd
    b: dict[int, float] = {}
    a: dict[int, float] = {}
    for p, q in bids.items():
        if p >= lo:
            k = math.floor(p / step + 1e-9)
            b[k] = b.get(k, 0.0) + q
    for p, q in asks.items():
        if p <= hi:
            k = math.ceil(p / step - 1e-9)
            a[k] = a.get(k, 0.0) + q
    return {"bb": bb, "ba": ba, "step": step,
            "b": [[round(k * step, 2), round(q, 4)] for k, q in sorted(b.items(), reverse=True)],
            "a": [[round(k * step, 2), round(q, 4)] for k, q in sorted(a.items())]}


class BinanceDepthFeed:
    def __init__(self, ws_url: str, rest_url: str, book: SpotDepthBook, limit: int = 5000) -> None:
        self.ws_url = ws_url
        self.rest_url = rest_url
        self.book = book
        self.limit = limit
        self._stop = False
        self._loading = False
        self._next_try = 0.0
        self.connected = False

    def stop(self) -> None:
        self._stop = True

    async def _snapshot(self, s: aiohttp.ClientSession) -> None:
        try:
            async with s.get(f"{self.rest_url}/api/v3/depth",
                             params={"symbol": "BTCUSDT", "limit": str(self.limit)}) as r:
                snap = await r.json(content_type=None)
            ok = self.book.apply_snapshot(snap)
            self._next_try = 0.0 if ok else time.time() + 0.5
            if ok:
                log.info("spot depth synced: %d bids / %d asks", len(self.book.bids), len(self.book.asks))
        except Exception as exc:  # noqa: BLE001
            log.warning("spot depth snapshot failed: %s", exc)
            self._next_try = time.time() + 3
        finally:
            self._loading = False

    async def run(self) -> None:
        backoff = 1.0
        while not self._stop:
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as s:
                    async with s.ws_connect(self.ws_url, heartbeat=20) as ws:
                        self.connected = True
                        backoff = 1.0
                        async for msg in ws:
                            if msg.type != aiohttp.WSMsgType.TEXT:
                                if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                    break
                                continue
                            d = json.loads(msg.data)
                            d = d.get("data", d)
                            if "U" not in d:
                                continue
                            self.book.on_event(d)
                            if not self.book.synced and not self._loading and time.time() >= self._next_try:
                                self._loading = True
                                asyncio.ensure_future(self._snapshot(s))
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.warning("spot depth ws: %s", exc)
            self.connected = False
            self.book.reset()
            if self._stop:
                break
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 15)


class DemoDepthFeed:
    """Synthetic book around the demo mid with a few walls that come and go."""

    def __init__(self, book: SpotDepthBook, mid: Callable[[], float]) -> None:
        self.book = book
        self.mid = mid
        self._stop = False
        self.connected = True
        self.walls: dict[float, float] = {}

    def stop(self) -> None:
        self._stop = True

    async def run(self) -> None:
        while not self._stop:
            m = self.mid()
            if m and not math.isnan(m):
                if random.random() < 0.05 or not self.walls:
                    self.walls = {round(m + random.choice((-1, 1)) * random.uniform(15, 400), 0): random.uniform(8, 40)
                                  for _ in range(4)}
                bids, asks = {}, {}
                for i in range(1, 900):
                    bids[round(m - 0.005 - i * 0.5, 2)] = abs(random.gauss(0.15, 0.2)) * (1.8 if i < 20 else 1)
                    asks[round(m + 0.005 + i * 0.5, 2)] = abs(random.gauss(0.15, 0.2)) * (1.8 if i < 20 else 1)
                for p, q in self.walls.items():
                    (bids if p < m else asks)[p] = q
                self.book.bids, self.book.asks = bids, asks
                self.book.synced = True
                self.book.last_event = time.time()
            await asyncio.sleep(0.5)
