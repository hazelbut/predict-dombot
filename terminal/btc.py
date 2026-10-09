"""BTCUSDT from Binance Spot: bookTicker mid (what Chainlink's stream is built from)
plus aggTrade volume; 1 s klines backfill the chart on start."""

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

log = logging.getLogger("btc")

Tick = Callable[[int, float, float, float], None]   # ts_ms, mid, bid, ask
Kline = tuple[int, float, float, float, float, float]   # open sec, o, h, l, c, volume


class BtcState:
    def __init__(self, minutes: int) -> None:
        self.mid = math.nan
        self.bid = math.nan
        self.ask = math.nan
        self.ts = 0
        self.connected = False
        # per-second history: (sec, mid)
        self.hist: deque[tuple[int, float]] = deque(maxlen=max(60, minutes * 60 + 120))
        self.on_kline: Callable[[Kline], None] | None = None
        # market-order flow per second: sec -> [qty_buy, notional_buy, qty_sell, notional_sell]
        self.flow: dict[int, list[float]] = {}
        self.flow_order: deque[int] = deque()
        self.flow_dirty: set[int] = set()
        self.flow_keep_s = 4 * 3600

    def on_trade(self, ts_ms: int, price: float, qty: float, sell: bool) -> None:
        sec = ts_ms // 1000
        row = self.flow.get(sec)
        if row is None:
            row = self.flow[sec] = [0.0, 0.0, 0.0, 0.0]
            self.flow_order.append(sec)
            while self.flow_order and self.flow_order[0] < sec - self.flow_keep_s:
                self.flow.pop(self.flow_order.popleft(), None)
        i = 2 if sell else 0
        row[i] += qty
        row[i + 1] += qty * price
        self.flow_dirty.add(sec)

    def flow_rows(self, secs: Any = None) -> list[list[float]]:
        keys = sorted(secs) if secs is not None else list(self.flow_order)
        return [[s, *(round(x, 5) for x in self.flow[s])] for s in keys if s in self.flow]

    def on_tick(self, ts_ms: int, mid: float, bid: float, ask: float) -> None:
        self.mid, self.bid, self.ask, self.ts = mid, bid, ask, ts_ms
        sec = ts_ms // 1000
        if self.hist and self.hist[-1][0] == sec:
            self.hist[-1] = (sec, mid)
        else:
            self.hist.append((sec, mid))

    def price_at(self, sec: int) -> float | None:
        """Last mid at or before `sec` (approximate start price before the official one is out)."""
        best = None
        for s, p in self.hist:
            if s <= sec:
                best = p
            else:
                break
        return best


class BinanceFeed:
    def __init__(self, ws_url: str, rest_url: str, state: BtcState, minutes: int) -> None:
        self.ws_url = ws_url
        self.rest_url = rest_url
        self.state = state
        self.minutes = minutes
        self._stop = False

    async def backfill(self) -> None:
        end = int(time.time() * 1000)
        start = end - self.minutes * 60_000
        points: list[tuple[int, float]] = []
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as s:
                cursor = start
                while cursor < end:
                    params = {"symbol": "BTCUSDT", "interval": "1s", "startTime": str(cursor), "limit": "1000"}
                    async with s.get(f"{self.rest_url}/api/v3/klines", params=params) as r:
                        rows = await r.json(content_type=None)
                    if not isinstance(rows, list) or not rows:
                        break
                    for k in rows:
                        points.append((int(k[0]) // 1000, float(k[4])))
                    cursor = int(rows[-1][0]) + 1000
                    if len(rows) < 1000:
                        break
        except Exception as exc:  # noqa: BLE001
            log.warning("binance klines backfill failed: %s", exc)
            return
        for sec, px in points:
            if not self.state.hist or sec > self.state.hist[-1][0]:
                self.state.hist.append((sec, px))

    def stop(self) -> None:
        self._stop = True

    async def run(self) -> None:
        await self.backfill()
        backoff = 1.0
        while not self._stop:
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.ws_connect(self.ws_url, heartbeat=20) as ws:
                        self.state.connected = True
                        backoff = 1.0
                        async for msg in ws:
                            if msg.type != aiohttp.WSMsgType.TEXT:
                                if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                    break
                                continue
                            data = json.loads(msg.data)
                            d = data.get("data", data)
                            if "b" in d and "a" in d and "u" in d:
                                bid, ask = float(d["b"]), float(d["a"])
                                self.state.on_tick(int(time.time() * 1000), (bid + ask) / 2, bid, ask)
                            elif d.get("e") == "aggTrade":
                                # m = buyer is maker -> the aggressor SOLD
                                self.state.on_trade(int(d["T"]), float(d["p"]), float(d["q"]), bool(d.get("m")))
                            elif d.get("e") == "kline" and self.state.on_kline is not None:
                                k = d["k"]
                                self.state.on_kline((int(k["t"]) // 1000, float(k["o"]), float(k["h"]), float(k["l"]),
                                                     float(k["c"]), float(k["v"])))
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.warning("binance ws: %s", exc)
            self.state.connected = False
            if self._stop:
                break
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 15)


class DemoBtcFeed:
    """Random walk with bursts (offline demo)."""

    def __init__(self, state: BtcState, start: float = 62_000.0) -> None:
        self.state = state
        self.px = start
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    async def run(self) -> None:
        now = int(time.time())
        px = self.px
        for sec in range(now - self.state.hist.maxlen + 5, now):  # type: ignore[operator]
            px += random.gauss(0, 4)
            self.state.hist.append((sec, px))
        self.px = px
        self.state.connected = True
        vol = 1.2
        bar: list[float] | None = None      # [sec, o, h, l, c, v]
        while not self._stop:
            if random.random() < 0.01:
                vol = random.choice((0.6, 1.2, 3.0))
            self.px += random.gauss(0, vol)
            spread = 0.01
            now = time.time()
            self.state.on_tick(int(now * 1000), self.px, self.px - spread / 2, self.px + spread / 2)
            sec = int(now)
            if bar is None or bar[0] != sec:
                bar = [sec, self.px, self.px, self.px, self.px, 0.0]
            bar[2], bar[3], bar[4] = max(bar[2], self.px), min(bar[3], self.px), self.px
            bar[5] += abs(random.gauss(0, 0.4)) * vol
            for _ in range(random.randint(0, 4)):
                q = random.paretovariate(1.6) * 0.01 * vol
                self.state.on_trade(int(now * 1000), self.px, q, random.random() < 0.5)
            if self.state.on_kline is not None:
                self.state.on_kline((int(bar[0]), bar[1], bar[2], bar[3], bar[4], round(bar[5], 4)))
            await asyncio.sleep(0.1)


INTERVAL_S = {"1s": 1, "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400}


def demo_klines(state: BtcState, interval: str, limit: int, end_ms: int | None) -> list[list[float]]:
    """Synthetic OHLCV for the demo: per-second history where known, a random walk
    generated backwards (continuous with the known part) before it."""
    step = INTERVAL_S.get(interval, 60)
    end_s = (end_ms // 1000 if end_ms else int(time.time())) // step * step
    hist = dict(state.hist)
    rng = random.Random(step * 7919 + end_s // (step * 500))
    nxt_open = state.hist[-1][1] if state.hist else 62_000.0
    out: list[list[float]] = []
    for i in range(limit):                       # newest -> oldest
        t0 = end_s - i * step
        secs = [hist[s] for s in range(t0, t0 + step) if s in hist]
        if secs:
            o, c, h, lo = secs[0], secs[-1], max(secs), min(secs)
        else:
            sigma = 4.0 * math.sqrt(step)
            c = nxt_open
            o = c - rng.gauss(0, sigma)
            h = max(o, c) + abs(rng.gauss(0, sigma * 0.4))
            lo = min(o, c) - abs(rng.gauss(0, sigma * 0.4))
        nxt_open = o
        out.append([t0 * 1000, o, h, lo, c, abs(rng.gauss(2, 1)) * step ** 0.5])
    out.reverse()
    return out
