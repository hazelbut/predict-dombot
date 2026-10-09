"""Local web server: serves the UI and bridges browser <-> venue over a WebSocket."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from collections import deque
from pathlib import Path
from typing import Any

from aiohttp import WSMsgType, web

import aiohttp

from .btc import INTERVAL_S, BinanceFeed, BtcState, DemoBtcFeed, demo_klines
from .depth import BinanceDepthFeed, DemoDepthFeed, SpotDepthBook, aggregate
from .config import Settings
from .trading import OrderRejected
from .venue import Venue

log = logging.getLogger("server")
HEAT_HALF = 400          # $ either side of mid kept for the heatmap
HEAT_KEEP_S = 1800       # 30 min of 1 s snapshots


def heat_row(agg: dict[str, Any], sec: int) -> list[Any]:
    """[sec, lo, [q0..q800]] with q in 0.01 BTC: bids floored, asks ceiled to $1 (same as the UI)."""
    mid = (agg["bb"] + agg["ba"]) / 2
    lo = math.floor(mid - HEAT_HALF)
    q = [0.0] * (2 * HEAT_HALF + 1)
    for p, v in agg["b"]:
        k = math.floor(p) - lo
        if 0 <= k < len(q):
            q[k] += v
    for p, v in agg["a"]:
        k = math.ceil(p) - lo
        if 0 <= k < len(q):
            q[k] += v
    iq = [round(x * 100) for x in q]
    i0 = next((i for i, x in enumerate(iq) if x), len(iq))          # trim empty edges: smaller hello
    i1 = len(iq) - next((i for i, x in enumerate(reversed(iq)) if x), len(iq))
    return [sec, lo + i0, iq[i0:i1]]
STATIC = Path(__file__).parent / "static"


class Hub:
    def __init__(self) -> None:
        self.clients: set[web.WebSocketResponse] = set()

    def publish(self, msg: dict[str, Any]) -> None:
        if not self.clients:
            return
        text = json.dumps(msg, separators=(",", ":"))
        for ws in list(self.clients):
            if ws.closed:
                self.clients.discard(ws)
                continue
            asyncio.ensure_future(self._send(ws, text))

    async def _send(self, ws: web.WebSocketResponse, text: str) -> None:
        try:
            await ws.send_str(text)
        except Exception:  # noqa: BLE001
            self.clients.discard(ws)


class Terminal:
    def __init__(self, s: Settings) -> None:
        self.s = s
        self.hub = Hub()
        self.btc = BtcState(s.chart_minutes)
        self.venue: Venue
        if s.demo:
            from .venue_demo import DemoVenue
            self.venue = DemoVenue(s, self.hub.publish, self.btc)
            self.btc_feed: Any = DemoBtcFeed(self.btc)
        else:
            from .venue_live import LiveVenue
            self.venue = LiveVenue(s, self.hub.publish)
            self.btc_feed = BinanceFeed(s.binance_ws_url, s.binance_rest_url, self.btc, s.chart_minutes)
        self.depth = SpotDepthBook()
        if s.demo:
            self.depth_feed: Any = DemoDepthFeed(self.depth, lambda: self.btc.mid)
        else:
            self.depth_feed = BinanceDepthFeed(s.binance_depth_ws_url, s.binance_rest_url, self.depth)
        self._depth_msg: dict[str, Any] | None = None
        # heatmap history: 1 snapshot/s, $1 buckets within ±HEAT_HALF of mid, volume in 0.01 BTC
        self.heat: deque[list[Any]] = deque(maxlen=HEAT_KEEP_S)
        self.armed = False
        self.btc.on_kline = self._on_kline
        self._kline_last: tuple | None = None
        self._http: aiohttp.ClientSession | None = None
        self._tasks: list[asyncio.Task[Any]] = []
        self._last_btc_sent = 0

    # ------------------------------------------------------------------
    def hello(self) -> list[dict[str, Any]]:
        s = self.s
        msgs: list[dict[str, Any]] = [
            {"t": "hello", "venue": self.venue.name, "demo": s.demo, "address": self.venue.address,
             "max_order_usdt": s.max_order_usdt, "default_size_usdt": s.default_size_usdt,
             "size_presets": [float(x) for x in s.size_presets.split(",") if x.strip()],
             "slippage_ticks": s.slippage_ticks, "armed": self.armed,
             "series": [{"prefix": x.prefix, "label": x.label, "period_s": x.period_s} for x in s.series]},
            self.venue.status(),
            {"t": "btc_hist", "points": list(self.btc.hist)},
            {"t": "flow_hist", "rows": self.btc.flow_rows()},
        ]
        if self._depth_msg is not None:
            msgs.append(self._depth_msg)
        if self.heat:
            msgs.append({"t": "heat_hist", "half": HEAT_HALF, "rows": list(self.heat)})
        now = time.time() * 1000
        items = [m.to_json() for m in self.venue.markets.values() if m.end_ms > now - 600_000]
        items.sort(key=lambda d: (d["series"], d["start_ms"]))
        msgs.append({"t": "markets", "items": items})
        msgs.append({"t": "account", **self.venue.account.to_json()})
        msgs.extend(self.venue.recent_logs)
        for mid, b in self.venue.books.items():
            msgs.append({"t": "book", "market_id": mid, **b.to_json()})
        return msgs

    async def _flush_loop(self) -> None:
        """Coalesce book/BTC updates: books up to 20/s per market, BTC 10/s."""
        n = 0
        while True:
            await asyncio.sleep(0.05)
            v = self.venue
            if v.dirty_books:
                dirty, v.dirty_books = v.dirty_books, set()
                for mid in dirty:
                    b = v.books.get(mid)
                    if b is not None:
                        self.hub.publish({"t": "book", "market_id": mid, **b.to_json()})
            n += 1
            if n % 2 == 0 and self.btc.ts and self.btc.ts != self._last_btc_sent and not math.isnan(self.btc.mid):
                self._last_btc_sent = self.btc.ts
                self.hub.publish({"t": "btc", "ts": self.btc.ts, "mid": self.btc.mid, "bid": self.btc.bid,
                                  "ask": self.btc.ask, "ok": self.btc.connected})
            if n % 5 == 0 and self.btc.flow_dirty:
                dirty, self.btc.flow_dirty = self.btc.flow_dirty, set()
                self.hub.publish({"t": "flow", "rows": self.btc.flow_rows(dirty)})
            if n % 10 == 0:
                d = self.depth
                age = time.time() - d.last_event if d.last_event else 1e9
                agg = aggregate(d.bids, d.asks, self.s.depth_range_usd) if d.synced and age < 3 else None
                self._depth_msg = {"t": "depth", "ok": agg is not None, **(agg or {})}
                self.hub.publish(self._depth_msg)
                if agg is not None:
                    self._heat_snap(agg)
            if n % 600 == 0 and self.depth.synced and not math.isnan(self.btc.mid) and not self.s.demo:
                self.depth.prune(self.btc.mid)
            if n % 40 == 0:
                st = v.status()
                st["btc_ok"] = self.btc.connected
                self.hub.publish(st)

    def _heat_snap(self, agg: dict[str, Any]) -> None:
        sec = int(time.time())
        if self.heat and self.heat[-1][0] == sec:
            return
        self.heat.append(heat_row(agg, sec))

    def _on_kline(self, k: tuple) -> None:
        if k == self._kline_last:
            return
        self._kline_last = k
        self.hub.publish({"t": "k", "k": list(k)})

    async def klines(self, request: web.Request) -> web.Response:
        """Binance BTCUSDT klines proxy: /api/klines?interval=1m&limit=500[&endTime=ms]."""
        interval = request.query.get("interval", "1m")
        if interval not in INTERVAL_S:
            return web.json_response({"error": "bad interval"}, status=400)
        limit = max(1, min(int(request.query.get("limit", "500")), 1000))
        end = request.query.get("endTime")
        if self.s.demo:
            return web.json_response(demo_klines(self.btc, interval, limit, int(end) if end else None))
        params = {"symbol": "BTCUSDT", "interval": interval, "limit": str(limit)}
        if end:
            params["endTime"] = end
        if self._http is None:
            self._http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
        try:
            async with self._http.get(f"{self.s.binance_rest_url}/api/v3/klines", params=params) as r:
                rows = await r.json(content_type=None)
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"error": str(exc)}, status=502)
        if not isinstance(rows, list):
            return web.json_response({"error": str(rows)[:200]}, status=502)
        return web.json_response([[int(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5])]
                                  for x in rows])

    async def start(self) -> None:
        await self.venue.start()
        self._tasks = [asyncio.create_task(self.btc_feed.run(), name="btc"),
                       asyncio.create_task(self.depth_feed.run(), name="depth"),
                       asyncio.create_task(self._flush_loop(), name="flush")]

    async def stop(self) -> None:
        self.btc_feed.stop()
        self.depth_feed.stop()
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        await self.venue.stop()
        if self._http is not None:
            await self._http.close()

    # ------------------------------------------------------------------
    async def handle_cmd(self, cmd: dict[str, Any]) -> dict[str, Any]:
        c = cmd.get("cmd")
        v = self.venue
        if c == "arm":
            self.armed = bool(cmd.get("on"))
            self.hub.publish({"t": "armed", "on": self.armed})
            return {"ok": True, "msg": "торговля включена" if self.armed else "торговля выключена"}
        if c == "refresh":
            await v.refresh_account()
            return {"ok": True, "msg": "обновлено"}
        if c in ("buy", "sell", "limit", "cancel") and not self.armed:
            return {"ok": False, "msg": "торговля выключена (переключатель ВКЛ вверху)"}
        try:
            if c == "buy":
                usdt = float(cmd["usdt"])
                if usdt <= 0 or usdt > self.s.max_order_usdt + 1e-9:
                    return {"ok": False, "msg": f"размер {usdt}$ вне лимита (макс {self.s.max_order_usdt}$)"}
                return await v.buy(int(cmd["market_id"]), str(cmd["outcome"]), float(cmd["price"]), usdt)
            if c == "sell":
                return await v.sell(int(cmd["market_id"]), str(cmd["outcome"]), float(cmd["price"]),
                                    float(cmd["shares"]))
            if c == "limit":
                side = str(cmd["side"]).upper()
                shares = float(cmd["shares"])
                price = float(cmd["price"])
                if side == "BUY" and shares * price > self.s.max_order_usdt + 1e-9:
                    return {"ok": False, "msg": f"размер {shares * price:.2f}$ больше лимита {self.s.max_order_usdt}$"}
                return await v.limit(int(cmd["market_id"]), str(cmd["outcome"]), side, price, shares,
                                     bool(cmd.get("post_only")))
            if c == "cancel":
                return await v.cancel([str(h) for h in cmd.get("hashes") or []])
        except OrderRejected as exc:
            return {"ok": False, "msg": f"не отправлено: {exc}"}
        except KeyError as exc:
            return {"ok": False, "msg": f"не отправлено: {exc.args[0] if exc.args else exc}"}
        except Exception as exc:  # noqa: BLE001
            log.exception("command failed")
            return {"ok": False, "msg": f"ошибка: {type(exc).__name__}: {exc}"}
        return {"ok": False, "msg": f"неизвестная команда {c}"}

    async def ws_handler(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=20, max_msg_size=2**20)
        await ws.prepare(request)
        for m in self.hello():
            await ws.send_str(json.dumps(m, separators=(",", ":")))
        self.hub.clients.add(ws)
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                try:
                    cmd = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue
                req = cmd.get("req")

                async def run(cmd: dict[str, Any] = cmd, req: Any = req) -> None:
                    t0 = time.perf_counter()
                    res = await self.handle_cmd(cmd)
                    res.setdefault("total_ms", (time.perf_counter() - t0) * 1000)
                    if not ws.closed:
                        await ws.send_str(json.dumps({"t": "result", "req": req, **res}))
                    if not res.get("ok") and cmd.get("cmd") in ("buy", "sell", "limit", "cancel"):
                        self.venue.log("warning", res.get("msg", ""))

                asyncio.ensure_future(run())
        finally:
            self.hub.clients.discard(ws)
        return ws


def make_app(term: Terminal) -> web.Application:
    app = web.Application()

    async def index(_: web.Request) -> web.FileResponse:
        return web.FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})

    async def on_startup(_: web.Application) -> None:
        await term.start()

    async def on_cleanup(_: web.Application) -> None:
        await term.stop()

    app.router.add_get("/", index)
    app.router.add_get("/ws", term.ws_handler)
    app.router.add_get("/api/klines", term.klines)
    app.router.add_static("/static/", STATIC)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app
