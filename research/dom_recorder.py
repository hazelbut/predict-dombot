"""Record what the DOM bot sees, so the bot can be replayed on it later (dom_replay.py).

    python dom_recorder.py               # runs until stopped, files in data/dom/
    python dom_recorder.py --check       # 30 s test: prints what arrives from each source

Read-only: only PREDICT_API_KEY is needed (no wallet key, no orders). Records every --step seconds (0.25):
  * Binance spot BTCUSDT order book, $1 buckets within +-RANGE bp of the mid (changes only, full snapshot each min);
  * Binance aggressor trades, summed per bucket and step (which book side was hit);
  * Binance mid (bookTicker);
  * predict.fun BTC Up/Down books (Up-quoted, top 15 levels) on every change, market info (start / end price).
Files: data/dom/dom_YYYYMMDD_HH.jsonl.gz (UTC hour); the total size is logged every minute.
Run it 24/7 for several days (tmux / systemd): the replay needs hundreds of windows to say anything.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import logging
import math
import signal
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from terminal.btc import BinanceFeed, BtcState  # noqa: E402
from terminal.config import load_settings, parse_series  # noqa: E402
from terminal.depth import BinanceDepthFeed, SpotDepthBook  # noqa: E402

log = logging.getLogger("recorder")


def buckets(levels: dict[float, float], lo: float, hi: float, bid: bool) -> dict[float, float]:
    """$1 buckets: bids floor, asks ceil (a bucket never crosses the spread)."""
    out: dict[float, float] = {}
    for p, q in levels.items():
        if lo <= p <= hi:
            k = float(math.floor(p + 1e-9) if bid else math.ceil(p - 1e-9))
            out[k] = out.get(k, 0.0) + q
    return {k: round(v, 5) for k, v in out.items() if round(v, 5) > 0}


def diff(prev: dict[float, float], cur: dict[float, float]) -> list[list[float]]:
    ch = [[p, q] for p, q in cur.items() if prev.get(p) != q]
    ch += [[p, 0] for p in prev if p not in cur]
    return ch


class Recorder:
    """Pure writer: give it the live state objects, call step(now) on a timer."""

    def __init__(self, out_dir: Path, btc: BtcState, depth: SpotDepthBook, venue: Any, range_bps: float = 40.0,
                 full_every: float = 60.0) -> None:
        self.dir = out_dir
        self.btc, self.depth, self.v = btc, depth, venue
        self.range_bps = range_bps
        self.full_every = full_every
        self.fh: Any = None
        self.hour = ""
        self.prev_b: dict[float, float] = {}
        self.prev_a: dict[float, float] = {}
        self.depth_ok = False
        self.last_full = 0.0
        self.last_mid = 0.0
        self.last_mid_t = 0.0
        self.book_ts: dict[int, tuple] = {}
        self.mkt: dict[int, tuple] = {}
        self.tr_b: dict[float, float] = {}       # aggressor SELLS hit bids: bucket -> qty
        self.tr_a: dict[float, float] = {}       # aggressor BUYS hit asks
        self.lines = 0
        self.last_flush = 0.0

    # ---- inputs
    def on_trade(self, ts_ms: int, price: float, qty: float, sell: bool) -> None:
        if sell:
            k = float(math.floor(price + 1e-9))
            self.tr_b[k] = self.tr_b.get(k, 0.0) + qty
        else:
            k = float(math.ceil(price - 1e-9))
            self.tr_a[k] = self.tr_a.get(k, 0.0) + qty

    # ---- output
    def _w(self, rec: dict[str, Any]) -> None:
        self.fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
        self.lines += 1

    def _open(self, now: float) -> None:
        hour = time.strftime("%Y%m%d_%H", time.gmtime(now))
        if hour == self.hour and self.fh is not None:
            return
        if self.fh is not None:
            self.fh.close()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.fh = gzip.open(self.dir / f"dom_{hour}.jsonl.gz", "at", encoding="utf-8", compresslevel=6)
        self.hour = hour
        # each file starts self-contained: markets, books, full depth
        self.mkt.clear()
        self.book_ts.clear()
        self.last_full = 0.0
        self.last_mid = 0.0

    def close(self) -> None:
        if self.fh is not None:
            self.fh.close()
            self.fh = None

    def step(self, now: float | None = None) -> None:
        now = now or time.time()
        self._open(now)
        t = round(now, 3)
        # markets
        for m in list(self.v.markets.values()):
            key = (m.start_price, m.end_price, m.trading_status, m.status)
            if self.mkt.get(m.market_id) != key:
                self.mkt[m.market_id] = key
                self._w({"k": "M", "t": t, "id": m.market_id, "slug": m.slug, "series": m.series, "s": m.start_ms,
                         "e": m.end_ms, "sp": m.start_price, "ep": m.end_price, "ts": m.trading_status,
                         "st": m.status, "pr": m.precision, "fee": m.fee_bps})
        # predict.fun books (Up-quoted)
        for mid, b in list(self.v.books.items()):
            key = (b.ts, b.received)
            if self.book_ts.get(mid) == key:
                continue
            self.book_ts[mid] = key
            self._w({"k": "p", "t": t, "id": mid, "b": [list(x) for x in b.up.bids[:15]],
                     "a": [list(x) for x in b.up.asks[:15]]})
        # Binance mid
        mid_px = self.btc.mid
        fresh = not math.isnan(mid_px) and now - self.btc.ts / 1000 < 5
        if fresh and (mid_px != self.last_mid or now - self.last_mid_t >= 2):
            self.last_mid, self.last_mid_t = mid_px, now
            self._w({"k": "m", "t": t, "m": round(mid_px, 2)})
        # trades
        if self.tr_b or self.tr_a:
            self._w({"k": "x", "t": t, "b": [[p, round(q, 5)] for p, q in self.tr_b.items()],
                     "a": [[p, round(q, 5)] for p, q in self.tr_a.items()]})
            self.tr_b, self.tr_a = {}, {}
        # depth
        ok = self.depth.synced and now - self.depth.last_event < 5 and fresh
        if not ok:
            if self.depth_ok:
                self._w({"k": "D", "t": t, "ok": 0, "b": [], "a": []})
                self.depth_ok = False
                self.prev_b, self.prev_a = {}, {}
        else:
            rng = mid_px * self.range_bps / 1e4
            cb = buckets(self.depth.bids, mid_px - rng, mid_px, True)
            ca = buckets(self.depth.asks, mid_px, mid_px + rng, False)
            if not self.depth_ok or now - self.last_full >= self.full_every:
                self._w({"k": "D", "t": t, "ok": 1, "b": [[p, q] for p, q in cb.items()],
                         "a": [[p, q] for p, q in ca.items()]})
                self.last_full = now
                self.depth_ok = True
            else:
                db, da = diff(self.prev_b, cb), diff(self.prev_a, ca)
                if db or da:
                    self._w({"k": "d", "t": t, "b": db, "a": da})
            self.prev_b, self.prev_a = cb, ca
        if now - self.last_flush > 5:
            self.fh.flush()
            self.last_flush = now


async def amain(a: argparse.Namespace) -> int:
    from terminal.venue_live import DataVenue
    s = load_settings(str(ROOT / ".env"), require_wallet=False)
    s = __import__("dataclasses").replace(s, series=parse_series(a.series))
    btc = BtcState(5)
    depth = SpotDepthBook()
    venue = DataVenue(s, lambda msg: None)
    rec = Recorder(Path(a.out), btc, depth, venue, a.range_bps)
    orig = btc.on_trade

    def hook(ts_ms: int, price: float, qty: float, sell: bool) -> None:
        orig(ts_ms, price, qty, sell)
        rec.on_trade(ts_ms, price, qty, sell)
    btc.on_trade = hook                       # type: ignore[method-assign]
    feeds = [BinanceFeed(s.binance_ws_url, s.binance_rest_url, btc, 5),
             BinanceDepthFeed(s.binance_depth_ws_url, s.binance_rest_url, depth)]
    tasks = [asyncio.create_task(f.run()) for f in feeds]
    await venue.start()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sg in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sg, stop.set)
        except NotImplementedError:
            pass
    if a.check:
        loop.call_later(30, stop.set)
    log.info("запись в %s каждые %.2f с, стакан ±%.0f б.п., рынки %s", a.out, a.step, a.range_bps, a.series)
    last_status = last_prune = time.time()
    n0 = 0
    while not stop.is_set():
        t0 = time.time()
        try:
            if not math.isnan(btc.mid) and t0 - last_prune > 30:
                last_prune = t0
                depth.prune(btc.mid)
            rec.step(t0)
        except Exception as exc:  # noqa: BLE001
            log.exception("запись: %s", exc)
        if t0 - last_status >= (10 if a.check else 60):
            last_status = t0
            size = sum(p.stat().st_size for p in Path(a.out).glob("dom_*.jsonl.gz")) / 1e6
            log.info("строк +%d | BTC %s | стакан %s | рынков %d, стаканов predict %d | файлы %.1f МБ",
                     rec.lines - n0, f"{btc.mid:.2f}" if not math.isnan(btc.mid) else "нет",
                     "OK" if depth.synced else "нет", len(venue.markets), len(venue.books), size)
            n0 = rec.lines
        try:
            await asyncio.wait_for(stop.wait(), timeout=max(0.01, a.step - (time.time() - t0)))
        except asyncio.TimeoutError:
            pass
    for f in feeds:
        f.stop()
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await venue.stop()
    rec.close()
    if a.check:
        ok = {"BTC": not math.isnan(btc.mid), "стакан BTCUSDT": depth.synced, "рынки predict.fun": bool(venue.markets),
              "стаканы predict.fun": bool(venue.books)}
        for k, v in ok.items():
            print(f"  {k:<22} {'OK' if v else 'НЕТ'}")
        print("всё работает — запускай без --check" if all(ok.values()) else "что-то не пришло — смотри лог выше")
        return 0 if all(ok.values()) else 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(HERE / "data" / "dom"))
    ap.add_argument("--series", default="btc-updown-5m:300,btc-updown-15m:900")
    ap.add_argument("--step", type=float, default=0.25)
    ap.add_argument("--range-bps", type=float, default=40.0, help="стакан BTC в ±N б.п. от середины")
    ap.add_argument("--check", action="store_true", help="30-секундная проверка источников")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("venue", "predict_api", "aiohttp", "btc", "depth"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return asyncio.run(amain(a))


if __name__ == "__main__":
    sys.exit(main())
