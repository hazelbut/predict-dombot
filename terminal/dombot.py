"""DOM bot: cheap outcomes (0.10-0.30) on predict.fun BTC Up/Down, confirmed by the Binance BTCUSDT order book.

Hybrid of xau_terminal (order-book signals EATEN / PULLED / IMBALANCE) and the predict.fun execution of
pf_sniper / predict_terminal (same keys, same API).

Entry rule for one side of a window (all must hold):
  * best ask of the side in [lo, hi] (default 0.10-0.30), spread <= max_spread, time to expiry in [tte_min, tte_max];
  * a fresh order-book signal on Binance spot BTCUSDT points towards this side (UP <- price up, DOWN <- price down)
    and no fresh signal points against it (switch off with require_signal=False);
  * model probability of the side minus the effective cost (ask grossed up for the taker fee) >= min_edge.
    Model: P(up) = Phi( ln(S/K) / (k * sigma * sqrt(tau)) ), S = Binance mid (optionally shifted by shift_bps towards
    the signal), K = window start price, sigma = per-second realised vol (blend of 15 and 60 min), tau = seconds left.
Hold to expiry, one entry per window, risk caps of the favourite bot (daily / total loss, checkpoints, STOP file).

Every time a side sits in the band the bot also records a CANDIDATE row (features + highest bid seen afterwards +
final outcome) to logs/dombot_candidates.csv, traded or not. `run_dombot.py --report` turns that into calibration:
how often a side bought at price p actually wins, with and without an order-book signal. That table, not the bot's
PnL over a few days, is the answer to "do cheap outcomes with a DOM signal pay".
"""

from __future__ import annotations

import csv
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .btc import BtcState
from .depth import SpotDepthBook
from .favbot import BotCfg, FavBot, Ledger
from .markets import MarketInfo

log = logging.getLogger("dombot")


# ============================================================================ order-book signals (from xau_terminal)
@dataclass
class SigCfg:
    zone_bps: float = 1.0            # wall zone width (BTC ~ $11 per zone)
    range_bps: float = 25.0          # look for walls within +-25 bp of the mid
    wall_mult: float = 5.0           # wall = zone >= average zone x N ...
    wall_min_usd: float = 150_000.0  # ... and at least this much
    window_s: float = 10.0           # how fast a wall must disappear
    eat_pct: float = 0.5             # EATEN: lost >= 50% of the peak ...
    eat_traded: float = 0.6          # ... and >= 60% of the loss were market trades into the zone
    pull_pct: float = 0.6            # PULLED: lost >= 60% of the peak ...
    pull_traded: float = 0.2         # ... with <= 20% of the loss traded
    pull_min_dist_bps: float = 1.0   # cancels right at the touch are normal quoting, not a pull
    imb_bps: float = 10.0            # IMBALANCE: bids vs asks within +-10 bp of the mid ...
    imb_signal: float = 0.6          # ... |imbalance| >= 0.6 ...
    imb_hold_s: float = 3.0          # ... held for >= 3 s
    cooldown_s: float = 30.0


@dataclass
class Sig:
    t: float
    kind: str          # eaten / pulled / imbalance
    dir: int           # +1 price expected up, -1 down
    price: float
    mid: float
    usd: float = 0.0


class FlowSignals:
    """Port of xau_terminal FlowDetector working on the Binance spot BTCUSDT book of predict_terminal.
    eaten     - a large wall is eaten by market orders (break in the direction of eating: bids eaten -> down);
    pulled    - a large wall vanished without trades (fake support/resistance: bid pulled -> down);
    imbalance - bids vs asks near the mid strongly skewed and holding."""

    def __init__(self, cfg: SigCfg, book: SpotDepthBook, tick: float = 0.01) -> None:
        self.c = cfg
        self.book = book
        self.tick_size = tick
        self.w: float | None = None
        self.walls: dict[tuple[str, int], dict[str, Any]] = {}
        self.trades: dict[tuple[str, int], deque] = {}
        self.cool: dict[Any, float] = {}
        self.recent: deque[Sig] = deque(maxlen=500)
        self.imb: float | None = None
        self.imb_dir = 0
        self.imb_since = 0.0

    def bucket(self, p: float) -> int:
        assert self.w is not None
        return int(p // self.w)

    def on_trade(self, ts_ms: int, p: float, q: float, sell: bool) -> None:
        if self.w is None:
            return
        key = ("bid" if sell else "ask", self.bucket(p))    # a market sell hits bids, a market buy hits asks
        dq = self.trades.get(key)
        if dq is None:
            dq = self.trades[key] = deque()
        dq.append((ts_ms / 1000, q))

    def _traded(self, key: tuple[str, int], since: float) -> float:
        dq = self.trades.get(key)
        return sum(q for t, q in dq if t >= since) if dq else 0.0

    def active(self, now: float, ttl: float) -> list[Sig]:
        return [s for s in self.recent if now - s.t <= ttl]

    def tick(self, now: float | None = None) -> list[Sig]:
        now = now or time.time()
        c, bk = self.c, self.book
        if not bk.synced or not bk.bids or not bk.asks:
            return []
        bb, ba = max(bk.bids), min(bk.asks)
        if ba <= bb:
            return []
        mid = (bb + ba) / 2
        if self.w is None:
            self.w = max(self.tick_size, round(mid * c.zone_bps / 1e4 / self.tick_size) * self.tick_size)
        out: list[Sig] = []
        win = c.window_s
        rng = mid * c.range_bps / 1e4
        ib = mid * c.imb_bps / 1e4
        zones: dict[str, dict[int, list[float]]] = {"bid": {}, "ask": {}}
        imb_b = imb_a = 0.0
        for side, levels, lo, hi in (("bid", bk.bids, mid - rng, mid), ("ask", bk.asks, mid, mid + rng)):
            z = zones[side]
            for p, q in levels.items():
                if p < lo or p > hi:
                    continue
                b = self.bucket(p)
                e = z.get(b)
                if e is None:
                    z[b] = [q, p, q]
                else:
                    e[0] += q
                    if q > e[2]:
                        e[1], e[2] = p, q
                if abs(p - mid) <= ib:
                    if side == "bid":
                        imb_b += q
                    else:
                        imb_a += q
        self.imb = (imb_b - imb_a) / (imb_b + imb_a) if imb_b + imb_a > 0 else None
        allq = [e[0] for z in zones.values() for e in z.values()]
        if allq:
            avg = sum(allq) / len(allq)
            thr = max(avg * c.wall_mult, c.wall_min_usd / mid)
            for side, z in zones.items():
                for b, (q, px, lvl) in z.items():
                    if q >= thr and (side, b) not in self.walls:
                        self.walls[(side, b)] = {"hist": deque(), "px": px, "lvl": lvl}
        for key, w in list(self.walls.items()):
            side, b = key
            e = zones[side].get(b)
            q = e[0] if e else 0.0
            if e and e[2] >= 0.5 * w["lvl"]:
                w["px"], w["lvl"] = e[1], max(w["lvl"], e[2])
            elif abs(w["px"] - mid) > rng * 0.9 and self._traded(key, now - win) == 0:
                del self.walls[key]          # price simply moved away from the wall
                continue
            h = w["hist"]
            h.append((now, q))
            while h and h[0][0] < now - win:
                h.popleft()
            peak_t, peak = max(h, key=lambda x: x[1])
            if peak <= 0 or (q == 0 and now - peak_t > win):
                del self.walls[key]
                continue
            drop = peak - q
            if drop <= 0 or peak * mid < c.wall_min_usd:
                continue
            traded = self._traded(key, peak_t - 0.5)
            dist_bps = abs(w["px"] - mid) / mid * 1e4
            kind = None
            if drop >= c.eat_pct * peak and traded >= c.eat_traded * drop:
                kind = "eaten"
            elif drop >= c.pull_pct * peak and traded <= c.pull_traded * drop and dist_bps >= c.pull_min_dist_bps:
                kind = "pulled"
            if kind and now - self.cool.get((kind, key), 0) >= c.cooldown_s:
                self.cool[(kind, key)] = now
                out.append(Sig(now, kind, -1 if side == "bid" else 1, w["px"], mid, peak * mid))
                del self.walls[key]
        if self.imb is not None and abs(self.imb) >= c.imb_signal:
            d = 1 if self.imb > 0 else -1
            if d != self.imb_dir:
                self.imb_dir, self.imb_since = d, now
            elif now - self.imb_since >= c.imb_hold_s and now - self.cool.get(("imbalance", d), 0) >= c.cooldown_s:
                self.cool[("imbalance", d)] = now
                out.append(Sig(now, "imbalance", d, mid, mid))
        else:
            self.imb_dir, self.imb_since = 0, 0.0
        for key in list(self.trades):
            dq = self.trades[key]
            while dq and dq[0][0] < now - win - 2:
                dq.popleft()
            if not dq:
                del self.trades[key]
        self.recent.extend(out)
        return out


# ============================================================================ fair probability (from pf_backtest)
def ncdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


class FairModel:
    """Per-second realised vol of Binance mid: 0.5 x var(last 15 min) + 0.5 x var(last 60 min)."""

    def __init__(self, btc: BtcState, k: float = 1.0, floor: float = 1e-5) -> None:
        self.btc = btc
        self.k = k
        self.floor = floor
        self._cache = (0, floor)

    def sigma(self, now: float) -> float:
        sec = int(now)
        if self._cache[0] == sec:
            return self._cache[1]
        h = list(self.btc.hist)
        v = []
        for win in (900, 3600):
            s2 = dt = 0.0
            for (t0, p0), (t1, p1) in zip(h, h[1:]):
                if t1 < sec - win or p0 <= 0 or p1 <= 0:
                    continue
                lr = math.log(p1 / p0)
                s2 += lr * lr
                dt += t1 - t0
            v.append(s2 / dt if dt > 0 else self.floor ** 2)
        sig = max(math.sqrt(0.5 * v[0] + 0.5 * v[1]), self.floor)
        self._cache = (sec, sig)
        return sig

    def p_up(self, s: float, k_price: float, tau: float, sigma: float) -> float:
        tau = max(tau, 0.5)
        return ncdf(math.log(s / k_price) / (self.k * sigma * math.sqrt(tau)))


# ============================================================================ decision
@dataclass
class DomCfg(BotCfg):
    lo: float = 0.10
    hi: float = 0.30
    max_price: float | None = 0.30
    tte_min: float = 30.0            # seconds to expiry: not later than this ...
    tte_max: float = 270.0           # ... and not earlier than this (5m window)
    min_edge: float = 0.02           # model probability - effective cost >= 2 cents
    require_signal: bool = True
    kinds: tuple = ("eaten", "pulled", "imbalance")
    sig_ttl: float = 20.0            # a signal is "fresh" for this many seconds
    shift_bps: float = 0.0           # move S towards an aligned signal by N bp before the model (0 = no guess)
    k: float = 1.0
    resample_s: float = 15.0         # candidate rows: every N s while a side stays in the band
    checkpoints: tuple = ((300, -0.10), (1000, 0.0))


@dataclass
class Cand:
    outcome: str
    ask: float
    bid: float | None
    tte: float
    p_model: float          # model probability of this side, no shift
    p_adj: float            # with shift_bps towards an aligned signal
    cost_eff: float         # ask grossed up for the taker fee (USDT per net share)
    edge: float             # p_adj - cost_eff
    aligned: str            # kinds of fresh signals towards this side, e.g. "eaten+imbalance"
    opposed: str            # kinds of fresh signals against it
    take: bool
    reason: str


def fee_cut(fee_rate: float, ask: float) -> float:
    return fee_rate * min(ask, 1 - ask) / ask if ask > 0 else 0.0


def evaluate(cfg: DomCfg, m: MarketInfo, book: Any, now: float, s: float, k_price: float | None, sigma: float,
             sigs: list[Sig], fair: FairModel) -> list[Cand]:
    out: list[Cand] = []
    tte = m.end_ms / 1000 - now
    if (m.trading_status not in ("", "OPEN") or not (cfg.tte_min <= tte <= cfg.tte_max) or not k_price
            or not s or math.isnan(s)):
        return out
    for o in ("UP", "DOWN"):
        ob = book.outcome(o)
        a, b = ob.best_ask, ob.best_bid
        if a is None or not (cfg.lo - 1e-9 <= a <= cfg.hi + 1e-9):
            continue
        d = 1 if o == "UP" else -1
        fresh = [x for x in sigs if x.kind in cfg.kinds and now - x.t <= cfg.sig_ttl]
        aligned = "+".join(sorted({x.kind for x in fresh if x.dir == d}))
        opposed = "+".join(sorted({x.kind for x in fresh if x.dir == -d}))
        pu = fair.p_up(s, k_price, tte, sigma)
        p_model = pu if o == "UP" else 1 - pu
        if aligned and cfg.shift_bps:
            pu2 = fair.p_up(s * (1 + d * cfg.shift_bps / 1e4), k_price, tte, sigma)
            p_adj = pu2 if o == "UP" else 1 - pu2
        else:
            p_adj = p_model
        cost = a / (1 - fee_cut(cfg.fee_rate, a))
        edge = p_adj - cost
        if b is None or a - b > cfg.max_spread + 1e-9:
            take, why = False, "широкий спред"
        elif cfg.require_signal and not aligned:
            take, why = False, "нет сигнала стакана"
        elif opposed:
            take, why = False, f"сигнал против ({opposed})"
        elif edge < cfg.min_edge:
            take, why = False, f"мал край {edge:+.3f}"
        else:
            take, why = True, "ok"
        out.append(Cand(o, a, b, tte, p_model, p_adj, cost, edge, aligned, opposed, take, why))
    return out


def flow_imb(btc: BtcState, now: float, secs: int = 30) -> float | None:
    """Market-order imbalance on Binance spot over the last `secs` seconds (by notional)."""
    buy = sell = 0.0
    s0 = int(now) - secs
    for s in range(s0, int(now) + 1):
        r = btc.flow.get(s)
        if r:
            buy += r[1]
            sell += r[3]
    return (buy - sell) / (buy + sell) if buy + sell > 0 else None


# ============================================================================ candidate log
CAND_FIELDS = ["time", "market_id", "slug", "series", "outcome", "why_row", "tte", "ask", "bid", "p_model", "p_adj",
               "cost_eff", "edge", "aligned", "opposed", "imb10", "flow30", "dist_bp", "sigma_bp", "take", "reason",
               "max_bid_after", "won"]


class CandidateLog:
    """Rows are kept in memory until the window resolves, then written with the outcome."""

    def __init__(self, path: Path, resample_s: float) -> None:
        self.path = path
        self.resample_s = resample_s
        self.rows: dict[int, list[dict[str, Any]]] = {}       # market_id -> rows
        self.meta: dict[int, MarketInfo] = {}
        self.last: dict[tuple[int, str], tuple[float, str]] = {}   # (market, outcome) -> (ts, aligned)
        self.took: set[tuple[int, str]] = set()                      # rule fired at least once (row written)
        self.written = 0

    def observe(self, m: MarketInfo, c: Cand, now: float, extra: dict[str, Any]) -> None:
        key = (m.market_id, c.outcome)
        prev = self.last.get(key)
        if prev is None:
            why = "first"
        elif c.take and key not in self.took:
            why = "take"
        elif c.aligned and c.aligned != prev[1]:
            why = "signal"
        elif now - prev[0] >= self.resample_s:
            why = "resample"
        else:
            return
        self.last[key] = (now, c.aligned)
        if c.take:
            self.took.add(key)
        self.meta[m.market_id] = m
        row = {"time": datetime.fromtimestamp(now, timezone.utc).isoformat(timespec="seconds"),
               "market_id": m.market_id, "slug": m.slug, "series": m.series, "outcome": c.outcome, "why_row": why,
               "tte": round(c.tte, 1), "ask": c.ask, "bid": c.bid, "p_model": round(c.p_model, 4),
               "p_adj": round(c.p_adj, 4), "cost_eff": round(c.cost_eff, 4), "edge": round(c.edge, 4),
               "aligned": c.aligned, "opposed": c.opposed, "take": int(c.take), "reason": c.reason,
               "max_bid_after": c.bid or 0.0, "won": "", **extra}
        self.rows.setdefault(m.market_id, []).append(row)

    def update_bids(self, market_id: int, book: Any) -> None:
        for r in self.rows.get(market_id, ()):
            b = book.outcome(r["outcome"]).best_bid
            if b is not None and b > r["max_bid_after"]:
                r["max_bid_after"] = b

    def due(self, now: float) -> list[MarketInfo]:
        return [m for mid, m in self.meta.items() if now >= m.end_ms / 1000 + 15 and mid in self.rows]

    def resolve(self, market_id: int, up_won: bool | None) -> None:
        rows = self.rows.pop(market_id, [])
        self.meta.pop(market_id, None)
        for k in [k for k in self.last if k[0] == market_id]:
            del self.last[k]
        self.took = {k for k in self.took if k[0] != market_id}
        if up_won is None or not rows:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists()
        with self.path.open("a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=CAND_FIELDS, extrasaction="ignore")
            if new:
                w.writeheader()
            for r in rows:
                r["won"] = int(up_won if r["outcome"] == "UP" else not up_won)
                w.writerow(r)
        self.written += len(rows)


# ============================================================================ bot
class DomBot(FavBot):
    def __init__(self, venue: Any, cfg: DomCfg, root: Path, btc: BtcState, depth: SpotDepthBook,
                 sig_cfg: SigCfg | None = None) -> None:
        super().__init__(venue, cfg, root)
        self.cfg: DomCfg = cfg
        mode = "live" if cfg.live else "paper"
        self.led = Ledger.load(root / "logs" / f"dombot_{mode}.json")
        self.csv = root / "logs" / f"dombot_{mode}.csv"
        self.stop_file = root / "logs" / "dombot.STOP"
        self.btc = btc
        self.depth = depth
        self.flow = FlowSignals(sig_cfg or SigCfg(), depth)
        self.fair = FairModel(btc, k=cfg.k)
        self.cands = CandidateLog(root / "logs" / "dombot_candidates.csv", cfg.resample_s)
        self.sig_count: dict[str, int] = {}
        self._last_prune = 0.0
        self._feed_warned = 0.0

    def on_btc_trade(self, ts_ms: int, price: float, qty: float, sell: bool) -> None:
        self.flow.on_trade(ts_ms, price, qty, sell)

    def _feeds_ok(self, now: float) -> str:
        if math.isnan(self.btc.mid) or now - self.btc.ts / 1000 > 5:
            return "нет свежей цены BTC с Binance"
        if not self.depth.synced or now - self.depth.last_event > 5:
            return "стакан BTCUSDT не синхронизирован"
        if len(self.btc.hist) < 900:
            return f"мало истории для волатильности ({len(self.btc.hist)} с из 900)"
        return ""

    async def tick(self, now: float | None = None) -> None:
        now = now or time.time()
        if self.cfg.live:
            self._confirm(now)
        if now - self._last_prune > 30 and not math.isnan(self.btc.mid):
            self._last_prune = now
            self.depth.prune(self.btc.mid)
        for s in self.flow.tick(now):
            self.sig_count[s.kind] = self.sig_count.get(s.kind, 0) + 1
            log.info("СИГНАЛ %s %s цена %.2f%s", s.kind, "▲" if s.dir > 0 else "▼", s.price,
                     f" плотность ${s.usd:,.0f}" if s.usd else f" дисб {self.flow.imb:+.2f}")
        bad = self._feeds_ok(now)
        if bad:
            if now - self._feed_warned > 30:
                self._feed_warned = now
                log.warning("жду данные: %s", bad)
            return
        S = self.btc.mid
        sigma = self.fair.sigma(now)
        sigs = list(self.flow.recent)
        extra = {"imb10": None if self.flow.imb is None else round(self.flow.imb, 3),
                 "flow30": None, "sigma_bp": round(sigma * 1e4, 3)}
        f30 = flow_imb(self.btc, now)
        extra["flow30"] = None if f30 is None else round(f30, 3)
        for m in list(self.v.markets.values()):
            if m.series != self.cfg.series:
                continue
            book = self.v.books.get(m.market_id)
            if book is None:
                continue
            self.cands.update_bids(m.market_id, book)
            k_price = m.start_price or self.btc.price_at(m.start_ms // 1000)
            cands = evaluate(self.cfg, m, book, now, S, k_price, sigma, sigs, self.fair)
            for c in cands:
                self.cands.observe(m, c, now, {**extra, "dist_bp": round((S / k_price - 1) * 1e4, 2)})
            if self.led.has(m.market_id) or m.market_id in self.pending:
                continue
            w = self.win.get(m.market_id)
            if w is not None and w["blocked"]:
                continue
            best = max((c for c in cands if c.take), key=lambda c: c.edge, default=None)
            if best is None:
                continue
            gate = self._gate(now)
            if gate:
                continue
            log.info("КАНДИДАТ %s %s аск %.2f модель %.3f край %+.3f сигнал %s tte %.0fс", m.slug, best.outcome,
                     best.ask, best.p_adj, best.edge, best.aligned or "-", best.tte)
            await self.enter(m, best.outcome, best.ask, book, now)

    def _gate(self, now: float) -> str:
        from .favbot import risk_gate
        gate = risk_gate(self.cfg, self.led, now, self.v.account.balance, self.stop_file,
                         sum(p[0].cost for p in self.pending.values()))
        if gate != self.last_gate:
            if gate:
                log.warning(gate)
            self.last_gate = gate
        return gate

    async def settle(self, now: float | None = None) -> None:
        await super().settle(now)
        now = now or time.time()
        for m in self.cands.due(now):
            up = await self._winner(SimpleNamespace(market_id=m.market_id, slug=m.slug, outcome="UP"))
            if up is None and now < m.end_ms / 1000 + 3600:
                continue
            self.cands.resolve(m.market_id, up)

    def status_line(self) -> str:
        sc = ", ".join(f"{k} {v}" for k, v in sorted(self.sig_count.items())) or "нет"
        return f"{super().status_line()} | сигналов: {sc} | кандидатов записано {self.cands.written}"


# ============================================================================ report
def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def report(root: Path, tp_mult: float = 2.0) -> str:
    path = root / "logs" / "dombot_candidates.csv"
    if not path.exists():
        return f"нет данных: {path} (бот пишет строки после закрытия окон)"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    rows = [r for r in rows if r.get("won") in ("0", "1")]
    out = [f"Кандидатов с итогом: {len(rows)}, окон: {len({r['market_id'] for r in rows})}", ""]

    def first_per(rows_: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen, res = set(), []
        for r in rows_:
            k = (r["market_id"], r["outcome"])
            if k not in seen:
                seen.add(k)
                res.append(r)
        return res

    def line(name: str, rs: list[dict[str, Any]]) -> str:
        n = len(rs)
        if n == 0:
            return f"{name:<34}{0:>6}"
        wins = sum(int(r["won"]) for r in rs)
        cost = sum(float(r["cost_eff"]) for r in rs)
        avg_cost = cost / n
        lo, hi = _wilson(wins, n)
        roi = wins / cost - 1 if cost else 0.0
        pm = sum(float(r["p_model"]) for r in rs) / n
        tp = sum(1 for r in rs if float(r["max_bid_after"] or 0) >= min(0.99, float(r["ask"]) * tp_mult)) / n
        return (f"{name:<34}{n:>6}{avg_cost:>8.3f}{wins / n:>8.1%}  [{lo:.0%}-{hi:.0%}]{pm:>8.3f}{roi:>+9.1%}"
                f"{tp:>9.0%}")

    hdr = f"{'группа (1 строка на окно/сторону)':<34}{'n':>6}{'цена':>8}{'выигр':>8}  {'95% дов.':<9}{'модель':>8}" \
          f"{'ROI':>9}{f'TP x{tp_mult:g}':>9}"
    out += ["Безубыток: доля выигрышей = цене (с комиссией). ROI > 0 только если «выигр» заметно выше «цена».",
            "TP: доля, где bid после входа дошёл до цены × множитель (можно было бы продать раньше).", "", hdr,
            "-" * len(hdr)]
    out.append(line("все кандидаты", first_per(rows)))
    for lo, hi in ((0.10, 0.15), (0.15, 0.20), (0.20, 0.25), (0.25, 0.301)):
        out.append(line(f"  цена {lo:.2f}-{min(hi, 0.30):.2f}",
                        first_per([r for r in rows if lo <= float(r["ask"]) < hi])))
    sig = [r for r in rows if r["aligned"]]
    out.append(line("с сигналом стакана (за сторону)", first_per(sig)))
    out.append(line("без сигнала", first_per([r for r in rows if not r["aligned"] and not r["opposed"]])))
    out.append(line("сигнал против", first_per([r for r in rows if r["opposed"]])))
    for kind in ("eaten", "pulled", "imbalance"):
        out.append(line(f"  сигнал {kind}", first_per([r for r in rows if kind in r["aligned"].split("+")])))
    out.append(line("модель: край > 0", first_per([r for r in rows if float(r["edge"]) > 0])))
    out.append(line("модель: край ≤ 0", first_per([r for r in rows if float(r["edge"]) <= 0])))
    out.append(line("правило бота (take=1)", first_per([r for r in rows if r["take"] == "1"])))
    for lo, hi in ((0, 60), (60, 120), (120, 200), (200, 10_000)):
        out.append(line(f"  до конца {lo}-{hi if hi < 10_000 else '∞'} с",
                        first_per([r for r in rows if lo <= float(r["tte"]) < hi])))
    out += ["", "Меньше ~200 строк в группе — шум: смотри на доверительный интервал, а не на ROI."]
    for mode in ("paper", "live"):
        p = root / "logs" / f"dombot_{mode}.json"
        if p.exists():
            st = Ledger.load(p).stats()
            out.append(f"Сделки бота ({mode}): {st['n']}, побед {st['wins']}, PnL {st['pnl']:+.2f}$, "
                       f"ROI {st['roi'] * 100:+.1f}%")
    return "\n".join(out)
