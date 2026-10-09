"""Replay the DOM bot on recordings of dom_recorder.py, for a grid of parameters.

    python dom_replay.py                                    # all files in data/dom, 5m windows
    python dom_replay.py --series 15m --from 2026-10-10 --to 2026-10-17
    python dom_replay.py --walls 150000 --ttls 20 --edges 0.02 --shifts 0     # one configuration

Uses the bot's own code (terminal/dombot.py): the order-book signal detector, the fair model and the entry rule.
Per variant: one position per window, entry = walk of the recorded predict.fun asks `--latency` s after the
signal (limit ask + 1 cent, never above --hi), up to 3 attempts per window, hold to expiry, taker fee withheld in
shares. Outcome = end price vs start price from predict.fun.
Also writes the bot's candidate log for the default configuration and prints the same calibration report as
`run_dombot.py --report`.
Warm-up: nothing is traded in the first 15 min of every continuous piece of recording (volatility needs history).
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import zlib
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from terminal.books import split_book, take_buy  # noqa: E402
from terminal.btc import BtcState  # noqa: E402
from terminal.depth import SpotDepthBook  # noqa: E402
from terminal.dombot import (CandidateLog, DomCfg, FairModel, FlowSignals, SigCfg, evaluate, fee_cut,  # noqa: E402
                             report)
from terminal.markets import MarketInfo, Outcome  # noqa: E402
from terminal.trading import MIN_ORDER_USDT  # noqa: E402
from terminal.venue import BookState  # noqa: E402

KINDS = {"все": ("eaten", "pulled", "imbalance"), "съели+сняли": ("eaten", "pulled"), "съели": ("eaten",),
         "сняли": ("pulled",), "дисб": ("imbalance",)}


# ---------------------------------------------------------------------------- reading
def records(files: list[Path]) -> Iterator[dict[str, Any]]:
    for f in files:
        try:
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        break                       # truncated tail of a file that was being written
        except (EOFError, OSError, zlib.error):
            continue                                # file cut off by a crash: use what was read


def pick_files(d: Path, d_from: str | None, d_to: str | None) -> list[Path]:
    out = []
    for f in sorted(d.glob("dom_*.jsonl.gz")):
        day = f.name[4:12]
        day = f"{day[:4]}-{day[4:6]}-{day[6:8]}"
        if (d_from and day < d_from) or (d_to and day > d_to):
            continue
        out.append(f)
    return out


# ---------------------------------------------------------------------------- variants
@dataclass(frozen=True)
class Var:
    sig_i: int
    kinds_name: str
    ttl: float
    min_edge: float
    shift: float
    require: bool

    def name(self, walls: list[float]) -> str:
        w = f"плотн ≥{walls[self.sig_i] / 1000:.0f}k"
        if not self.require:
            return f"{w} | БЕЗ сигнала (контроль) | край ≥{self.min_edge:.2f}"
        sh = f" | сдвиг {self.shift:g}бп" if self.shift else ""
        return f"{w} | {self.kinds_name} ≤{self.ttl:g}с | край ≥{self.min_edge:.2f}{sh}"


@dataclass
class Pos:
    outcome: str
    t: float
    ask: float
    shares: float
    usdt: float
    end_s: float
    won: bool | None = None
    pnl: float = 0.0


@dataclass
class VarState:
    pos: dict[int, Pos] = field(default_factory=dict)
    pending: dict[int, tuple[float, str, float, float]] = field(default_factory=dict)   # due, outcome, limit, ask
    attempts: dict[int, int] = field(default_factory=dict)
    misses: int = 0


# ---------------------------------------------------------------------------- engine
class Replay:
    def __init__(self, a: argparse.Namespace, out_root: Path) -> None:
        self.a = a
        self.prefix = f"btc-updown-{a.series}"
        period = 300 if a.series == "5m" else 900
        self.base = DomCfg(lo=a.lo, hi=a.hi, max_price=a.hi, series=self.prefix, tte_min=a.tte_min,
                           tte_max=a.tte_max if a.tte_max is not None else period - 30, k=a.k)
        self.btc = BtcState(65)
        self.depth = SpotDepthBook()
        self.fair = FairModel(self.btc, k=a.k)
        self.walls = a.walls
        self.sigs = [FlowSignals(SigCfg(wall_min_usd=w), self.depth) for w in a.walls]
        self.vars: list[Var] = []
        for i in range(len(a.walls)):
            for kn in a.kinds:
                for ttl in a.ttls:
                    for e in a.edges:
                        for sh in a.shifts:
                            self.vars.append(Var(i, kn, ttl, e, sh, True))
            for e in a.edges:
                self.vars.append(Var(i, "все", 0, e, 0, False))
        self.vs = {v: VarState() for v in self.vars}
        self.markets: dict[int, MarketInfo] = {}
        self.books: dict[int, BookState] = {}
        self.last_t = 0.0
        self.last_step = 0.0
        self.warm_from = 0.0
        self.cands = CandidateLog(out_root / "logs" / "dombot_candidates.csv", self.base.resample_s)
        self.steps = 0
        self.t_first = self.t_last = 0.0
        self.gaps = 0

    # -- state updates
    def apply(self, r: dict[str, Any]) -> None:
        k, t = r["k"], float(r["t"])
        if self.t_first == 0:
            self.t_first = t
        if self.last_t and t - self.last_t > 5:            # recorder was down: everything is stale
            self.gaps += 1
            self.depth.synced = False
            self.warm_from = t + 900
            for st in self.vs.values():
                st.pending.clear()
        if self.last_t == 0:
            self.warm_from = t + 900
        self.last_t = self.t_last = t
        if k == "m":
            ms = int(t * 1000)
            self.btc.on_tick(ms, r["m"], r["m"], r["m"])
        elif k == "D":
            self.depth.bids = {float(p): float(q) for p, q in r["b"]}
            self.depth.asks = {float(p): float(q) for p, q in r["a"]}
            self.depth.synced = bool(r.get("ok"))
            self.depth.last_event = t
        elif k == "d":
            if self.depth.synced:
                for side, d in (("b", self.depth.bids), ("a", self.depth.asks)):
                    for p, q in r[side]:
                        if q <= 0:
                            d.pop(float(p), None)
                        else:
                            d[float(p)] = float(q)
                self.depth.last_event = t
        elif k == "x":
            ms = int(t * 1000)
            for side, sell in (("b", True), ("a", False)):
                for p, q in r[side]:
                    self.btc.on_trade(ms, p, q, sell)
                    for fs in self.sigs:
                        fs.on_trade(ms, p, q, sell)
        elif k == "p":
            m = self.markets.get(r["id"])
            up, down = split_book(r["b"], r["a"], m.precision if m else 2)
            self.books[r["id"]] = BookState(up, down, int(t * 1000), t)
        elif k == "M":
            if r["series"] != self.prefix:
                return
            old = self.markets.get(r["id"])
            self.markets[r["id"]] = MarketInfo(
                r["series"], r["series"], r["slug"], r["slug"], 0, r["id"], int(r["s"]), int(r["e"]),
                r.get("sp") if r.get("sp") is not None else (old.start_price if old else None),
                r.get("ep") if r.get("ep") is not None else (old.end_price if old else None),
                r.get("ts") or "", r.get("st") or "", int(r.get("pr") or 2), int(r.get("fee") or 200), False, False,
                Outcome("Up", "", 1), Outcome("Down", "", 2))

    # -- one decision step
    def step(self, now: float) -> None:
        self.steps += 1
        for fs in self.sigs:
            fs.tick(now)
        h = self.btc.hist
        if not math.isnan(self.btc.mid) and (not h or h[-1][0] < int(now)):
            h.append((int(now), self.btc.mid))           # one point per second even when the mid did not move
        if now < self.warm_from or not self.depth.synced or math.isnan(self.btc.mid) \
                or now - self.btc.ts / 1000 > 5 or len(self.btc.hist) < 900:
            return
        S = self.btc.mid
        sigma = self.fair.sigma(now)
        # latest signal time per (kind, dir) for each detector
        last = []
        for fs in self.sigs:
            d: dict[tuple[str, int], float] = {}
            for s in fs.recent:
                if now - s.t <= 120:
                    d[(s.kind, s.dir)] = max(d.get((s.kind, s.dir), 0.0), s.t)
            last.append(d)
        imb = self.sigs[0].imb
        for m in self.markets.values():
            book = self.books.get(m.market_id)
            if book is None:
                continue
            self._fills(m, book, now)
            k_price = m.start_price or self.btc.price_at(m.start_ms // 1000)
            if not k_price:
                continue
            self.cands.update_bids(m.market_id, book)
            base = evaluate(self.base, m, book, now, S, k_price, sigma, list(self.sigs[0].recent), self.fair)
            for c in base:
                self.cands.observe(m, c, now, {"imb10": None if imb is None else round(imb, 3), "flow30": None,
                                               "dist_bp": round((S / k_price - 1) * 1e4, 2),
                                               "sigma_bp": round(sigma * 1e4, 3)})
            if not base:
                continue
            # shared features of in-band sides
            feats = []
            for c in base:
                if c.bid is None or c.ask - c.bid > self.base.max_spread + 1e-9:
                    continue
                d = 1 if c.outcome == "UP" else -1
                shifted = {}
                for sh in self.a.shifts:
                    if sh:
                        pu = self.fair.p_up(S * (1 + d * sh / 1e4), k_price, c.tte, sigma)
                        shifted[sh] = pu if d == 1 else 1 - pu
                feats.append((c, d, shifted))
            if not feats:
                continue
            for v, st in self.vs.items():
                mid = m.market_id
                if mid in st.pos or mid in st.pending or st.attempts.get(mid, 0) >= 3:
                    continue
                lt = last[v.sig_i]
                kinds = KINDS[v.kinds_name]
                best = None
                for c, d, shifted in feats:
                    aligned = any(now - lt.get((kd, d), -1e9) <= v.ttl for kd in kinds) if v.require else False
                    opposed = any(now - lt.get((kd, -d), -1e9) <= v.ttl for kd in kinds) if v.require else False
                    if v.require and (not aligned or opposed):
                        continue
                    p = shifted.get(v.shift, c.p_model) if (aligned and v.shift) else c.p_model
                    edge = p - c.cost_eff
                    if edge >= v.min_edge and (best is None or edge > best[1]):
                        best = (c, edge)
                if best is not None:
                    c = best[0]
                    limit = round(min(c.ask + 10 ** -m.precision, self.a.hi), m.precision)
                    st.pending[mid] = (now + self.a.latency, c.outcome, limit, c.ask)
                    st.attempts[mid] = st.attempts.get(mid, 0) + 1

    def _fills(self, m: MarketInfo, book: BookState, now: float) -> None:
        mid = m.market_id
        for st in self.vs.values():
            p = st.pending.get(mid)
            if p is None or now < p[0]:
                continue
            del st.pending[mid]
            _, o, limit, ask = p
            f = take_buy(book.outcome(o).asks, limit, budget_usdt=self.a.stake)
            if f.usdt < MIN_ORDER_USDT or now >= m.end_ms / 1000:
                st.misses += 1
                continue
            st.pos[mid] = Pos(o, now, ask, f.shares, f.usdt, m.end_ms / 1000)

    def run(self, recs: Iterator[dict[str, Any]]) -> None:
        dt = self.a.step
        for r in recs:
            t = float(r["t"])
            if self.last_step == 0:
                self.last_step = t
            while t - self.last_step >= dt and t - self.last_step < 5:
                self.last_step += dt
                self.step(self.last_step)
            if t - self.last_step >= 5:
                self.last_step = t
            self.apply(r)
            if r["k"] == "M" and r.get("ep") is not None:
                self._resolve_cands(r["id"])

    def _resolve_cands(self, market_id: int) -> None:
        m = self.markets.get(market_id)
        if m is None or market_id not in self.cands.rows:
            return
        up = self.winner_up(m)
        if up is not None:
            self.cands.resolve(market_id, up)

    def winner_up(self, m: MarketInfo) -> bool | None:
        sp, ep = m.start_price, m.end_price
        if sp is None or ep is None or sp == ep:
            return None
        return ep > sp

    def settle(self) -> None:
        for mid in list(self.cands.rows):
            self._resolve_cands(mid)
        for st in self.vs.values():
            for mid, p in st.pos.items():
                m = self.markets.get(mid)
                up = self.winner_up(m) if m else None
                if up is None:
                    continue
                p.won = up if p.outcome == "UP" else not up
                avg = p.usdt / p.shares
                net = p.shares * (1 - fee_cut(self.base.fee_rate, avg))
                p.pnl = (net if p.won else 0.0) - p.usdt


# ---------------------------------------------------------------------------- report
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def var_stats(ps: list[Pos], cut: float, fee_rate: float) -> dict[str, Any]:
    done = [p for p in ps if p.won is not None]
    n = len(done)
    if n == 0:
        return {"n": 0}
    cost = sum(p.usdt for p in done)
    pnl = sum(p.pnl for p in done)
    w = sum(p.won for p in done)
    h1 = [p for p in done if p.t < cut]
    h2 = [p for p in done if p.t >= cut]
    days: dict[str, float] = defaultdict(float)
    for p in done:
        days[datetime.fromtimestamp(p.t, timezone.utc).strftime("%Y-%m-%d")] += p.pnl
    avg = sum(p.usdt / p.shares for p in done) / n
    be = sum((p.usdt / p.shares) / (1 - fee_cut(fee_rate, p.usdt / p.shares)) for p in done) / n
    return {"n": n, "w": w, "avg": avg, "be": be, "roi": pnl / cost, "pnl": pnl,
            "h1": sum(p.pnl for p in h1) / max(1e-9, sum(p.usdt for p in h1)),
            "h2": sum(p.pnl for p in h2) / max(1e-9, sum(p.usdt for p in h2)),
            "dpos": sum(v > 0 for v in days.values()), "days": len(days)}


HDR = (f"{'вариант':<58}{'сделок':>7}{'цена':>7}{'безуб.':>8}{'выигр':>8}  {'95% дов.':<11}{'ROI':>8}"
       f"{'1-я пол':>9}{'2-я пол':>9}{'дни+':>8}{'мимо':>6}")


def fmt(name: str, s: dict[str, Any], misses: int) -> str:
    if s["n"] == 0:
        return f"{name:<58}{0:>7}{'':>66}{misses:>6}"
    lo, hi = wilson(s["w"], s["n"])
    ci = f"[{lo:.0%}-{hi:.0%}]"
    return (f"{name:<58}{s['n']:>7}{s['avg']:>7.3f}{s['be']:>8.1%}{s['w'] / s['n']:>8.1%}  {ci:<11}{s['roi']:>+8.1%}"
            f"{s['h1']:>+9.1%}{s['h2']:>+9.1%}{s['dpos']:>4}/{s['days']:<3}{misses:>6}")


def floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(HERE / "data" / "dom"))
    ap.add_argument("--out", default=str(HERE / "out" / "replay"))
    ap.add_argument("--series", default="5m", choices=("5m", "15m"))
    ap.add_argument("--from", dest="d_from", default=None, help="YYYY-MM-DD (UTC)")
    ap.add_argument("--to", dest="d_to", default=None, help="YYYY-MM-DD (UTC), включительно")
    ap.add_argument("--lo", type=float, default=0.10)
    ap.add_argument("--hi", type=float, default=0.30)
    ap.add_argument("--tte-min", type=float, default=30)
    ap.add_argument("--tte-max", type=float, default=None)
    ap.add_argument("--k", type=float, default=1.0, help="масштаб волатильности (возьми из dog_history.py)")
    ap.add_argument("--stake", type=float, default=1.1)
    ap.add_argument("--latency", type=float, default=0.25, help="задержка от решения до исполнения, с")
    ap.add_argument("--step", type=float, default=0.25)
    ap.add_argument("--walls", default="150000,300000", help="минимальные плотности BTCUSDT, $ (через запятую)")
    ap.add_argument("--kinds", default=",".join(KINDS), help="наборы сигналов: " + ", ".join(KINDS))
    ap.add_argument("--ttls", default="10,20,40")
    ap.add_argument("--edges", default="0,0.02,0.05")
    ap.add_argument("--shifts", default="0,3")
    a = ap.parse_args()
    a.walls, a.ttls, a.edges, a.shifts = floats(a.walls), floats(a.ttls), floats(a.edges), floats(a.shifts)
    a.kinds = [k.strip() for k in a.kinds.split(",") if k.strip()]
    bad = [k for k in a.kinds if k not in KINDS]
    if bad:
        ap.error(f"неизвестные наборы сигналов: {bad}")
    files = pick_files(Path(a.dir), a.d_from, a.d_to)
    if not files:
        print(f"нет записей в {a.dir} — сначала запусти dom_recorder.py")
        return 1
    out_root = Path(a.out)
    cand = out_root / "logs" / "dombot_candidates.csv"
    if cand.exists():
        cand.unlink()
    rp = Replay(a, out_root)
    print(f"файлов {len(files)} ({files[0].name} … {files[-1].name}), вариантов {len(rp.vars)}", flush=True)
    rp.run(records(files))
    rp.settle()
    hours = (rp.t_last - rp.t_first) / 3600
    windows = sum(1 for m in rp.markets.values() if rp.winner_up(m) is not None)
    print(f"записано {hours:.1f} ч, разрывов записи {rp.gaps}, окон {a.series} с итогом {windows}\n")
    cut = rp.t_first + (rp.t_last - rp.t_first) / 2
    rows = []
    for v in rp.vars:
        st = rp.vs[v]
        s = var_stats(list(st.pos.values()), cut, rp.base.fee_rate)
        rows.append((v, s, st.misses))
    print("Безубыток = цена с комиссией. Смотри: «выигр» выше «безуб.», ROI > 0 в ОБЕИХ половинах, и сравни с")
    print("контролем «БЕЗ сигнала»: если с сигналом не лучше, стакан ничего не добавляет. Вариантов много —")
    print("лучший по ROI почти всегда завышен случайно.\n")
    print(HDR)
    print("-" * len(HDR))
    for v, s, miss in rows:
        print(fmt(v.name(a.walls), s, miss))
    top = sorted((r for r in rows if r[1]["n"] >= 30), key=lambda r: -r[1]["roi"])[:10]
    if top:
        print("\nЛучшие 10 по ROI (≥ 30 сделок):")
        print(HDR)
        for v, s, miss in top:
            print(fmt(v.name(a.walls), s, miss))
    out_root.mkdir(parents=True, exist_ok=True)
    with (out_root / "trades.csv").open("w", encoding="utf-8") as fh:
        fh.write("variant,time,outcome,ask,avg,shares,usdt,won,pnl\n")
        for v in rp.vars:
            for mid, p in rp.vs[v].pos.items():
                fh.write(f"\"{v.name(a.walls)}\",{datetime.fromtimestamp(p.t, timezone.utc).isoformat()},{p.outcome},"
                         f"{p.ask},{p.usdt / p.shares:.4f},{p.shares:.4f},{p.usdt:.4f},"
                         f"{'' if p.won is None else int(p.won)},{p.pnl:.4f}\n")
    print(f"\nсделки: {out_root / 'trades.csv'}")
    print("\nКалибровка (как run_dombot.py --report), правило бота по умолчанию:\n")
    print(report(out_root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
