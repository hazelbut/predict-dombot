"""Backtest of volatility trading inside predict.fun BTC Up/Down windows on real trade history.

    python backtest.py                     # full report
    python backtest.py --series 5m

Price evidence comes from real prints (collect.py). Rules are deliberately conservative:
  * taker entry: an outcome print where someone BOUGHT at <= X proves the ask was there; we pay that
    price + --slip (we come after that buyer) plus the taker fee measured from the data;
  * maker limit buy at L fills only if somebody later SOLD below L (price went through our level);
  * maker limit sell at E fills only if somebody later BOUGHT above E;
  * whatever is not sold by expiry settles at 1 (won) or 0 (lost).
A Down print at p is the same book as an Up print at 1-p (the book is quoted in Up/Yes).
"""

from __future__ import annotations

import argparse
import csv
import math
import sqlite3
import sys
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

HERE = Path(__file__).resolve().parent
OUTS = ("UP", "DOWN")


# ---------------------------------------------------------------------------- data
class Stream:
    """One outcome's prints: times (s from start), prices in that outcome's terms, isA = someone BOUGHT this
    outcome (ask print); otherwise someone sold it (bid print)."""
    __slots__ = ("ts", "px", "isA", "sh")

    def __init__(self, ts: list[float], px: list[float], isA: list[bool], sh: list[float] | None = None) -> None:
        self.ts = np.asarray(ts, dtype=np.float64)
        self.px = np.asarray(px, dtype=np.float64)
        self.isA = np.asarray(isA, dtype=bool)
        self.sh = np.asarray(sh if sh is not None else [1.0] * len(ts), dtype=np.float64)


@dataclass(slots=True)
class Market:
    market_id: int
    start_s: int
    end_s: int
    winner: str
    ev: dict[str, Stream] = field(default_factory=dict)

    @property
    def day(self) -> str:
        return datetime.fromtimestamp(self.start_s, timezone.utc).strftime("%Y-%m-%d")


def build_events(rows: Iterable[tuple], start_s: int) -> dict[str, Stream]:
    """rows: (ts_ms, outcome, taker side, price[, shares]) -> per-outcome event streams."""
    raw: dict[str, tuple[list, list, list, list]] = {o: ([], [], [], []) for o in OUTS}
    for row in sorted(rows):
        ts_ms, outcome, side, price = row[:4]
        shares = row[4] if len(row) > 4 else 1.0
        if outcome not in OUTS:
            continue
        t = ts_ms / 1000 - start_s
        bought = side == "BUY"
        for o in OUTS:
            same = o == outcome
            raw[o][0].append(t)
            raw[o][1].append(price if same else round(1 - price, 3))
            raw[o][2].append(bought if same else not bought)
            raw[o][3].append(shares)
    return {o: Stream(*raw[o]) for o in OUTS}


def load(db_path: Path, series: str) -> tuple[list[Market], "FeeModel"]:
    if not db_path.exists():
        sys.exit(f"нет базы {db_path} — сначала запусти collect.py")
    db = sqlite3.connect(db_path)
    ms = db.execute("SELECT market_id, start_s, end_s, winner, fee_bps FROM markets "
                    "WHERE series=? AND complete=1 AND n_trades>0 ORDER BY start_s", (series,)).fetchall()
    info = {mid: (s0, s1, win) for mid, s0, s1, win, _ in ms if win in OUTS}
    built: dict[int, Market] = {}
    cur_id, rows = None, []

    def flush() -> None:
        if cur_id in info and rows:
            s0, s1, win = info[cur_id]
            built[cur_id] = Market(cur_id, s0, s1, win, build_events(rows, s0))

    n = 0
    for mid, ts, o, side, p, sh in db.execute("SELECT market_id, ts_ms, outcome, side, price, shares FROM trades "
                                              "ORDER BY market_id, ts_ms"):
        if mid != cur_id:
            flush()
            cur_id, rows = mid, []
        if mid in info:
            rows.append((ts, o, side, p, sh))
        n += 1
        if n % 200_000 == 0:
            print(f"  загружено сделок {n}", file=sys.stderr, flush=True)
    flush()
    out = [built[mid] for mid, *_ in ms if mid in built]
    fee = FeeModel.from_db(db, [m[4] for m in ms])
    return out, fee


class FeeModel:
    """predict.fun taker fee: rate x min(p, 1-p) per share. On buys it is taken in SHARES (you receive fewer
    shares), on sells in USDT. The rate and the form are measured from real fills."""

    def __init__(self, rate: float, buy_in_shares: bool, note: str) -> None:
        self.rate, self.buy_in_shares, self.note = rate, buy_in_shares, note

    @classmethod
    def from_db(cls, db: sqlite3.Connection, fee_bps: list[int]) -> "FeeModel":
        bps = sorted(b for b in fee_bps if b) or [200]
        default = bps[len(bps) // 2] / 10_000
        rates: list[float] = []
        n_buy = n_buy_sh = 0
        for p, sh, fee, ft, side in db.execute(
                "SELECT price, shares, fee, fee_type, side FROM trades WHERE fee > 0 AND shares > 0 LIMIT 200000"):
            val = fee * p if ft == "SHARES" else fee
            base = sh * min(p, 1 - p)
            if base > 0:
                rates.append(val / base)
            if side == "BUY":
                n_buy += 1
                n_buy_sh += ft == "SHARES"
        if len(rates) >= 30:
            rates.sort()
            rate = rates[len(rates) // 2]
            in_sh = n_buy_sh * 2 >= n_buy
            note = (f"{rate:.2%} × min(p,1−p) за акцию (измерено по {len(rates)} сделкам), при покупке "
                    + ("удерживается акциями" if in_sh else "в USDT"))
        else:
            rate, in_sh = default, True
            note = f"{rate:.2%} × min(p,1−p) за акцию (feeRateBps рынка, в сделках данных нет), при покупке акциями"
        return cls(rate, in_sh, note)

    def per_share(self, p: float) -> float:
        """fee in USDT per share traded at price p"""
        return self.rate * min(p, 1 - p)

    def share_cut(self, p: float) -> float:
        """fraction of bought shares withheld as the fee"""
        return self.per_share(p) / p if p > 0 else 0.0


def settle_buy(fee: "FeeModel", taker: bool, entry: float, sh: float, ex: float, out_fee_per_share: float,
               stake: float) -> tuple[float, float]:
    """(pnl, fee paid in USDT-equivalent) for `sh` shares bought at `entry` and closed at `ex`."""
    if not taker:
        net, f_in = sh, 0.0
    elif fee.buy_in_shares:
        net = sh * (1 - fee.share_cut(entry))
        f_in = (sh - net) * ex                   # withheld shares would have been worth ex
    else:
        net, f_in = sh, fee.per_share(entry) * sh
    f_out = out_fee_per_share * net
    pnl = net * ex - stake - (0.0 if fee.buy_in_shares and taker else f_in) - f_out
    return pnl, f_in + f_out


# ---------------------------------------------------------------------------- fills
def first_after(ev: Stream, t_from: float, t_to: float, kind: str, lo: float = -1.0, hi: float = 2.0) -> int:
    """Index of the first `kind` print (A = buy of this outcome, B = sell) in [t_from, t_to] with lo <= price <= hi."""
    i0 = int(np.searchsorted(ev.ts, t_from, "left"))
    i1 = int(np.searchsorted(ev.ts, t_to, "right"))
    if i1 <= i0:
        return -1
    px = ev.px[i0:i1]
    k = ev.isA[i0:i1] if kind == "A" else ~ev.isA[i0:i1]
    hit = k & (px >= lo) & (px <= hi)
    j = int(hit.argmax())
    return i0 + j if hit[j] else -1


@dataclass(slots=True)
class Trade:
    strat: str
    market_id: int
    day: str
    outcome: str
    t_entry: float
    entry: float
    shares: float
    exit: float          # exit price (target) or 1/0 at settlement
    how: str             # target | settle_win | settle_loss | stop
    t_exit: float
    fee: float
    pnl: float
    level: float = 0.0

    @property
    def cost(self) -> float:
        return self.entry * self.shares


def exit_leg(m: Market, o: str, t_entry: float, entry: float, target: float, stop_s: float | None,
             latency: float, slip: float, fee: FeeModel, touch: bool = False) -> tuple[float, str, float, float]:
    """Limit sell at entry+target; else settle (or market-sell `stop_s` before the end)."""
    ev, dur = m.ev[o], m.end_s - m.start_s
    tgt = round(entry + target, 3)
    t_last = dur - (stop_s or 0)
    if target > 0 and tgt < 1:
        j = first_after(ev, t_entry + latency, t_last, "A", lo=(tgt - 1e-9) if touch else (tgt + 1e-9))
        if j >= 0:
            return tgt, "target", float(ev.ts[j]), 0.0
    if stop_s:
        # sell into the last bid print before the stop time
        k1 = int(np.searchsorted(ev.ts, t_last, "left"))
        bids = np.nonzero(~ev.isA[:k1])[0]
        if len(bids) and ev.ts[bids[-1]] > t_entry:
            p = max(0.0, float(ev.px[bids[-1]]) - slip)
            return p, "stop", t_last, fee.per_share(p)
    win = m.winner == o
    return (1.0 if win else 0.0), ("settle_win" if win else "settle_loss"), dur, 0.0


def run_fade(m: Market, c: dict[str, Any], fee: FeeModel) -> list[Trade]:
    """Taker: the first time one side is offered at lo..X inside [t_from, t_to], buy it."""
    best: tuple[float, str, int] | None = None
    for o in OUTS:
        i = first_after(m.ev[o], c["t_from"], c["t_to"], "A", lo=c["lo"], hi=c["x"])
        if i >= 0 and (best is None or m.ev[o].ts[i] < best[0]):
            best = (float(m.ev[o].ts[i]), o, i)
    if best is None:
        return []
    t, o, i = best
    entry = min(0.99, round(float(m.ev[o].px[i]) + c["slip"], 3))
    sh = c["stake"] / entry
    ex, how, tx, f_out_unit = exit_leg(m, o, t, entry, c["target"], c.get("stop_s"), c["latency"],
                                       c["slip"], fee)
    pnl, f = settle_buy(fee, True, entry, sh, ex, f_out_unit, c["stake"])
    return [Trade(c["name"], m.market_id, m.day, o, t, entry, sh, ex, how, tx, f, pnl)]


def run_grid(m: Market, c: dict[str, Any], fee: FeeModel) -> list[Trade]:
    """Maker: resting bids at every level on both sides; each filled level re-offers at level+target."""
    out = []
    for o in OUTS:
        for lv in c["levels"]:
            touch = c.get("touch", False)
            i = first_after(m.ev[o], c["t_from"], c["t_to"], "B", hi=(lv + 1e-9) if touch else (lv - 1e-9))
            if i < 0:
                continue
            t = float(m.ev[o].ts[i])
            sh = c["stake"] / lv
            ex, how, tx, f_out_unit = exit_leg(m, o, t, lv, c["target"], c.get("stop_s"), c["latency"], c["slip"], fee,
                                               touch)
            pnl, f = settle_buy(fee, False, lv, sh, ex, f_out_unit, c["stake"])
            out.append(Trade(c["name"], m.market_id, m.day, o, t, lv, sh, ex, how, tx, f, pnl, lv))
    return out


# ---------------------------------------------------------------------------- stats
def stats(trades: list[Trade], markets: list[Market]) -> dict[str, Any]:
    n = len(trades)
    cost = sum(t.cost for t in trades)
    pnl = sum(t.pnl for t in trades)
    days: dict[str, float] = defaultdict(float)
    for m in markets:
        days[m.day] += 0.0
    for t in trades:
        days[t.day] += t.pnl
    eq = peak = dd = 0.0
    for t in sorted(trades, key=lambda x: (x.market_id, x.t_entry)):
        eq += t.pnl
        peak = max(peak, eq)
        dd = min(dd, eq - peak)
    return {"n": n, "pnl": pnl, "roi": pnl / cost if cost else 0.0,
            "hit": sum(t.how == "target" for t in trades) / n if n else 0.0,
            "loss0": sum(t.how == "settle_loss" for t in trades) / n if n else 0.0,
            "avg": pnl / n if n else 0.0, "dd": dd,
            "days_pos": sum(v > 0 for v in days.values()), "days": len(days),
            "worst_day": min(days.values()) if days else 0.0}


def halves(trades: list[Trade], markets: list[Market]) -> tuple[float, float]:
    if not markets:
        return 0.0, 0.0
    cut = markets[len(markets) // 2].start_s
    ids1 = {m.market_id for m in markets if m.start_s < cut}
    a = [t for t in trades if t.market_id in ids1]
    b = [t for t in trades if t.market_id not in ids1]
    return (sum(t.pnl for t in a) / max(1e-9, sum(t.cost for t in a)),
            sum(t.pnl for t in b) / max(1e-9, sum(t.cost for t in b)))


def fmt_row(name: str, s: dict[str, Any], h: tuple[float, float]) -> str:
    return (f"{name:<44} {s['n']:>5} {s['hit']:>6.0%} {s['loss0']:>6.0%} {s['avg']:>+7.3f} {s['pnl']:>+9.2f} "
            f"{s['roi']:>+7.1%} {h[0]:>+7.1%} {h[1]:>+7.1%} {s['dd']:>+8.2f} {s['days_pos']:>3}/{s['days']:<3}")


HDR = (f"{'стратегия':<44} {'сделок':>5} {'цель':>6} {'в ноль':>6} {'$/сд':>7} {'PnL $':>9} {'ROI':>7} "
       f"{'1-я пол':>7} {'2-я пол':>7} {'просад':>8} {'дни+':>7}")


# ---------------------------------------------------------------------------- reports
def calibration(markets: list[Market], fee: FeeModel, t_from: float, t_to: float, delta: float) -> None:
    print(f"\n=== 1. Честна ли цена? Первая покупка стороны в корзине цены, t = {t_from:.0f}–{t_to:.0f} с от старта ===")
    print("   «выиграла» выше «ср.цены» = сторону недооценивают, это и есть край; «край-ком» — после комиссии тейкера")
    print(f"   «дошла+{delta:.2f}» — как часто после такой покупки кто-то покупал эту сторону дороже на {delta:.2f}+")
    print(f"{'цена':>9} {'n':>6} {'ср.цена':>8} {'выиграла':>9} {'край':>7} {'край-ком':>9} {'дошла+' + f'{delta:.2f}':>11}")
    for b in range(1, 19):
        lo, hi = b * 0.05, (b + 1) * 0.05
        n = win = touch = 0
        sp = 0.0
        for m in markets:
            for o in OUTS:
                i = first_after(m.ev[o], t_from, t_to, "A", lo=lo, hi=hi - 1e-9)
                if i < 0:
                    continue
                p = float(m.ev[o].px[i])
                n += 1
                sp += p
                win += int(m.winner == o)
                if p + delta < 1:
                    touch += int(first_after(m.ev[o], float(m.ev[o].ts[i]) + 1, m.end_s - m.start_s, "A",
                                             lo=p + delta + 1e-9) >= 0)
        if n < 20:
            continue
        ap = sp / n
        edge = win / n - ap
        print(f"{lo:>4.2f}-{hi:<4.2f} {n:>6} {ap:>8.3f} {win / n:>9.1%} {edge:>+7.3f} {win / n * (1 - fee.share_cut(ap)) - ap:>+9.3f} "
              f"{touch / n:>11.1%}")


def per_level(trades: list[Trade]) -> None:
    by: dict[float, list[Trade]] = defaultdict(list)
    for t in trades:
        by[t.level].append(t)
    print(f"{'уровень':>9} {'входов':>7} {'цель':>6} {'в ноль':>7} {'PnL $':>9} {'ROI':>7}")
    for lv in sorted(by, reverse=True):
        ts = by[lv]
        cost = sum(t.cost for t in ts)
        pnl = sum(t.pnl for t in ts)
        print(f"{lv:>9.2f} {len(ts):>7} {sum(t.how == 'target' for t in ts) / len(ts):>6.0%} "
              f"{sum(t.how == 'settle_loss' for t in ts) / len(ts):>7.0%} {pnl:>+9.2f} {pnl / cost:>+7.1%}")


def run_favdip(m: Market, c: dict[str, Any], fee: FeeModel) -> list[Trade]:
    """Maker on a pullback: once a side has traded at >= L + arm, rest a bid at L; filled when sellers come
    through it (strict: price below L, touch: at L). Hold to expiry."""
    out = []
    for o in OUTS:
        ev = m.ev[o]
        for lv in c["levels"]:
            j = first_after(ev, c["t_from"], c["t_to"], "A", lo=lv + c["arm"] - 1e-9)
            if j < 0:
                continue
            t_arm = float(ev.ts[j])
            i = first_after(ev, t_arm + c["latency"], c["t_to"], "B",
                            hi=(lv + 1e-9) if c.get("touch") else (lv - 1e-9))
            if i < 0:
                continue
            sh = c["stake"] / lv
            win = m.winner == o
            pnl, f = settle_buy(fee, False, lv, sh, 1.0 if win else 0.0, 0.0, c["stake"])
            pnl += 0.25 * fee.per_share(lv) * sh          # maker rebate
            out.append(Trade(c["name"], m.market_id, m.day, o, float(ev.ts[i]), lv, sh, 1.0 if win else 0.0,
                             "settle_win" if win else "settle_loss", float(m.end_s - m.start_s), f, pnl, lv))
    return out


def favorite_section(markets: list[Market], fee: FeeModel, base: dict[str, Any], t_froms: list[int],
                     t_tos: list[int], series: str, out: Path) -> None:
    """Reverse of idea 1: buy the side the market already favours and hold to expiry."""
    def run(cfgs: list[dict], fn) -> list[tuple[dict, list[Trade], dict, tuple]]:
        res = []
        for k, c in enumerate(cfgs, 1):
            if k % 20 == 0 or k == len(cfgs):
                print(f"  посчитано вариантов {k}/{len(cfgs)}", file=sys.stderr, flush=True)
            tr = [t for m in markets for t in fn(m, c, fee)]
            res.append((c, tr, stats(tr, markets), halves(tr, markets)))
        return res

    bands = [(0.50, 0.60), (0.55, 0.65), (0.60, 0.70), (0.65, 0.75), (0.70, 0.80), (0.75, 0.85), (0.80, 0.90),
             (0.55, 0.80)]
    takers = []
    for lo, hi in bands:
        for slip in (0.0, 0.01):
            for tf in t_froms:
                for tt in t_tos:
                    takers.append({**base, "slip": slip, "x": hi, "lo": lo, "target": 0.0, "t_from": tf, "t_to": tt,
                                   "name": f"тейкер {lo:.2f}-{hi:.2f} +{slip:.2f} t{tf}-{tt}"})
    makers = []
    for levels in ([0.55, 0.60, 0.65], [0.65, 0.70, 0.75], [0.75, 0.80, 0.85], [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]):
        for arm in (0.03, 0.06):
            for touch in (False, True):
                for tt in t_tos:
                    makers.append({**base, "levels": levels, "arm": arm, "t_from": t_froms[0], "t_to": tt,
                                   "touch": touch,
                                   "name": f"откат {levels[0]:.2f}-{levels[-1]:.2f} взвод+{arm:.2f} до t{tt}"
                                           + (" [кас]" if touch else " [скв]")})
    rt = run(takers, run_fade)
    rm = run(makers, run_favdip)
    print("\n=== 4. Реверс: покупаем ФАВОРИТА (сторону дороже 0.50) и держим до конца окна ===")
    print("   тейкер: первая сделка-покупка стороны в диапазоне цены; +0.00 = успел по той же цене, +0.01 = на цент хуже")
    print(HDR)
    for c, tr, st, h in sorted([r for r in rt if r[2]["n"] >= 100], key=lambda r: -r[2]["roi"])[:16]:
        print(fmt_row(c["name"], st, h))
    print("-- те же диапазоны, худшие 4:")
    for c, tr, st, h in sorted([r for r in rt if r[2]["n"] >= 100], key=lambda r: r[2]["roi"])[:4]:
        print(fmt_row(c["name"], st, h))
    print("\n   мейкер на откате: сторона поторговалась ≥ L+взвод → ставим лимитку на покупку по L; держим до конца;")
    print("   без комиссии, с ребейтом 25%. [скв] — цена прошла сквозь L, [кас] — исполнено при касании L")
    print(HDR)
    for c, tr, st, h in sorted([r for r in rm if r[2]["n"] >= 100], key=lambda r: -r[2]["roi"]):
        print(fmt_row(c["name"], st, h))
    best = max((r for r in rm if not r[0]["touch"]), key=lambda r: r[2]["roi"])
    print(f"\n-- по уровням: {best[0]['name']}")
    per_level(best[1])
    print("\nВерить можно только варианту, который в плюсе в ОБЕИХ половинах и в большинстве дней; лучший из десятков "
          "вариантов по одной половине — это случайность.")
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"fav_{series}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["strategy", "n", "roi", "roi_h1", "roi_h2", "days_pos", "days"])
        for c, tr, st, h in rt + rm:
            w.writerow([c["name"], st["n"], round(st["roi"], 4), round(h[0], 4), round(h[1], 4), st["days_pos"],
                        st["days"]])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", default="5m")
    ap.add_argument("--db", default=str(HERE / "data" / "pf_hist.sqlite3"))
    ap.add_argument("--stake", type=float, default=10.0, help="$ на вход / на уровень сетки")
    ap.add_argument("--slip", type=float, default=0.01, help="ухудшение цены тейкер-входа, 0.01 = 1 цент")
    ap.add_argument("--latency", type=float, default=1.0, help="с между входом и выставлением выхода")
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--fav", action="store_true", help="только реверс: покупка фаворита")
    a = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    print("загрузка истории…", file=sys.stderr, flush=True)

    markets, fee = load(Path(a.db), a.series)
    if len(markets) < 50:
        print(f"мало рынков в базе ({len(markets)}), нужно хотя бы 50 — докачай collect.py")
        return 1
    dur = markets[0].end_s - markets[0].start_s
    d0, d1 = markets[0].day, markets[-1].day
    print(f"BTC {a.series}: {len(markets)} рынков {d0} … {d1}, UP выиграл {sum(m.winner == 'UP' for m in markets)}")
    print(f"комиссия тейкера {fee.note}; мейкер 0")
    print(f"ставка ${a.stake:g} на вход, проскальзывание тейкера +{a.slip:.2f}, задержка {a.latency:g} с")

    # price convention check: prints in the last 20 s must point to the winner
    ok = tot = 0
    for m in markets:
        ts, px = m.ev["UP"].ts, m.ev["UP"].px
        k = bisect_left(ts, dur - 20)
        if k < len(ts) and ts[-1] <= dur + 5 and abs(px[-1] - 0.5) > 0.2:
            tot += 1
            ok += int((px[-1] > 0.5) == (m.winner == "UP"))
    if tot and ok / tot < 0.8:
        print(f"!!! цены не сходятся с исходами ({ok}/{tot}) — формат данных другой, результаты недостоверны")
        return 2
    if tot:
        print(f"проверка данных: цена в конце окна совпадает с исходом в {ok / tot:.0%} рынков")

    calibration(markets, fee, 60, dur - 120, 0.10)

    base = {"stake": a.stake, "slip": a.slip, "latency": a.latency}
    fades, grids = [], []
    t_froms = [60, 120, 180] if dur >= 900 else [30, 60]
    t_tos = [dur // 2, dur - 180] if dur >= 900 else [dur // 2, dur - 60]
    if a.fav:
        favorite_section(markets, fee, base, t_froms, t_tos, a.series, Path(a.out))
        return 0
    for x in (0.40, 0.35, 0.30, 0.25, 0.20):
        for target in (0.05, 0.10, 0.15, 0.0):
            for tf in t_froms:
                for tt in t_tos:
                    tname = f"+{target:.2f}" if target else "до конца"
                    fades.append({**base, "name": f"тейкер ≤{x:.2f} {tname} t{tf}-{tt}", "x": x, "lo": 0.05,
                                  "target": target, "t_from": tf, "t_to": tt})
    levels_all = [round(0.40 - 0.05 * i, 2) for i in range(7)]   # 0.40 … 0.10
    for touch in (False, True):
        for target in (0.05, 0.10, 0.15):
            for tt in t_tos:
                for stop in (None, 60):
                    grids.append({**base, "levels": levels_all, "target": target, "t_from": 0, "t_to": tt,
                                  "stop_s": stop, "touch": touch,
                                  "name": f"сетка 40-10 +{target:.2f} до t{tt}" + (" стоп-1м" if stop else "")
                                          + (" [касание]" if touch else " [сквозь]")})

    def run(cfgs: list[dict], fn) -> list[tuple[dict, list[Trade], dict, tuple]]:
        res = []
        for k, c in enumerate(cfgs, 1):
            if k % 20 == 0 or k == len(cfgs):
                print(f"  посчитано вариантов {k}/{len(cfgs)}", file=sys.stderr, flush=True)
            tr = [t for m in markets for t in fn(m, c, fee)]
            res.append((c, tr, stats(tr, markets), halves(tr, markets)))
        return res

    rf = run(fades, run_fade)
    rg = run(grids, run_grid)
    Path(a.out).mkdir(parents=True, exist_ok=True)

    print("\n=== 2. Твоя идея №1: купить подешевевшую сторону тейкером, выйти лимиткой +Δ (или держать до конца) ===")
    print("   «цель» — доля сделок, закрытых по цели; «в ноль» — сторона проиграла и сгорела; ROI — PnL / вложено")
    print(HDR)
    mine = [r for r in rf if r[0]["x"] == 0.35 and r[0]["target"] in (0.10, 0.15) and r[0]["t_from"] == t_froms[1]]
    for c, tr, s, h in mine:
        print(fmt_row(c["name"], s, h))
    print("-- лучшие 12 из всех вариантов по ROI (при ≥100 сделок):")
    for c, tr, s, h in sorted([r for r in rf if r[2]["n"] >= 100], key=lambda r: -r[2]["roi"])[:12]:
        print(fmt_row(c["name"], s, h))
    print("-- худшие 3:")
    for c, tr, s, h in sorted([r for r in rf if r[2]["n"] >= 100], key=lambda r: r[2]["roi"])[:3]:
        print(fmt_row(c["name"], s, h))

    print("\n=== 3. Твоя идея №2: сетка лимиток 0.40…0.10 на обе стороны, выход +Δ (мейкер, без комиссии) ===")
    print("   [сквозь] — лимитка исполнена, только если цена прошла сквозь уровень (пессимистично, без очереди)")
    print("   [касание] — исполнена при первой сделке на уровне (оптимистично: как будто ты первый в очереди)")
    print("   правда где-то между ними")
    print(HDR)
    for c, tr, s, h in rg:
        print(fmt_row(c["name"], s, h))
    for touch in (False, True):
        g = next(r for r in rg if abs(r[0]["target"] - 0.10) < 1e-9 and r[0]["t_to"] == t_tos[-1]
                 and not r[0]["stop_s"] and r[0]["touch"] == touch)
        print(f"\n-- по уровням: {g[0]['name']}")
        per_level(g[1])
        if not touch:
            g_main = g

    # trades of the headline configs for a closer look
    keep = mine[:1] + [g_main] + sorted([r for r in rf if r[2]["n"] >= 100], key=lambda r: -r[2]["roi"])[:1]
    path = Path(a.out) / f"trades_{a.series}.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["strategy", "market_id", "day", "outcome", "level", "t_entry_s", "entry", "shares", "exit", "how",
                    "t_exit_s", "fee", "pnl"])
        for c, tr, _, _ in keep:
            for t in tr:
                w.writerow([t.strat, t.market_id, t.day, t.outcome, t.level, round(t.t_entry, 1), t.entry,
                            round(t.shares, 3), t.exit, t.how, round(t.t_exit, 1), round(t.fee, 4), round(t.pnl, 4)])
    with (Path(a.out) / f"summary_{a.series}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["strategy", "n", "hit", "loss0", "avg", "pnl", "roi", "roi_h1", "roi_h2", "dd", "days_pos", "days"])
        for c, tr, s, h in rf + rg:
            w.writerow([c["name"], s["n"], round(s["hit"], 4), round(s["loss0"], 4), round(s["avg"], 4),
                        round(s["pnl"], 2), round(s["roi"], 4), round(h[0], 4), round(h[1], 4), round(s["dd"], 2),
                        s["days_pos"], s["days"]])
    print(f"\nсделки: {path}\nвсе варианты: {Path(a.out) / f'summary_{a.series}.csv'}")
    print("Смотри на колонки «1-я пол» и «2-я пол»: если плюс только в одной половине — это подгонка, не край.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
