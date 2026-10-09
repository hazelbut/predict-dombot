"""History backtest of the cheap-side rule (0.10-0.30) on real predict.fun prints.

    python dog_history.py                       # 5m windows, everything in data/pf_hist.sqlite3
    python dog_history.py --series 15m --lo 0.10 --hi 0.30 --slip 0.01

What it can test on history (and what it cannot):
  * the PRICE: does a side bought at 0.10-0.30 win more often than it costs (calibration, after the taker fee);
  * the MODEL filter of the DOM bot: fair P(side) from Binance 1 s prices (1 s before the print) minus the cost;
  * the market-order FLOW of Binance BTCUSDT (aggressor buys vs sells over the last N seconds, from aggTrades
    archives, btc_flow.py) - the part of the order-book picture that has a public history;
  * NOT the resting book (walls eaten / pulled, bid-ask imbalance): there is no public history of it.
    That part is tested with dom_recorder.py (record) + dom_replay.py (replay the bot).

Fills are conservative: a print where somebody BOUGHT the side at price p proves the ask was there; we pay
p + --slip (we come after that buyer) plus the measured taker fee, hold to expiry. One trade per window
(the earliest qualifying print of either side), like the bot.
The volatility scale k of the model is fitted on the FIRST half of the windows only.
"""

from __future__ import annotations

import argparse
import csv
import math
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

import backtest as bt
from fair import Btc

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------- data
class Flow:
    """Dense per-second cumulative sums of aggressor buy / sell notional."""

    def __init__(self, ts: np.ndarray, buy: np.ndarray, sell: np.ndarray) -> None:
        self.t0 = int(ts[0])
        n = int(ts[-1]) - self.t0 + 1
        b, s, have = np.zeros(n), np.zeros(n), np.zeros(n, dtype=bool)
        b[ts - self.t0], s[ts - self.t0], have[ts - self.t0] = buy, sell, True
        self.cb = np.concatenate([[0.0], np.cumsum(b)])
        self.cs = np.concatenate([[0.0], np.cumsum(s)])
        self.ch = np.concatenate([[0], np.cumsum(have)])
        self.n = n

    @classmethod
    def from_db(cls, db: sqlite3.Connection) -> "Flow | None":
        try:
            rows = db.execute("SELECT ts, buy_usd, sell_usd FROM btc_flow_1s ORDER BY ts").fetchall()
        except sqlite3.OperationalError:
            return None
        if len(rows) < 3600:
            return None
        a = np.asarray(rows, dtype=np.float64)
        return cls(a[:, 0].astype(np.int64), a[:, 1], a[:, 2])

    def imb(self, t_end: np.ndarray, win: int) -> np.ndarray:
        """(buy - sell) / (buy + sell) over seconds [t_end - win, t_end); NaN where data is missing."""
        j = np.clip(np.asarray(t_end, dtype=np.int64) - self.t0, 0, self.n)
        i = np.clip(j - win, 0, self.n)
        b, s = self.cb[j] - self.cb[i], self.cs[j] - self.cs[i]
        cover = (self.ch[j] - self.ch[i]) / max(win, 1)      # seconds before the data start count as missing
        out = np.where(b + s > 0, (b - s) / np.maximum(b + s, 1e-9), 0.0)
        return np.where(cover > 0.9, out, np.nan)


@dataclass
class Cand:
    """All in-band ask prints of one side of one window, with features."""
    market: bt.Market
    outcome: str
    t: np.ndarray          # seconds from window start
    px: np.ndarray
    cost: np.ndarray       # entry (px + slip) grossed up for the fee share cut
    p: np.ndarray          # model probability of this side
    flow: np.ndarray       # flow towards this side (+ = market orders push price this side's way), NaN if no data


def fit_k(markets: list[bt.Market], btc: Btc, strikes: dict[int, float]) -> tuple[float, float]:
    """Volatility scale k by log-loss on UP prints (every 15 s) of the first half of the windows."""
    half = markets[: len(markets) // 2]
    ts_all, k_all, end_all, y_all = [], [], [], []
    for m in half:
        ev = m.ev["UP"]
        if not len(ev.ts):
            continue
        keep, last = [], -1e9
        for i, t in enumerate(ev.ts):
            if t - last >= 15 and 5 <= m.end_s - m.start_s - t:
                keep.append(i)
                last = t
        if not keep:
            continue
        tt = m.start_s + ev.ts[keep] - 1.0
        ts_all.append(tt)
        k_all.append(np.full(len(tt), strikes[m.market_id]))
        end_all.append(np.full(len(tt), m.end_s))
        y_all.append(np.full(len(tt), 1.0 if m.winner == "UP" else 0.0))
    t = np.concatenate(ts_all)
    kk = np.concatenate(k_all)
    ee = np.concatenate(end_all)
    y = np.concatenate(y_all)
    best = (1.0, float("inf"))
    for k in np.arange(0.6, 2.01, 0.1):
        tau = np.maximum(ee - t, 0.5)
        z = np.log(btc.price_at(t) / kk) / (k * btc.sigma(t) * np.sqrt(tau))
        from fair import ncdf
        p = np.clip(ncdf(z), 1e-4, 1 - 1e-4)
        ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        if ll < best[1]:
            best = (float(k), ll)
    return best


def candidates(markets: list[bt.Market], btc: Btc, flow: Flow | None, strikes: dict[int, float], fee: bt.FeeModel,
               k: float, a: argparse.Namespace) -> list[Cand]:
    out = []
    for m in markets:
        dur = m.end_s - m.start_s
        for o in bt.OUTS:
            ev = m.ev[o]
            tte = dur - ev.ts
            sel = ev.isA & (ev.px >= a.lo - 1e-9) & (ev.px <= a.hi + 1e-9) & (tte >= a.tte_min) & (tte <= a.tte_max)
            if not sel.any():
                continue
            t, px = ev.ts[sel], ev.px[sel]
            entry = np.minimum(0.99, np.round(px + a.slip, 3))
            cut = np.array([fee.share_cut(float(e)) for e in entry])
            cost = entry / (1 - cut)
            ta = m.start_s + t - a.lag
            pu = btc.p_up(ta, strikes[m.market_id], m.end_s, k)
            p = pu if o == "UP" else 1 - pu
            d = 1.0 if o == "UP" else -1.0
            fl = flow.imb(np.floor(ta), a.flow_win) * d if flow is not None else np.full(len(t), np.nan)
            out.append(Cand(m, o, t, px, cost, p, fl))
    return out


Rule = Callable[[Cand], np.ndarray]


def run_rule(name: str, cands: list[Cand], rule: Rule, fee: bt.FeeModel, slip: float, stake: float) -> list[bt.Trade]:
    best: dict[int, tuple[float, Cand, int]] = {}
    for c in cands:
        ok = rule(c)
        if not ok.any():
            continue
        i = int(ok.argmax())
        mid = c.market.market_id
        if mid not in best or c.t[i] < best[mid][0]:
            best[mid] = (float(c.t[i]), c, i)
    trades = []
    for mid, (t, c, i) in best.items():
        entry = min(0.99, round(float(c.px[i]) + slip, 3))
        sh = stake / entry
        win = c.market.winner == c.outcome
        pnl, f = bt.settle_buy(fee, True, entry, sh, 1.0 if win else 0.0, 0.0, stake)
        tr = bt.Trade(name, mid, c.market.day, c.outcome, t, entry, sh, 1.0 if win else 0.0,
                      "settle_win" if win else "settle_loss", float(c.market.end_s - c.market.start_s), f, pnl)
        tr.level = float(c.cost[i])          # effective cost incl. fee: the break-even win rate
        trades.append(tr)
    return trades


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


HDR = (f"{'правило':<46}{'сделок':>7}{'цена':>7}{'безуб.':>8}{'выигр':>8}  {'95% дов.':<11}{'ROI':>8}"
       f"{'1-я пол':>9}{'2-я пол':>9}{'дни+':>9}")


def row(name: str, trades: list[bt.Trade], markets: list[bt.Market]) -> str:
    n = len(trades)
    if n == 0:
        return f"{name:<46}{0:>7}"
    s = bt.stats(trades, markets)
    h = bt.halves(trades, markets)
    w = sum(t.how == "settle_win" for t in trades)
    lo, hi = wilson(w, n)
    avg = sum(t.entry for t in trades) / n
    be = sum(t.level for t in trades) / n
    ci = f"[{lo:.0%}-{hi:.0%}]"
    return (f"{name:<46}{n:>7}{avg:>7.3f}{be:>8.1%}{w / n:>8.1%}  {ci:<11}"
            f"{s['roi']:>+8.1%}{h[0]:>+9.1%}{h[1]:>+9.1%}{s['days_pos']:>5}/{s['days']:<3}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", default="5m", choices=("5m", "15m"))
    ap.add_argument("--db", default=str(HERE / "data" / "pf_hist.sqlite3"))
    ap.add_argument("--lo", type=float, default=0.10)
    ap.add_argument("--hi", type=float, default=0.30)
    ap.add_argument("--tte-min", type=float, default=30, help="не позже, чем за N с до конца")
    ap.add_argument("--tte-max", type=float, default=None, help="не раньше, чем за N с до конца (по умолч. окно−30)")
    ap.add_argument("--slip", type=float, default=0.01)
    ap.add_argument("--lag", type=float, default=1.0, help="модель и поток видят Binance на N с раньше сделки")
    ap.add_argument("--flow-win", type=int, default=30, help="окно потока маркет-ордеров, с")
    ap.add_argument("--stake", type=float, default=1.1)
    ap.add_argument("--k", type=float, default=None, help="масштаб волатильности (по умолч. подбор на 1-й половине)")
    ap.add_argument("--out", default=str(HERE / "out"))
    a = ap.parse_args()
    dur = 300 if a.series == "5m" else 900
    if a.tte_max is None:
        a.tte_max = dur - 30

    markets, fee = bt.load(Path(a.db), a.series)
    db = sqlite3.connect(a.db)
    btc = Btc.from_db(db)
    if btc is None:
        sys.exit("нет секундных цен BTC — сначала запусти btc1s.py")
    strikes = {mid: sp for mid, sp in db.execute(
        "SELECT market_id, start_price FROM markets WHERE series=? AND start_price > 0", (a.series,))}
    markets = [m for m in markets if m.market_id in strikes and btc.coverage(m.start_s - 900, m.end_s) > 0.97]
    if len(markets) < 50:
        sys.exit(f"мало окон с ценами BTC ({len(markets)}) — проверь collect.py и btc1s.py")
    flow = Flow.from_db(db)
    if a.k is None:
        k, ll = fit_k(markets, btc, strikes)
        k_note = f"k = {k:.1f} (подобран на 1-й половине окон, лог-лосс {ll:.4f})"
    else:
        k, k_note = a.k, f"k = {a.k:.2f} (задан вручную)"

    print(f"окон BTC {a.series}: {len(markets)}  ({markets[0].day} … {markets[-1].day})")
    print(f"комиссия: {fee.note}")
    print(f"модель: {k_note}")
    if flow is None:
        print("поток маркет-ордеров: нет данных (запусти btc_flow.py) — правила с потоком пропущены")
    else:
        cov = sum(1 for m in markets if not np.isnan(flow.imb(np.array([m.end_s]), a.flow_win))[0]) / len(markets)
        print(f"поток маркет-ордеров: есть для {cov:.0%} окон (окно {a.flow_win} с)")
    print(f"вход: первая покупка стороны по {a.lo:.2f}–{a.hi:.2f} за {a.tte_min:.0f}–{a.tte_max:.0f} с до конца, "
          f"цена + {a.slip:.2f}, ставка {a.stake}$, до расчёта, одна сделка на окно")
    cands = candidates(markets, btc, flow, strikes, fee, k, a)

    rules: list[tuple[str, Rule]] = [("любая сторона в диапазоне", lambda c: np.ones(len(c.t), dtype=bool))]
    for e in (0.0, 0.02, 0.05, 0.10):
        rules.append((f"модель: край ≥ {e:.2f}", lambda c, e=e: c.p - c.cost >= e))
    rules.append(("модель: край < 0 (контроль)", lambda c: c.p - c.cost < 0))
    if flow is not None:
        for th in (0.2, 0.4, 0.6):
            rules.append((f"поток {a.flow_win}с к стороне ≥ {th:.1f}", lambda c, th=th: np.nan_to_num(c.flow, nan=-9) >= th))
        rules.append((f"поток против стороны ≤ −0.4 (контроль)", lambda c: np.nan_to_num(c.flow, nan=9) <= -0.4))
        for e in (0.0, 0.02, 0.05):
            for th in (0.2, 0.4):
                rules.append((f"модель ≥ {e:.2f} + поток ≥ {th:.1f}",
                              lambda c, e=e, th=th: (c.p - c.cost >= e) & (np.nan_to_num(c.flow, nan=-9) >= th)))

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\nБезубыток = средняя цена с комиссией: «выигр» должен быть выше «безуб.», и в обеих половинах ROI > 0.")
    print("Вариантов много: лучший из них почти всегда выглядит неплохо случайно. Верь только тому, что держится")
    print("во 2-й половине и имеет доверительный интервал выше безубытка.\n")
    print(HDR)
    print("-" * len(HDR))
    all_trades: dict[str, list[bt.Trade]] = {}
    for name, rule in rules:
        tr = run_rule(name, cands, rule, fee, a.slip, a.stake)
        all_trades[name] = tr
        print(row(name, tr, markets))

    print("\nПо ценам (правило «любая сторона» и «модель: край ≥ 0.02»):")
    print(HDR)
    for name in ("любая сторона в диапазоне", "модель: край ≥ 0.02"):
        tr = all_trades[name]
        lo = a.lo
        while lo < a.hi - 1e-9:
            hi = round(lo + 0.05, 2)
            last = hi >= a.hi - 1e-9
            sub = [t for t in tr if lo - 1e-9 <= t.entry - a.slip < (a.hi + 1e-9 if last else hi - 1e-9)]
            print(row(f"  {name[:22]} {lo:.2f}-{hi:.2f}", sub, markets))
            lo = hi

    path = out_dir / f"dog_history_{a.series}.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["rule", "market_id", "day", "outcome", "t_entry", "entry", "cost_eff", "won", "pnl"])
        for name, tr in all_trades.items():
            for t in tr:
                w.writerow([name, t.market_id, t.day, t.outcome, round(t.t_entry, 2), t.entry, round(t.level, 4),
                            int(t.how == "settle_win"), round(t.pnl, 4)])
    print(f"\nсделки по всем правилам: {path}")
    print("Стакан (плотности, снятия, дисбаланс) здесь не проверяется — его истории нет. Для него: dom_recorder.py "
          "→ dom_replay.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
