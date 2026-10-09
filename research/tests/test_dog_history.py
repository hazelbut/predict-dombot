"""dog_history must show no edge on a fairly priced market and find the edge when predict.fun lags Binance."""

from __future__ import annotations

import argparse
import math
import random
import sqlite3
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import backtest as bt  # noqa: E402
import btc1s  # noqa: E402
import btc_flow  # noqa: E402
import collect  # noqa: E402
import dog_history as dh  # noqa: E402
from fair import Btc  # noqa: E402

DUR = 300
T0 = 1_700_000_000


def ncdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def synth(path: Path, n: int, quote_lag: int, seed: int = 3) -> None:
    """BTC 1 s walk (2 bp/s); predict.fun quotes the fair P(up) computed from the BTC price `quote_lag` s ago."""
    rnd = random.Random(seed)
    sig = 0.0002
    total = 3600 + n * DUR + 5
    lp = [math.log(60_000.0)]
    for _ in range(total):
        lp.append(lp[-1] + rnd.gauss(0, sig))
    px = np.exp(np.asarray(lp))
    db = collect.connect(path)
    bdb = btc1s.connect(path)
    fdb = btc_flow.connect(path)
    bdb.executemany("INSERT INTO btc_1s VALUES (?,?)", [(T0 - 3600 + i, float(px[i])) for i in range(total)])
    bdb.commit()
    fdb.executemany("INSERT INTO btc_flow_1s VALUES (?,?,?)",
                    [(T0 - 3600 + i, 1000.0, 1000.0) for i in range(total)])
    fdb.commit()

    def p_at(sec: int) -> float:
        return float(px[sec - (T0 - 3600)])

    for k in range(n):
        s0 = T0 + k * DUR
        mid = 5000 + k
        K = p_at(s0)
        rows, t = [], 1
        while t < DUR - 1:
            sq = s0 + t - quote_lag
            fair = ncdf(math.log(p_at(sq) / K) / (sig * math.sqrt(max(1, DUR - t + quote_lag))))
            q = min(0.98, max(0.02, fair))
            up_buy = rnd.random() < 0.5
            pr = round(min(0.99, q + 0.01), 2) if up_buy else round(max(0.01, q - 0.01), 2)
            if rnd.random() < 0.5:
                rows.append((mid, (s0 + t) * 1000, "UP", "BUY" if up_buy else "SELL", pr, 10.0, 0.0, "", f"{k}-{t}"))
            else:
                rows.append((mid, (s0 + t) * 1000, "DOWN", "SELL" if up_buy else "BUY", round(1 - pr, 2), 10.0, 0.0,
                             "", f"{k}-{t}"))
            t += rnd.randint(1, 3)
        win = "UP" if p_at(s0 + DUR) > K else "DOWN"
        db.execute("INSERT INTO markets VALUES (?,?,?,?,?,?,?,?,?,?,?,1,0,0,NULL)",
                   (f"btc-updown-5m-{s0}", "5m", s0, s0 + DUR, mid, K, p_at(s0 + DUR), win, 200, "RESOLVED", len(rows)))
        db.executemany("INSERT OR IGNORE INTO trades VALUES (?,?,?,?,?,?,?,?,?)", rows)
    db.commit()
    bdb.commit()
    fdb.commit()


def run(path: Path) -> dict[str, dict]:
    a = argparse.Namespace(lo=0.10, hi=0.30, tte_min=30, tte_max=270, slip=0.01, lag=1.0, flow_win=30, stake=1.1)
    markets, fee = bt.load(path, "5m")
    db = sqlite3.connect(path)
    btc = Btc.from_db(db)
    flow = dh.Flow.from_db(db)
    assert btc is not None and flow is not None
    strikes = dict(db.execute("SELECT market_id, start_price FROM markets"))
    k, _ = dh.fit_k(markets, btc, strikes)
    cands = dh.candidates(markets, btc, flow, strikes, fee, k, a)
    out = {}
    for name, rule in (("any", lambda c: np.ones(len(c.t), dtype=bool)),
                       ("model05", lambda c: c.p - c.cost >= 0.05),
                       ("flow", lambda c: np.nan_to_num(c.flow, nan=-9) >= 0.2)):
        tr = dh.run_rule(name, cands, rule, fee, a.slip, a.stake)
        out[name] = {**bt.stats(tr, markets), "k": k}
    return out


def test_fair_market_no_edge(tmp_path):
    p = tmp_path / "h.sqlite3"
    synth(p, 1500, quote_lag=0, seed=11)               # 500 windows are too few: +-10% ROI is noise there
    r = run(p)
    assert r["any"]["n"] > 1000
    assert r["any"]["roi"] < 0.0, r["any"]             # spread + slip + fee
    assert r["model05"]["roi"] < 0.0, r["model05"]     # a 1 s stale model has nothing over a fair quote
    assert abs(r["any"]["k"] - 1.0) <= 0.2
    assert r["flow"]["n"] == 0                          # flow is balanced in the fixture: never >= 0.2


def test_lagging_market_found_by_model(tmp_path):
    p = tmp_path / "h.sqlite3"
    synth(p, 500, quote_lag=12)                        # predict.fun 12 s behind Binance
    r = run(p)
    assert r["model05"]["n"] >= 40
    assert r["model05"]["roi"] > 0.15, r["model05"]


def test_flow_imbalance_window():
    ts = np.arange(100, 200)
    buy = np.where(ts < 150, 300.0, 100.0)
    sell = np.full(len(ts), 100.0)
    f = dh.Flow(ts, buy, sell)
    assert abs(f.imb(np.array([150]), 10)[0] - 0.5) < 1e-9          # 300 vs 100
    assert abs(f.imb(np.array([180]), 10)[0]) < 1e-9
    assert math.isnan(f.imb(np.array([105]), 30)[0])               # not enough data before ts 100
