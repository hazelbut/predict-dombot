"""Recorder -> files -> replay: the replay must rebuild the books and take exactly the trades the rule implies."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

import dom_recorder as drec  # noqa: E402
import dom_replay as drp  # noqa: E402
from terminal.books import split_book  # noqa: E402
from terminal.btc import BtcState  # noqa: E402
from terminal.depth import SpotDepthBook  # noqa: E402
from terminal.markets import MarketInfo, Outcome  # noqa: E402
from terminal.venue import BookState  # noqa: E402

T0 = 1_800_000_000          # window start (multiple of 300)


def depth_at(mid: float, heavy_bids: bool) -> tuple[dict, dict]:
    bq = 8.0 if heavy_bids else 0.5
    bids = {round(mid - 0.5 - i, 2): bq for i in range(300)}
    asks = {round(mid + 0.5 + i, 2): 0.5 for i in range(300)}
    return bids, asks


def record(out: Path, end_price: float) -> None:
    btc, depth = BtcState(5), SpotDepthBook()
    m = MarketInfo("btc-updown-5m", "BTC 5M", f"btc-updown-5m-{T0}", "t", 1, 77, T0 * 1000, (T0 + 300) * 1000,
                   100_000.0, None, "OPEN", "", 2, 200, False, False, Outcome("Up", "u", 1), Outcome("Down", "d", 2))
    venue = SimpleNamespace(markets={77: m}, books={})
    rec = drec.Recorder(out, btc, depth, venue)
    t = T0 - 1000.0
    while t <= T0 + 320:
        mid = 100_000.0 + (1.0 if int(t) % 2 else 0.0)        # ~1e-5 per second volatility
        btc.on_tick(int(t * 1000), mid, mid - 0.01, mid + 0.01)
        depth.bids, depth.asks = depth_at(mid, heavy_bids=T0 + 140 <= t < T0 + 200)
        depth.synced, depth.last_event = True, t
        if T0 <= t < T0 + 300:
            ub, ua = ((0.18, 0.20) if t >= T0 + 100 else (0.48, 0.52))
            up, down = split_book([[ub, 100]], [[ua, 100]], 2)
            cur = venue.books.get(77)
            if cur is None or cur.up.bids[0][0] != ub:
                venue.books[77] = BookState(up, down, int(t * 1000), t)
        if int(t * 4) % 4 == 0:
            rec.on_trade(int(t * 1000), mid - 0.5, 0.01, True)
        if t >= T0 + 305 and m.end_price is None:
            m.end_price, m.trading_status = end_price, "CLOSED"
        rec.step(t)
        t += 0.25
    rec.close()


def args(**kw) -> argparse.Namespace:
    a = dict(series="5m", lo=0.10, hi=0.30, tte_min=30, tte_max=None, k=1.0, stake=1.1, latency=0.25, step=0.25,
             walls=[150_000.0], kinds=["все", "съели"], ttls=[20.0], edges=[0.02], shifts=[0.0])
    a.update(kw)
    return argparse.Namespace(**a)


def replay(tmp: Path, end_price: float) -> drp.Replay:
    d = tmp / "dom"
    record(d, end_price)
    rp = drp.Replay(args(), tmp / "out")
    rp.run(drp.records(drp.pick_files(d, None, None)))
    rp.settle()
    return rp


def by_name(rp: drp.Replay) -> dict[str, drp.VarState]:
    return {v.name(rp.walls): rp.vs[v] for v in rp.vars}


def test_recorder_writes_compact_diffs(tmp_path):
    record(tmp_path / "dom", 100_010.0)
    files = sorted((tmp_path / "dom").glob("*.jsonl.gz"))
    assert files
    kinds = [json.loads(x)["k"] for f in files for x in gzip.open(f, "rt")]
    assert kinds.count("D") >= 20 and kinds.count("d") > 100 and "M" in kinds and "p" in kinds and "x" in kinds
    assert kinds.count("p") <= 6                           # predict book only on change (+ per new file)


def test_replay_takes_signal_trade_and_settles(tmp_path):
    rp = replay(tmp_path, 100_010.0)                        # UP wins
    vs = by_name(rp)
    sig_all = vs["плотн ≥150k | все ≤20с | край ≥0.02"]
    eaten = vs["плотн ≥150k | съели ≤20с | край ≥0.02"]
    ctrl = vs["плотн ≥150k | БЕЗ сигнала (контроль) | край ≥0.02"]
    assert list(sig_all.pos) == [77]
    p = sig_all.pos[77]
    assert p.outcome == "UP" and T0 + 143 <= p.t <= T0 + 165 and abs(p.usdt - 1.1) < 1e-6
    assert p.won is True and p.pnl > 3                      # 1.1$ at 0.20 -> ~5.4 shares
    assert not eaten.pos                                    # there was no eaten wall
    assert list(ctrl.pos) == [77] and ctrl.pos[77].t < p.t  # control enters as soon as the price is in band
    assert rp.depth.synced and 99_699 < min(rp.depth.bids) < max(rp.depth.bids) < 100_001


def test_replay_loss_and_candidates(tmp_path):
    rp = replay(tmp_path, 99_990.0)                         # DOWN wins
    p = by_name(rp)["плотн ≥150k | все ≤20с | край ≥0.02"].pos[77]
    assert p.won is False and abs(p.pnl + 1.1) < 1e-6
    cand = (tmp_path / "out" / "logs" / "dombot_candidates.csv").read_text(encoding="utf-8").splitlines()
    assert len(cand) > 3 and all(line.endswith(",0") for line in cand[1:] if ",UP," in line)
