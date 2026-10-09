"""DOM bot: order-book signals, fair model, entry rule, candidate log and report."""

from __future__ import annotations

import asyncio
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from terminal.btc import BtcState  # noqa: E402
from terminal.depth import SpotDepthBook  # noqa: E402
from terminal.dombot import (CandidateLog, DomBot, DomCfg, FairModel, FlowSignals, SigCfg, Sig,  # noqa: E402
                             evaluate, fee_cut, report)
from tests.test_favbot import FakeVenue, book, market, S0  # noqa: E402


def depth_book(mid: float, bid_wall: float = 0.0, ask_wall: float = 0.0, bid_q: float = 0.5,
               ask_q: float = 0.5) -> SpotDepthBook:
    bk = SpotDepthBook()
    bk.bids = {round(mid - 0.5 - i, 2): bid_q for i in range(200)}
    bk.asks = {round(mid + 0.5 + i, 2): ask_q for i in range(200)}
    if bid_wall:
        bk.bids[round(mid - 60.5, 2)] = bid_wall
    if ask_wall:
        bk.asks[round(mid + 60.5, 2)] = ask_wall
    bk.synced = True
    return bk


def test_wall_eaten_gives_down_signal():
    bk = depth_book(100_000.0, bid_wall=30.0)            # 30 BTC ~ $3M bid wall 6 bp below
    fs = FlowSignals(SigCfg(), bk)
    assert fs.tick(1000.0) == []
    assert fs.walls, "wall must be detected"
    # market sells eat the wall within 2 s
    for i in range(20):
        fs.on_trade(int((1000.5 + i * 0.05) * 1000), 99_939.5, 1.0, True)
    bk.bids[99_939.5] = 5.0
    sigs = fs.tick(1002.0)
    assert [(s.kind, s.dir) for s in sigs] == [("eaten", -1)]


def test_wall_pulled_gives_signal_against_wall():
    bk = depth_book(100_000.0, ask_wall=30.0)
    fs = FlowSignals(SigCfg(), bk)
    fs.tick(1000.0)
    del bk.asks[100_060.5]                               # ask wall cancelled, no trades -> fake resistance -> up
    sigs = fs.tick(1001.0)
    assert [(s.kind, s.dir) for s in sigs] == [("pulled", 1)]


def test_wall_left_behind_is_not_a_pull():
    bk = depth_book(100_000.0, ask_wall=30.0)
    fs = FlowSignals(SigCfg(), bk)
    fs.tick(1000.0)
    moved = depth_book(99_500.0)                         # price fell $500, the wall is out of range
    bk.bids, bk.asks = moved.bids, moved.asks
    assert fs.tick(1001.0) == []


def test_imbalance_needs_to_hold():
    bk = depth_book(100_000.0, bid_q=5.0, ask_q=0.5)
    fs = FlowSignals(SigCfg(), bk)
    assert fs.tick(1000.0) == []
    assert fs.imb is not None and fs.imb > 0.6
    assert fs.tick(1002.0) == []
    sigs = fs.tick(1003.5)
    assert [(s.kind, s.dir) for s in sigs] == [("imbalance", 1)]
    assert fs.tick(1010.0) == []                         # cooldown


def test_fair_model_sigma_and_probability():
    btc = BtcState(65)
    px = 100_000.0
    for i in range(3600):
        px *= math.exp(0.0001 * (1 if i % 2 else -1))   # 1 bp per second
        btc.hist.append((10_000 + i, px))
    f = FairModel(btc)
    sig = f.sigma(10_000 + 3600)
    assert abs(sig - 1e-4) < 1e-6
    assert abs(f.p_up(100_000, 100_000, 100, sig) - 0.5) < 1e-9
    # 10 bp above start with 100 s left and 1 bp/s vol -> z = 1
    assert abs(f.p_up(100_000 * math.exp(0.001), 100_000, 100, sig) - 0.8413) < 1e-3


def _fair_flat() -> FairModel:
    btc = BtcState(65)
    for i in range(1000):
        btc.hist.append((i, 100_000.0 * math.exp(0.0001 * (i % 2))))
    return FairModel(btc)


def test_evaluate_rule():
    cfg = DomCfg()
    m = market()                                   # start price 100, 5m window
    f = _fair_flat()
    sigma = 1e-4
    now = S0 + 200                                 # 100 s left
    b = book(0.18, 0.20)                           # UP ask 0.20, DOWN ask 0.82
    up_sig = [Sig(now - 5, "eaten", 1, 0, 0)]
    # S at start: model 0.5 for UP, ask 0.20 -> big edge; signal towards UP -> take
    c = evaluate(cfg, m, b, now, 100.0, 100.0, sigma, up_sig, f)
    assert [x.outcome for x in c] == ["UP"] and c[0].take and c[0].aligned == "eaten"
    assert abs(c[0].cost_eff - 0.20 / (1 - fee_cut(0.02, 0.20))) < 1e-12
    # no signal -> recorded but not taken
    c = evaluate(cfg, m, b, now, 100.0, 100.0, sigma, [], f)
    assert not c[0].take and "сигнал" in c[0].reason
    # stale signal
    c = evaluate(cfg, m, b, now, 100.0, 100.0, sigma, [Sig(now - 30, "eaten", 1, 0, 0)], f)
    assert not c[0].take
    # opposite signal blocks
    c = evaluate(cfg, m, b, now, 100.0, 100.0, sigma, up_sig + [Sig(now - 1, "pulled", -1, 0, 0)], f)
    assert not c[0].take and "против" in c[0].reason
    # price far below start: model says UP is cheap for a reason -> no edge
    c = evaluate(cfg, m, b, now, 100.0 * math.exp(-0.003), 100.0, sigma, up_sig, f)
    assert not c[0].take and "край" in c[0].reason
    # too close to the end / too early
    assert evaluate(cfg, m, b, S0 + 280, 100.0, 100.0, sigma, up_sig, f) == []
    assert evaluate(cfg, m, b, S0 + 10, 100.0, 100.0, sigma, up_sig, f) == []
    # outside the band
    assert evaluate(cfg, m, book(0.38, 0.40), now, 100.0, 100.0, sigma, up_sig, f) == []
    # DOWN side in band (UP bid 0.85 -> DOWN ask 0.15) needs a DOWN signal
    c = evaluate(cfg, m, book(0.85, 0.87), now, 100.0, 100.0, sigma, [Sig(now - 2, "imbalance", -1, 0, 0)], f)
    assert [(x.outcome, x.take) for x in c] == [("DOWN", True)]
    # no signal required
    c = evaluate(DomCfg(require_signal=False), m, b, now, 100.0, 100.0, sigma, [], f)
    assert c[0].take


def test_shift_raises_probability_only_with_signal():
    m = market()
    f = _fair_flat()
    now = S0 + 200
    b = book(0.18, 0.20)
    s = 100.0 * math.exp(-0.0005)
    base = evaluate(DomCfg(), m, b, now, s, 100.0, 1e-4, [Sig(now, "eaten", 1, 0, 0)], f)[0]
    sh = evaluate(DomCfg(shift_bps=3), m, b, now, s, 100.0, 1e-4, [Sig(now, "eaten", 1, 0, 0)], f)[0]
    assert sh.p_adj > base.p_adj and sh.p_model == base.p_model


def test_candidate_log_and_report(tmp_path):
    cl = CandidateLog(tmp_path / "logs" / "dombot_candidates.csv", 15)
    m = market()
    from terminal.dombot import Cand
    c = Cand("UP", 0.20, 0.18, 100, 0.5, 0.5, 0.204, 0.296, "eaten", "", True, "ok")
    cl.observe(m, c, S0 + 200, {"imb10": 0.1, "flow30": 0.2, "dist_bp": 0, "sigma_bp": 1})
    cl.observe(m, c, S0 + 205, {})                       # too soon, same signal -> skipped
    cl.observe(m, c, S0 + 216, {})                       # resample
    nt = Cand("DOWN", 0.20, 0.18, 100, 0.5, 0.5, 0.204, 0.296, "", "", False, "x")
    cl.observe(m, nt, S0 + 200, {})
    cl.observe(m, Cand("DOWN", 0.20, 0.18, 100, 0.5, 0.5, 0.204, 0.296, "eaten", "", True, "ok"), S0 + 201, {})
    cl.update_bids(m.market_id, book(0.45, 0.47))
    assert cl.due(S0 + 300) == [] and cl.due(S0 + 316) == [m]
    cl.resolve(m.market_id, True)
    rows = list(csv.DictReader((tmp_path / "logs" / "dombot_candidates.csv").open(encoding="utf-8")))
    assert len(rows) == 4 and rows[0]["won"] == "1" and float(rows[0]["max_bid_after"]) == 0.45
    assert [r["why_row"] for r in rows if r["outcome"] == "DOWN"] == ["first", "take"]
    assert rows[-1]["won"] == "0"
    txt = report(tmp_path)
    assert "все кандидаты" in txt and "правило бота" in txt


def test_bot_paper_entry_and_candidates(tmp_path):
    v = FakeVenue()
    m = market()
    v.markets[m.market_id] = m
    v.books[m.market_id] = book(0.18, 0.20)
    btc = BtcState(65)
    now = S0 + 200
    for i in range(1000):
        btc.hist.append((int(now) - 1000 + i, 100.0 * math.exp(0.0001 * (i % 2))))
    btc.mid, btc.ts = 100.0, int(now * 1000)
    depth = depth_book(100_000.0)
    depth.last_event = now
    bot = DomBot(v, DomCfg(), tmp_path, btc, depth)
    bot.flow.recent.append(Sig(now - 3, "eaten", 1, 0, 0))
    asyncio.run(bot.tick(now))
    assert bot.led.has(m.market_id)
    e = bot.led.entries[str(m.market_id)]
    assert e.outcome == "UP" and e.limit == 0.21 and e.cost <= 1.1 + 1e-9
    assert bot.cands.rows[m.market_id][0]["take"] == 1
    # second tick: one entry per window
    asyncio.run(bot.tick(now + 1))
    assert len(bot.led.entries) == 1
    # settle: UP won
    m.end_price = 101.0
    asyncio.run(bot.settle(S0 + 320))
    e = bot.led.entries[str(m.market_id)]
    assert e.settled and e.won and e.pnl > 0
    assert (tmp_path / "logs" / "dombot_candidates.csv").exists()


def test_bot_waits_for_feeds(tmp_path):
    v = FakeVenue()
    m = market()
    v.markets[m.market_id] = m
    v.books[m.market_id] = book(0.18, 0.20)
    btc = BtcState(65)
    bot = DomBot(v, DomCfg(), tmp_path, btc, SpotDepthBook())
    asyncio.run(bot.tick(S0 + 200))
    assert not bot.led.entries
