"""The backtest must not invent an edge on a fairly priced market, and must find one when it exists."""

from __future__ import annotations

import asyncio
import math
import random
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import backtest as bt  # noqa: E402
import collect  # noqa: E402

DUR = 900


def ncdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def synth_db(path: Path, n: int, overreact: float = 1.0, seed: int = 7) -> None:
    """BTC random walk; market quotes fair P(Up) (overreact>1 pushes quotes away from 0.5 = mispricing),
    1-cent spread, prints every ~3 s, half of them reported as the equivalent Down trade."""
    rnd = random.Random(seed)
    db = collect.connect(path)
    sig = 0.0015 / math.sqrt(DUR)       # 0.15 % per 15 min
    t0 = 1_700_000_000
    for k in range(n):
        s0 = t0 + k * DUR
        mid = 1000 + k
        x = 0.0
        rows = []
        t = 0.0
        while t < DUR:
            dt = rnd.uniform(1, 5)
            x += rnd.gauss(0, sig * math.sqrt(dt))
            t += dt
            if t >= DUR:
                break
            fair = ncdf(x / (sig * math.sqrt(max(1.0, DUR - t))))
            q = min(0.98, max(0.02, 0.5 + overreact * (fair - 0.5)))
            ask, bid = round(min(0.99, q + 0.01), 2), round(max(0.01, q - 0.01), 2)
            up_buy = rnd.random() < 0.5
            px = ask if up_buy else bid
            if rnd.random() < 0.5:
                rows.append((mid, int((s0 + t) * 1000), "UP", "BUY" if up_buy else "SELL", px, 10.0, 0.0, "", f"tx{k}-{t}"))
            else:   # same fill seen from the Down side
                rows.append((mid, int((s0 + t) * 1000), "DOWN", "SELL" if up_buy else "BUY", round(1 - px, 2), 10.0,
                             0.0, "", f"tx{k}-{t}"))
        win = "UP" if x >= 0 else "DOWN"
        db.execute("INSERT INTO markets VALUES (?,?,?,?,?,?,?,?,?,?,?,1,0,0,NULL)",
                   (f"btc-updown-15m-{s0}", "15m", s0, s0 + DUR, mid, 1.0, 1.0, win, 200, "RESOLVED", len(rows)))
        db.executemany("INSERT OR IGNORE INTO trades VALUES (?,?,?,?,?,?,?,?,?)", rows)
    db.commit()


def run_all(path: Path):
    markets, fee = bt.load(path, "15m")
    base = {"stake": 10.0, "slip": 0.01, "latency": 1.0}
    fades = [{**base, "name": "f", "x": x, "lo": 0.05, "target": tg, "t_from": 120, "t_to": 600}
             for x in (0.40, 0.35, 0.30, 0.25) for tg in (0.05, 0.10, 0.15, 0.0)]
    grid = {**base, "name": "g", "levels": [0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10], "target": 0.10,
            "t_from": 0, "t_to": 720, "stop_s": None}
    rf = [bt.stats([t for m in markets for t in bt.run_fade(m, c, fee)], markets) for c in fades]
    rg = bt.stats([t for m in markets for t in bt.run_grid(m, grid, fee)], markets)
    rg_touch = bt.stats([t for m in markets for t in bt.run_grid(m, {**grid, "touch": True}, fee)], markets)
    assert rg_touch["n"] >= rg["n"]
    rg["roi_touch"] = rg_touch["roi"]
    return markets, fee, rf, rg


def test_events_normalise_down_to_up_book():
    ev = bt.build_events([(1000, "DOWN", "BUY", 0.35), (2000, "UP", "BUY", 0.66)], 0)
    up, dn = ev["UP"], ev["DOWN"]
    assert (up.ts.tolist(), up.px.tolist(), up.isA.tolist()) == ([1.0, 2.0], [0.65, 0.66], [False, True])
    assert (dn.ts.tolist(), dn.px.tolist(), dn.isA.tolist()) == ([1.0, 2.0], [0.35, 0.34], [True, False])


def test_fair_market_shows_no_edge(tmp_path):
    p = tmp_path / "h.sqlite3"
    synth_db(p, 600, overreact=1.0)
    markets, fee, rf, rg = run_all(p)
    assert len(markets) == 600
    assert all(m.winner in ("UP", "DOWN") for m in markets)
    # no config may look profitable on a martingale: costs (spread, slip, fee) must make them all <= ~0
    best = max(s["roi"] for s in rf if s["n"] >= 100)
    assert best < 0.03, best
    assert rg["roi"] < 0.03 and rg["roi_touch"] < 0.03, rg


def test_mispriced_market_is_detected(tmp_path):
    p = tmp_path / "h.sqlite3"
    synth_db(p, 600, overreact=1.6)          # the cheap side is too cheap -> buying it must pay
    _, _, rf, _ = run_all(p)
    best = max(s["roi"] for s in rf if s["n"] >= 100)
    assert best > 0.05, best


def test_target_exit_needs_a_print_above():
    m = bt.Market(1, 0, 900, "DOWN", bt.build_events(
        [(10_000, "UP", "BUY", 0.35), (20_000, "UP", "BUY", 0.45), (30_000, "UP", "BUY", 0.46)], 0))
    fee = bt.FeeModel(0.0, True, "")
    c = {"name": "f", "x": 0.35, "lo": 0.05, "target": 0.10, "t_from": 0, "t_to": 600, "stake": 10, "slip": 0.0,
         "latency": 1.0}
    (t,) = bt.run_fade(m, c, fee)
    assert t.how == "target" and t.exit == 0.45 and t.t_exit == 30.0     # the print AT 0.45 is not enough
    c2 = {**c, "target": 0.12}
    (t2,) = bt.run_fade(m, c2, fee)
    assert t2.how == "settle_loss" and math.isclose(t2.pnl, -10.0)


def test_grid_fill_needs_a_print_below():
    m = bt.Market(1, 0, 900, "UP", bt.build_events([(5_000, "UP", "SELL", 0.40), (6_000, "UP", "SELL", 0.39)], 0))
    c = {"name": "g", "levels": [0.40], "target": 0.10, "t_from": 0, "t_to": 600, "stake": 4.0, "slip": 0.0,
         "latency": 1.0, "stop_s": None}
    (t,) = bt.run_grid(m, c, bt.FeeModel(0.0, True, ""))
    assert t.t_entry == 6.0 and t.how == "settle_win" and math.isclose(t.pnl, 10 * 0.6)


# ---------------------------------------------------------------------------- collector against a fake API
def test_collect_against_fake_api(tmp_path, monkeypatch):
    from aiohttp import web

    async def scenario():
        hits = {"cat": 0, "match": 0}

        async def category(req):
            hits["cat"] += 1
            assert req.headers.get("x-api-key") == "k"
            slug = req.match_info["slug"]
            start = int(slug.rsplit("-", 1)[1])
            if (start // 900) % 3 == 0:
                return web.json_response({"success": False}, status=404)
            return web.json_response({"success": True, "data": {"slug": slug, "markets": [{
                "id": start, "feeRateBps": 200, "status": "RESOLVED",
                "variantData": {"startPrice": "100000", "endPrice": "100010"},
                "resolution": {"name": "Up", "indexSet": 1},
                "outcomes": [{"name": "Up", "indexSet": 1, "status": "WON"}, {"name": "Down", "indexSet": 2}]}]}})

        async def matches(req):
            hits["match"] += 1
            mid = int(req.query["marketId"])
            page = int(req.query.get("after") or 0)
            items = [{"market": {"id": mid}, "amountFilled": str(5 * 10 ** 18), "priceExecuted": str(35 * 10 ** 16),
                      "executedAt": f"2026-01-01T00:00:{page * 2 + j:02d}Z", "transactionHash": f"0x{page}{j}",
                      "taker": {"quoteType": "Bid", "outcome": {"name": "Down", "indexSet": 2},
                                "fee": {"amount": str(35 * 10 ** 15), "type": "COLLATERAL"}}} for j in range(2)]
            return web.json_response({"success": True, "cursor": str(page + 1) if page < 2 else None, "data": items})

        app = web.Application()
        app.router.add_get("/v1/categories/{slug}", category)
        app.router.add_get("/v1/orders/matches", matches)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        db = tmp_path / "c.sqlite3"
        monkeypatch.setenv("PREDICT_API_KEY", "k")
        monkeypatch.setattr(sys, "argv", ["collect.py", "--days", "0.25", "--rpm", "100000", "--db", str(db),
                                          "--api", f"http://127.0.0.1:{port}"])
        rc = await collect.main()
        # second run: nothing left to fetch
        before = dict(hits)
        rc2 = await collect.main()
        after2 = dict(hits)
        monkeypatch.setattr(sys, "argv", ["collect.py", "--probe", "--rpm", "100000", "--db", str(db),
                                          "--api", f"http://127.0.0.1:{port}"])
        assert await collect.main() in (0, 1)        # 1 = the newest window is a 404 one in the fake
        await runner.cleanup()
        return rc, rc2, db, after2, before

    rc, rc2, db, hits, before = asyncio.run(scenario())
    assert rc == 0 and rc2 == 0
    c = sqlite3.connect(db)
    n_ok, n_miss = c.execute("SELECT SUM(complete), SUM(missing) FROM markets").fetchone()
    assert n_ok > 0 and n_miss > 0
    assert hits == before or hits["match"] == before["match"]          # resumable: no re-download
    rows = c.execute("SELECT outcome, side, price, shares, fee FROM trades LIMIT 1").fetchall()
    assert rows == [("DOWN", "BUY", 0.35, 5.0, 0.035)]
    assert c.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == n_ok * 6   # 3 pages x 2


def test_fee_taken_in_shares_on_buy():
    fee = bt.FeeModel(0.02, True, "")
    m = bt.Market(1, 0, 900, "UP", bt.build_events([(10_000, "UP", "BUY", 0.35)], 0))
    c = {"name": "f", "x": 0.35, "lo": 0.05, "target": 0.0, "t_from": 0, "t_to": 600, "stake": 35.0, "slip": 0.0,
         "latency": 1.0}
    (t,) = bt.run_fade(m, c, fee)
    # 100 shares bought, 2 % of them withheld (0.02*min(.35,.65)/.35) -> 98 pay out $1
    assert math.isclose(t.pnl, 98 - 35) and math.isclose(t.fee, 2.0)
    assert math.isclose(fee.share_cut(0.99), 0.02 * 0.01 / 0.99)


def test_fee_model_measured_from_fills(tmp_path):
    db = collect.connect(tmp_path / "f.sqlite3")
    rows = [(1, i, "UP", "BUY", 0.35, 100.0, 2.0, "SHARES", f"t{i}") for i in range(20)]
    rows += [(1, 100 + i, "DOWN", "BUY", 0.99, 10.0, 0.02 * 0.01 / 0.99 * 10, "SHARES", f"u{i}") for i in range(20)]
    rows += [(1, 200 + i, "UP", "SELL", 0.80, 10.0, 0.02 * 0.2 * 10, "COLLATERAL", f"v{i}") for i in range(20)]
    db.executemany("INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?)", rows)
    fm = bt.FeeModel.from_db(db, [200])
    assert math.isclose(fm.rate, 0.02, rel_tol=1e-6) and fm.buy_in_shares


def test_favdip_needs_arming_before_fill():
    fee = bt.FeeModel(0.02, True, "")
    # side trades at .66 (sell print) BEFORE it ever traded at .70+: must not fill; later .72 buy arms, .64 sell fills
    m = bt.Market(1, 0, 900, "UP", bt.build_events(
        [(5_000, "UP", "SELL", 0.60), (10_000, "UP", "BUY", 0.72), (20_000, "UP", "SELL", 0.64)], 0))
    c = {"name": "d", "levels": [0.65], "arm": 0.06, "t_from": 0, "t_to": 600, "stake": 6.5, "slip": 0.0,
         "latency": 1.0, "touch": False}
    (t,) = bt.run_favdip(m, c, fee)
    assert t.t_entry == 20.0 and t.entry == 0.65 and t.how == "settle_win"
    assert math.isclose(t.pnl, 10 * 0.35 + 0.25 * 0.02 * 0.35 * 10)
