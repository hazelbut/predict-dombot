"""Favourite bot: entry rule, risk caps, settlement, live retry — with a fake venue."""

from __future__ import annotations

import asyncio
import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from terminal.books import split_book  # noqa: E402
from terminal.config import parse_series  # noqa: E402
from terminal.favbot import BotCfg, Entry, FavBot, Ledger, risk_gate, settle_pnl, signal  # noqa: E402
from terminal.markets import MarketInfo, Outcome  # noqa: E402
from terminal.venue import Account, BookState, Position  # noqa: E402

S0 = 1_790_000_100 // 300 * 300


def market(mid: int = 7, start: int = S0, sp: float | None = 100.0, ep: float | None = None) -> MarketInfo:
    return MarketInfo("btc-updown-5m", "BTC 5M", f"btc-updown-5m-{start}", "t", 1, mid, start * 1000,
                      (start + 300) * 1000, sp, ep, "OPEN", "RESOLVED" if ep else "", 2, 200, False, False,
                      Outcome("Up", "tu", 1), Outcome("Down", "td", 2))


def book(up_bid: float, up_ask: float, size: float = 50) -> BookState:
    up, down = split_book([[up_bid, size]], [[up_ask, size]], 2)
    return BookState(up, down, 0, 0.0)


class FakeVenue:
    def __init__(self) -> None:
        self.markets: dict[int, MarketInfo] = {}
        self.books: dict[int, BookState] = {}
        self.account = Account(balance=50.0)
        self.s = SimpleNamespace(series=parse_series("btc-updown-5m:300"), predict_account="", private_key="")
        self.rest = SimpleNamespace(category=self._cat, positions=self._pos)
        self.orders: list[tuple] = []
        self.fail_first = False

    async def _cat(self, slug):
        return None

    async def _pos(self):
        return []

    async def buy(self, market_id, outcome, limit, usdt):
        self.orders.append((market_id, outcome, limit, usdt))
        if self.fail_first and len(self.orders) == 1:
            return {"ok": False, "msg": "busy"}
        return {"ok": True, "msg": "OK"}


def test_signal_window_band_and_spread():
    cfg = BotCfg()
    m = market()
    assert signal(cfg, m, book(0.63, 0.65), S0 + 60) == ("UP", 0.65)
    assert signal(cfg, m, book(0.33, 0.35), S0 + 60) == ("DOWN", 0.67)      # Down ask = 1 - Up bid
    assert signal(cfg, m, book(0.63, 0.65), S0 + 20) is None                # too early
    assert signal(cfg, m, book(0.63, 0.65), S0 + 160) is None               # too late
    assert signal(cfg, m, book(0.50, 0.52), S0 + 60) is None                # not in band
    assert signal(cfg, m, book(0.55, 0.65), S0 + 60) is None                # spread 10c > 5c


def test_risk_gate(tmp_path):
    cfg = BotCfg(live=True)
    led = Ledger(tmp_path / "s.json")
    stop = tmp_path / "STOP"
    assert risk_gate(cfg, led, S0, 50.0, stop) == ""
    for i in range(9):      # -9.9 $ today
        led.add(Entry(i, "s", "UP", S0, 0.65, 0.66, 1.69, 1.1, S0 + 300, settled=True, won=False, pnl=-1.1))
    assert "дневной" in risk_gate(cfg, led, S0, 50.0, stop)
    assert "общий" in risk_gate(BotCfg(live=True, total_loss=5), led, S0, 50.0, stop)
    assert risk_gate(cfg, Ledger(tmp_path / "x.json"), S0, 1.0, stop).startswith("мало USDT")
    stop.write_text("")
    assert "пауза" in risk_gate(cfg, Ledger(tmp_path / "y.json"), S0, 50.0, stop)


def test_checkpoint_stops_bad_edge(tmp_path):
    cfg = BotCfg(checkpoints=((10, -0.03),), daily_loss=1e9, total_loss=1e9)
    led = Ledger(tmp_path / "s.json")
    for i in range(10):
        won = i < 5
        led.add(Entry(i, "s", "UP", S0 - 86400 * i, 0.65, 0.66, 1.69, 1.1, S0, settled=True, won=won,
                      pnl=(1.69 - 1.1) if won else -1.1))
    assert "контрольной" in risk_gate(cfg, led, S0, 50.0, tmp_path / "STOP")


def test_settle_pnl_uses_fee_in_shares():
    cfg = BotCfg()
    e = Entry(1, "s", "UP", S0, 0.65, 0.66, 100 / 0.65 * 0.65 / 0.65, 1.0, S0)
    e.shares, e.cost = 10.0, 6.5
    cut = 0.02 * 0.35 / 0.65
    assert math.isclose(settle_pnl(cfg, e, True), 10 * (1 - cut) - 6.5)
    assert math.isclose(settle_pnl(cfg, e, False), -6.5)
    e.shares_seen = 9.8
    assert math.isclose(settle_pnl(cfg, e, True), 9.8 - 6.5)


def test_paper_flow_entry_once_then_settle(tmp_path):
    async def go():
        v = FakeVenue()
        m = market()
        v.markets[m.market_id] = m
        v.books[m.market_id] = book(0.63, 0.65)
        bot = FavBot(v, BotCfg(), tmp_path)
        await bot.tick(S0 + 40)
        await bot.tick(S0 + 41)          # second signal in the same window: ignored
        assert len(bot.led.entries) == 1 and not v.orders
        e = bot.led.entries[str(m.market_id)]
        assert e.outcome == "UP" and math.isclose(e.cost, 1.1, rel_tol=1e-6)
        await bot.settle(S0 + 320)       # no end price yet -> still open
        assert bot.led.open()
        v.markets[m.market_id] = market(ep=101.0)
        await bot.settle(S0 + 320)
        assert not bot.led.open() and e.won is True and e.pnl > 0.5
        assert (tmp_path / "logs" / "favbot_paper.csv").exists()
    asyncio.run(go())


def test_live_fill_confirmed_only_by_position(tmp_path):
    async def go():
        v = FakeVenue()
        v.fail_first = True
        m = market()
        v.markets[m.market_id] = m
        v.books[m.market_id] = book(0.33, 0.35)        # Down ask 0.67
        bot = FavBot(v, BotCfg(live=True), tmp_path)
        await bot.tick(S0 + 40)
        assert len(v.orders) == 1 and not bot.led.entries and not bot.pending     # rejected -> retry later
        await bot.tick(S0 + 41)
        assert len(v.orders) == 2 and v.orders[1][1:3] == ("DOWN", 0.68) and m.market_id in bot.pending
        await bot.tick(S0 + 42)                         # accepted but no position yet: still pending, no new order
        assert len(v.orders) == 2 and not bot.led.entries
        v.account.positions = [Position(m.market_id, "DOWN", "td", 1.6, 0.67)]
        await bot.tick(S0 + 43)
        e = bot.led.entries[str(m.market_id)]
        assert e.filled and e.shares_seen == 1.6 and not bot.pending
        v.account.positions = [Position(m.market_id, "DOWN", "td", 1.6, 0.67, resolved="LOST")]
        await bot.settle(S0 + 330)
        assert e.won is False and math.isclose(e.pnl, -e.cost)
        assert Ledger.load(tmp_path / "logs" / "favbot_live.json").entries[str(m.market_id)].settled
    asyncio.run(go())


def test_no_fill_event_allows_retry(tmp_path):
    async def go():
        v = FakeVenue()
        m = market()
        v.markets[m.market_id] = m
        v.books[m.market_id] = book(0.63, 0.65)
        bot = FavBot(v, BotCfg(live=True), tmp_path)
        await bot.tick(S0 + 40)
        assert m.market_id in bot.pending and len(v.orders) == 1
        await bot.tick(S0 + 45)                          # no event, no position: keep waiting, no new order
        assert len(v.orders) == 1
        bot.on_wallet({"type": "orderNotAccepted", "details": {"outcomeName": "Up"}})
        await bot.tick(S0 + 46)                          # definitive no-fill -> miss + immediate retry
        assert len(bot.led.misses) == 1 and len(v.orders) == 2
        bot.on_wallet({"type": "orderNotAccepted", "details": {"outcomeName": "Up"}})
        await bot.tick(S0 + 47)
        bot.on_wallet({"type": "orderNotAccepted", "details": {"outcomeName": "Up"}})
        await bot.tick(S0 + 48)
        assert len(v.orders) == 3 and len(bot.led.misses) == 3       # max 3 attempts
        v.markets[m.market_id] = market(ep=101.0)
        await bot.settle(S0 + 330)
        assert bot.led.stats()["n"] == 0
        assert bot.led.miss_stats() == {"n": 3, "settled": 3, "would_win": 3}
        assert "неисполнено 3" in bot.status_line()
    asyncio.run(go())


def test_unconfirmed_order_blocks_window_and_late_fills_are_booked(tmp_path):
    async def go():
        v = FakeVenue()
        m = market()
        v.markets[m.market_id] = m
        v.books[m.market_id] = book(0.63, 0.65)
        bot = FavBot(v, BotCfg(live=True), tmp_path)
        await bot.tick(S0 + 40)
        await bot.tick(S0 + 61)                          # 21 s, nothing known -> stop buying in this window
        assert len(v.orders) == 1 and len(bot.led.misses) == 1
        await bot.tick(S0 + 70)
        assert len(v.orders) == 1                        # blocked: no doubling
        unit_net = bot.win[m.market_id]["tpl"].shares * (1 - bot._cut(0.65))
        v.account.positions = [Position(m.market_id, "UP", "tu", unit_net, 0.65)]
        await bot.tick(S0 + 75)                          # the "miss" filled late -> booked, miss removed
        e = bot.led.entries[str(m.market_id)]
        assert not bot.led.misses and math.isclose(e.shares_seen, unit_net) and math.isclose(e.cost, 1.1, rel_tol=0.02)
        v.account.positions = [Position(m.market_id, "UP", "tu", 3 * unit_net, 0.65)]
        await bot.tick(S0 + 80)                          # two more late fills appear: cost follows the shares
        assert math.isclose(e.shares_seen, 3 * unit_net) and math.isclose(e.cost, 3.3, rel_tol=0.02)
        v.account.positions = [Position(m.market_id, "UP", "tu", 3 * unit_net, 0.65, resolved="WON")]
        await bot.settle(S0 + 330)
        assert e.won and math.isclose(e.pnl, 3 * unit_net - e.cost)
    asyncio.run(go())


def test_price_cap_never_above_max(tmp_path):
    async def go():
        for ask_bid, extra, want in (((0.63, 0.65), 3, 0.68), ((0.67, 0.69), 3, 0.70), ((0.68, 0.70), 3, 0.70)):
            v = FakeVenue()
            m = market()
            v.markets[m.market_id] = m
            v.books[m.market_id] = book(*ask_bid)
            bot = FavBot(v, BotCfg(live=True, extra_ticks=extra, max_price=0.70), tmp_path / str(want) / str(ask_bid))
            await bot.tick(S0 + 40)
            assert v.orders and v.orders[0][2] == want, (ask_bid, v.orders)
    asyncio.run(go())


def test_audit_positions_split():
    import run_favbot
    w = 10 ** 18
    raw = [{"amount": str(2 * w), "averageBuyPriceUsd": "0.65", "outcome": {"status": "WON"}},
           {"amount": str(3 * w), "averageBuyPriceUsd": "0.60", "outcome": {"status": "LOST"}},
           {"amount": str(1 * w), "averageBuyPriceUsd": "0.70", "outcome": {"status": None}},
           {"amount": "0", "outcome": {"status": "WON"}}]
    r = run_favbot.audit_positions(raw)
    assert r["won_n"] == 1 and r["won_usd"] == 2 and r["lost_n"] == 1 and r["open_n"] == 1
    assert math.isclose(r["open_cost"], 0.7)
