from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import pytest
from eth_account import Account
from eth_account.messages import encode_defunct, encode_typed_data
from predict_sdk.constants import ADDRESSES_BY_CHAIN_ID, ChainId

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terminal.books import split_book, take_buy, take_sell  # noqa: E402
from terminal.config import Settings, parse_series  # noqa: E402
from terminal.markets import MarketInfo, Outcome, parse_category, slug_for, window_start  # noqa: E402
from terminal.trading import OrderFactory, OrderRejected  # noqa: E402

ADDR = ADDRESSES_BY_CHAIN_ID[ChainId.BNB_MAINNET]


def market(yield_bearing: bool = True, neg_risk: bool = False) -> MarketInfo:
    now = int(time.time() * 1000)
    return MarketInfo("btc-updown-5m", "BTC 5M", "btc-updown-5m-1", "t", 1, 2032581, now, now + 300_000, 60000.0,
                      None, "OPEN", "REGISTERED", 2, 200, neg_risk, yield_bearing,
                      Outcome("Up", "1046133410909118936158999", 1), Outcome("Down", "2222", 2))


# ---------------------------------------------------------------- books
def test_split_book_complements_and_swaps_sides() -> None:
    up, down = split_book([[0.55, 100], [0.54, 50]], [[0.57, 40], [0.58, 100]], 2)
    assert up.best_bid == 0.55 and up.best_ask == 0.57
    assert down.asks == [(0.45, 100.0), (0.46, 50.0)]      # from Up bids
    assert down.bids == [(0.43, 40.0), (0.42, 100.0)]      # from Up asks


def test_take_buy_respects_limit_and_budget() -> None:
    asks = [(0.57, 40.0), (0.58, 100.0), (0.60, 300.0)]
    f = take_buy(asks, 0.58, budget_usdt=1000)
    assert f.shares == pytest.approx(140) and f.worst == 0.58           # stops before 0.60
    f2 = take_buy(asks, 0.60, budget_usdt=30)
    assert f2.usdt == pytest.approx(30) and f2.worst == 0.58
    assert f2.shares == pytest.approx(40 + (30 - 22.8) / 0.58)


def test_take_sell_respects_floor_and_size() -> None:
    bids = [(0.43, 40.0), (0.42, 100.0), (0.40, 300.0)]
    f = take_sell(bids, 0.42, 1000)
    assert f.shares == 140 and f.worst == 0.42
    assert take_sell(bids, 0.42, 10).shares == 10


# ---------------------------------------------------------------- signing
def recover(order: dict, m: MarketInfo) -> str:
    domain = {"name": "predict.fun CTF Exchange", "version": "1", "chainId": 56,
              "verifyingContract": ADDR.YIELD_BEARING_CTF_EXCHANGE if m.is_yield_bearing else ADDR.CTF_EXCHANGE}
    types = {"Order": [{"name": n, "type": t} for n, t in (
        ("salt", "uint256"), ("maker", "address"), ("signer", "address"), ("taker", "address"),
        ("tokenId", "uint256"), ("makerAmount", "uint256"), ("takerAmount", "uint256"), ("expiration", "uint256"),
        ("nonce", "uint256"), ("feeRateBps", "uint256"), ("side", "uint8"), ("signatureType", "uint8"))]}
    msg = {k: order[k] for k in ("salt", "maker", "signer", "taker", "tokenId", "makerAmount", "takerAmount",
                                 "expiration", "nonce", "feeRateBps", "side", "signatureType")}
    for k in ("salt", "tokenId", "makerAmount", "takerAmount", "expiration", "nonce", "feeRateBps"):
        msg[k] = int(msg[k])
    signable = encode_typed_data(domain_data=domain, message_types=types, message_data=msg)
    return Account.recover_message(signable, signature=order["signature"])


@pytest.mark.parametrize("yb", [True, False])
def test_eoa_market_buy_is_signed_by_wallet_and_capped_at_click_price(yb: bool) -> None:
    acct = Account.create()
    f = OrderFactory(acct.key.hex())
    m = market(yield_bearing=yb)
    up, _ = split_book([[0.55, 100]], [[0.57, 40], [0.58, 100], [0.60, 300]], 2)
    b = f.market_buy(m, "UP", up, 0.58, 30)
    o = b.data["order"]
    assert b.data["strategy"] == "MARKET" and o["side"] == 0 and o["tokenId"] == m.up.token_id
    assert o["maker"] == o["signer"] == acct.address and o["signatureType"] == 0
    assert o["signature"].startswith("0x") and o["hash"].startswith("0x") and len(o["hash"]) == 66
    assert recover(o, m) == acct.address
    worst = int(o["makerAmount"]) / int(o["takerAmount"])
    assert worst <= 0.58 + 1e-9                     # never pays above the clicked price
    assert int(o["makerAmount"]) / 1e18 <= 30 * 0.58 / 0.5724 + 0.01


def test_market_sell_floor_and_down_token() -> None:
    f = OrderFactory(Account.create().key.hex())
    m = market()
    _, down = split_book([[0.55, 100]], [[0.57, 40], [0.58, 100]], 2)   # down bids 0.43 / 0.42
    b = f.market_sell(m, "DOWN", down, 0.42, 120)
    o = b.data["order"]
    assert o["side"] == 1 and o["tokenId"] == m.down.token_id
    min_px = int(o["takerAmount"]) / int(o["makerAmount"])
    assert 0.42 - 1e-6 <= min_px <= 0.43
    assert b.shares == pytest.approx(120)


def test_limit_order_expires_at_market_end_and_post_only() -> None:
    f = OrderFactory(Account.create().key.hex())
    m = market()
    b = f.limit(m, "UP", "BUY", 0.5, 10, post_only=True)
    o = b.data["order"]
    assert b.data["strategy"] == "LIMIT" and b.data["isPostOnly"] is True
    assert o["expiration"] == m.end_ms // 1000
    assert int(o["makerAmount"]) == 5 * 10**18 and int(o["takerAmount"]) == 10 * 10**18
    assert b.data["pricePerShare"] == str(5 * 10**17)


def test_predict_account_orders_use_deposit_address_and_kernel_signature() -> None:
    acct = Account.create()
    deposit = "0x" + "ab" * 20
    f = OrderFactory(acct.key.hex(), deposit)
    assert f.address == deposit
    m = market()
    up, _ = split_book([[0.55, 100]], [[0.57, 40]], 2)
    o = f.market_buy(m, "UP", up, 0.57, 10).data["order"]
    assert o["maker"].lower() == o["signer"].lower() == deposit
    assert o["signature"].lower().startswith("0x01" + ADDR.ECDSA_VALIDATOR[2:].lower())
    sig = f.sign_auth_message("hello")
    assert sig.startswith("0x01") and len(sig) == 2 + 2 + 40 + 130


def test_eoa_auth_signature_recovers() -> None:
    acct = Account.create()
    f = OrderFactory(acct.key.hex())
    sig = f.sign_auth_message("Sign in to predict.fun")
    assert Account.recover_message(encode_defunct(text="Sign in to predict.fun"), signature=sig) == acct.address


def test_rejects_empty_book_and_dust() -> None:
    f = OrderFactory(Account.create().key.hex())
    m = market()
    up, _ = split_book([[0.55, 100]], [[0.57, 1]], 2)
    with pytest.raises(OrderRejected):
        f.market_buy(m, "UP", up, 0.56, 10)          # nothing at or below 0.56
    with pytest.raises(OrderRejected):
        f.market_buy(m, "UP", up, 0.57, 10)          # only 0.57$ available < 0.9$ minimum


# ---------------------------------------------------------------- markets / config
RAW = {
    "id": 405022, "slug": "btc-updown-5m-1789017900", "title": "Bitcoin Up or Down", "startsAt": "2026-09-10T05:25:00.000Z",
    "endsAt": "2026-09-10T05:30:00.000Z", "isNegRisk": False, "isYieldBearing": True,
    "markets": [{"id": 2032581, "decimalPrecision": 2, "feeRateBps": 200, "isNegRisk": False, "isYieldBearing": True,
                 "tradingStatus": "OPEN", "status": "REGISTERED",
                 "outcomes": [{"indexSet": 2, "name": "Down", "onChainId": "222"},
                              {"indexSet": 1, "name": "Up", "onChainId": "111"}],
                 "variantData": {"startPrice": "112345.67", "endPrice": None, "type": "CRYPTO_UP_DOWN"}}],
}


def test_parse_category() -> None:
    series = parse_series("btc-updown-5m:300")[0]
    m = parse_category(RAW, series)
    assert m is not None
    assert m.market_id == 2032581 and m.up.token_id == "111" and m.down.token_id == "222"
    assert m.start_price == 112345.67 and m.end_price is None and m.is_yield_bearing
    assert m.end_ms - m.start_ms == 300_000


def test_slug_window() -> None:
    s = parse_series("btc-updown-5m:300,btc-updown-15m:900")
    assert slug_for(s[0], window_start(s[0], 1789017999.0)) == "btc-updown-5m-1789017900"
    assert window_start(s[1], 1789017999.0) % 900 == 0


# ---------------------------------------------------------------- server guards (demo venue)
def test_server_guards_armed_and_size_cap(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    from terminal.server import Terminal

    async def run() -> None:
        term = Terminal(Settings(demo=True, max_order_usdt=20))
        await term.venue.start()
        term.btc.on_tick(int(time.time() * 1000), 60000.0, 59999.99, 60000.01)
        await asyncio.sleep(0.6)
        mid = next(iter(term.venue.markets))
        r = await term.handle_cmd({"cmd": "buy", "market_id": mid, "outcome": "UP", "price": 0.99, "usdt": 5})
        assert not r["ok"] and "выключена" in r["msg"]
        await term.handle_cmd({"cmd": "arm", "on": True})
        r = await term.handle_cmd({"cmd": "buy", "market_id": mid, "outcome": "UP", "price": 0.99, "usdt": 25})
        assert not r["ok"] and "лимит" in r["msg"]
        r = await term.handle_cmd({"cmd": "buy", "market_id": mid, "outcome": "UP", "price": 0.99, "usdt": 5})
        assert r["ok"], r
        r = await term.handle_cmd({"cmd": "sell", "market_id": mid, "outcome": "DOWN", "price": 0.01, "shares": 5})
        assert not r["ok"] and "нет позиции" in r["msg"]
        await term.venue.stop()

    asyncio.run(run())


def test_klines_endpoint_and_kline_publish(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    from aiohttp.test_utils import make_mocked_request

    from terminal.server import Terminal

    async def run() -> None:
        term = Terminal(Settings(demo=True))
        sent: list[dict] = []
        term.hub.publish = sent.append  # type: ignore[method-assign]
        now = int(time.time())
        for i in range(120):
            term.btc.on_tick((now - 120 + i) * 1000, 60000.0 + i, 0, 0)
        req = make_mocked_request("GET", "/api/klines?interval=1m&limit=50")
        res = await term.klines(req)
        rows = __import__("json").loads(res.body)
        assert len(rows) == 50 and all(rows[i][0] < rows[i + 1][0] for i in range(49))
        assert all(r[2] >= max(r[1], r[4]) - 1e-9 and r[3] <= min(r[1], r[4]) + 1e-9 for r in rows)
        bad = await term.klines(make_mocked_request("GET", "/api/klines?interval=7m"))
        assert bad.status == 400
        term._on_kline((now, 1.0, 2.0, 0.5, 1.5, 3.0))
        term._on_kline((now, 1.0, 2.0, 0.5, 1.5, 3.0))       # duplicate suppressed
        assert [m["t"] for m in sent] == ["k"]

    asyncio.run(run())


# ---------------------------------------------------------------- spot depth / trade flow
def test_spot_depth_sync_gap_and_aggregate() -> None:
    from terminal.depth import SpotDepthBook, aggregate
    book = SpotDepthBook()
    book.on_event({"U": 90, "u": 95, "b": [["100.50", "9"]], "a": []})          # older than snapshot: dropped
    book.on_event({"U": 96, "u": 102, "b": [["100.50", "2"]], "a": []})         # straddles lid+1
    assert book.apply_snapshot({"lastUpdateId": 100, "bids": [["100.50", "1"], ["99.20", "3"]],
                                "asks": [["101.10", "4"], ["103.00", "5"]]})
    assert book.bids == {100.5: 2.0, 99.2: 3.0} and book.last_u == 102
    assert not book.on_event({"U": 110, "u": 111, "b": [], "a": []})            # gap -> resync
    assert not book.synced and book.gaps == 1
    agg = aggregate({100.5: 2.0, 100.2: 1.0, 99.2: 3.0}, {101.1: 4.0, 101.9: 1.0, 103.0: 5.0}, range_usd=10)
    assert agg is not None
    assert agg["b"] == [[100.0, 3.0], [99.0, 3.0]]            # bids floor to $1
    assert agg["a"] == [[102.0, 5.0], [103.0, 5.0]]           # asks ceil to $1
    assert agg["bb"] == 100.5 and agg["ba"] == 101.1


def test_trade_flow_buckets_per_second() -> None:
    from terminal.btc import BtcState
    st = BtcState(5)
    st.on_trade(1_000_500, 100.0, 2.0, sell=False)
    st.on_trade(1_000_900, 102.0, 1.0, sell=False)
    st.on_trade(1_000_950, 99.0, 3.0, sell=True)
    st.on_trade(1_001_100, 101.0, 1.0, sell=True)
    rows = st.flow_rows()
    assert rows[0][0] == 1000 and rows[0][1] == 3.0 and abs(rows[0][2] - 302.0) < 1e-9
    assert rows[0][3] == 3.0 and rows[1] == [1001, 0.0, 0.0, 1.0, 101.0]
    assert st.flow_dirty == {1000, 1001}


def test_heat_row_buckets_like_ui():
    from terminal.server import HEAT_HALF, heat_row
    agg = {"bb": 100000.4, "ba": 100000.6, "b": [[100000.4, 1.234], [99990.2, 0.5]], "a": [[100000.6, 2.0], [100500.0, 9.0]]}
    t, lo, q = heat_row(agg, 123)
    assert t == 123 and lo == 99990 and len(q) == 12   # trimmed to the non-empty span 99990..100001
    assert HEAT_HALF >= 100
    assert q[100000 - lo] == 123          # bid floored to 100000
    assert q[100001 - lo] == 200          # ask ceiled to 100001
    assert q[99990 - lo] == 50
    assert sum(q) == 373                  # the ask $500 away is outside the window
