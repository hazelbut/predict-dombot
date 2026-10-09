"""LiveVenue against a fake predict.fun (REST + WS) and a fake BSC RPC."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from aiohttp import web
from eth_account import Account
from eth_account.messages import encode_defunct

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terminal import venue_live  # noqa: E402
from terminal.config import Settings, parse_series  # noqa: E402

API_KEY = "test-key"


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat().replace("+00:00", "Z")


class FakePredict:
    def __init__(self, wallet: str, hash_route: bool = True) -> None:
        self.wallet = wallet
        self.hash_route = hash_route
        self.open: list[dict] = []
        self.removed_ids: list[str] = []
        self.orders: list[dict] = []
        self.headers_seen: list[dict] = []
        self.subscribed: list[str] = []
        self.heartbeat_echo: list = []
        self.auth_signer: str | None = None
        self.removed: list[str] = []

    def category(self, slug: str) -> dict | None:
        if not slug.startswith("btc-updown-5m-"):
            return None
        start = int(slug.rsplit("-", 1)[1])
        mid = start // 300
        return {"id": mid, "slug": slug, "title": f"BTC {start}", "startsAt": iso(start), "endsAt": iso(start + 300),
                "markets": [{"id": mid, "decimalPrecision": 2, "feeRateBps": 200, "isNegRisk": False,
                             "isYieldBearing": True, "tradingStatus": "OPEN",
                             "outcomes": [{"indexSet": 1, "name": "Up", "onChainId": f"{mid}1"},
                                          {"indexSet": 2, "name": "Down", "onChainId": f"{mid}2"}],
                             "variantData": {"startPrice": "60000.5"}}]}

    def app(self) -> web.Application:
        app = web.Application()

        def need_key(req: web.Request) -> None:
            if req.headers.get("x-api-key") != API_KEY:
                raise web.HTTPUnauthorized(text=json.dumps({"success": False, "error": "unauthorized"}))

        def need_jwt(req: web.Request) -> None:
            need_key(req)
            if req.headers.get("Authorization") != "Bearer JWT123":
                raise web.HTTPUnauthorized(text=json.dumps({"success": False, "error": "unauthorized"}))

        async def auth_message(req: web.Request) -> web.Response:
            need_key(req)
            return web.json_response({"success": True, "data": {"message": "Sign in to predict 42"}})

        async def auth(req: web.Request) -> web.Response:
            need_key(req)
            body = await req.json()
            signer = Account.recover_message(encode_defunct(text=body["message"]), signature=body["signature"])
            self.auth_signer = body["signer"]
            assert signer == body["signer"]
            return web.json_response({"success": True, "data": {"token": "JWT123"}})

        async def category(req: web.Request) -> web.Response:
            need_key(req)
            c = self.category(req.match_info["slug"])
            if c is None:
                return web.json_response({"success": False, "error": "not_found"}, status=404)
            return web.json_response({"success": True, "data": c})

        async def orderbook(req: web.Request) -> web.Response:
            need_key(req)
            return web.json_response({"success": True, "data": {"marketId": int(req.match_info["id"]),
                                                                 "updateTimestampMs": 1, "asks": [[0.6, 50]],
                                                                 "bids": [[0.58, 70]]}})

        async def positions(req: web.Request) -> web.Response:
            need_jwt(req)
            mid = int(time.time()) // 300
            return web.json_response({"success": True, "cursor": None, "data": [
                {"amount": str(12 * 10**18), "averageBuyPriceUsd": "0.61", "market": {"id": mid},
                 "outcome": {"onChainId": f"{mid}1", "name": "Up", "status": None}}]})

        async def orders_get(req: web.Request) -> web.Response:
            need_jwt(req)
            return web.json_response({"success": True, "data": self.open})

        async def orders_post(req: web.Request) -> web.Response:
            need_jwt(req)
            body = await req.json()
            self.orders.append(body)
            if body["data"]["strategy"] == "LIMIT":
                self.open.append({"id": str(len(self.orders)), "marketId": int(time.time()) // 300,
                                  "amountFilled": "0", "status": "OPEN", "strategy": "LIMIT",
                                  "order": body["data"]["order"]})
            return web.json_response({"success": True, "data": {"code": "OK", "orderId": "1",
                                                                 "orderHash": body["data"]["order"]["hash"]}},
                                     status=201)

        async def remove(req: web.Request) -> web.Response:
            need_jwt(req)
            body = await req.json()
            self.removed += body["data"]["hashes"]
            return web.json_response({"success": True, "removed": body["data"]["hashes"], "rejected": [], "noop": []})

        async def remove_ids(req: web.Request) -> web.Response:
            need_jwt(req)
            body = await req.json()
            self.removed_ids += body["data"]["ids"]
            self.open = [o for o in self.open if o["id"] not in body["data"]["ids"]]
            return web.json_response({"success": True})

        async def ws(req: web.Request) -> web.WebSocketResponse:
            if req.query.get("apiKey") != API_KEY:
                raise web.HTTPUnauthorized()
            sock = web.WebSocketResponse()
            await sock.prepare(req)
            await sock.send_str(json.dumps({"type": "M", "topic": "heartbeat", "data": 777}))
            async for msg in sock:
                d = json.loads(msg.data)
                if d.get("method") == "heartbeat":
                    self.heartbeat_echo.append(d.get("data"))
                elif d.get("method") == "subscribe":
                    topic = d["params"][0]
                    self.subscribed.append(topic)
                    await sock.send_str(json.dumps({"type": "R", "requestId": d["requestId"], "success": True}))
                    if topic.startswith("predictOrderbook/"):
                        mid = int(topic.split("/")[1])
                        await sock.send_str(json.dumps({"type": "M", "topic": topic, "data": {
                            "marketId": mid, "updateTimestampMs": int(time.time() * 1000),
                            "asks": [[0.62, 100], [0.63, 200]], "bids": [[0.60, 80], [0.59, 90]]}}))
            return sock

        async def rpc(req: web.Request) -> web.Response:
            return web.json_response({"jsonrpc": "2.0", "id": 1, "result": hex(123 * 10**18)})

        app.router.add_get("/v1/auth/message", auth_message)
        app.router.add_post("/v1/auth", auth)
        app.router.add_get("/v1/categories/{slug}", category)
        app.router.add_get("/v1/markets/{id}/orderbook", orderbook)
        app.router.add_get("/v1/positions", positions)
        app.router.add_get("/v1/orders", orders_get)
        app.router.add_post("/v1/orders", orders_post)
        if self.hash_route:
            app.router.add_post("/v1/orders/remove-by-hash", remove)
        app.router.add_post("/v1/orders/remove", remove_ids)
        app.router.add_get("/ws", ws)
        app.router.add_post("/rpc", rpc)
        return app


import pytest  # noqa: E402


@pytest.mark.parametrize("hash_route", [True, False])
def test_live_venue_end_to_end(tmp_path, monkeypatch, hash_route) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    acct = Account.create()
    fake = FakePredict(acct.address, hash_route)

    async def run() -> None:
        runner = web.AppRunner(fake.app())
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
        monkeypatch.setattr(venue_live, "BSC_RPC", f"http://127.0.0.1:{port}/rpc")
        s = Settings(api_key=API_KEY, private_key=acct.key.hex(), api_url=f"http://127.0.0.1:{port}",
                     ws_url=f"ws://127.0.0.1:{port}/ws", series=parse_series("btc-updown-5m:300"))
        events: list[dict] = []
        v = venue_live.LiveVenue(s, events.append)
        await v.start()
        mid = int(time.time()) // 300
        for _ in range(60):
            await asyncio.sleep(0.1)
            if mid in v.books and v.account.positions and fake.heartbeat_echo:
                break
        assert fake.auth_signer == acct.address
        assert f"predictOrderbook/{mid}" in fake.subscribed
        assert any(t.startswith("predictWalletEvents/JWT123") for t in fake.subscribed)
        assert fake.heartbeat_echo == [777]
        b = v.books[mid]
        assert b.up.best_ask == 0.62 and b.down.best_ask == 0.40 and b.down.best_bid == 0.38
        assert v.account.balance == 123.0
        p = v.account.positions[0]
        assert p.outcome == "UP" and p.shares == 12 and p.avg == 0.61
        assert v.markets[mid].start_price == 60000.5

        r = await v.buy(mid, "UP", 0.63, 20)
        assert r["ok"], r
        sent = fake.orders[-1]["data"]
        assert sent["strategy"] == "MARKET" and sent["order"]["tokenId"] == f"{mid}1"
        assert int(sent["order"]["makerAmount"]) / int(sent["order"]["takerAmount"]) <= 0.63 + 1e-9
        r = await v.sell(mid, "DOWN", 0.37, 10)
        assert r["ok"] and fake.orders[-1]["data"]["order"]["side"] == 1
        r = await v.limit(mid, "UP", "BUY", 0.55, 5, post_only=True)
        assert r["ok"] and fake.orders[-1]["data"]["isPostOnly"] is True
        await v.refresh_account()
        assert v.account.orders and v.account.orders[0].order_id
        r = await v.cancel([r["hash"]])
        assert r["ok"], r
        assert (fake.removed if hash_route else fake.removed_ids)
        assert v.rest.rpm > 0
        assert (tmp_path / "logs" / "trades.csv").exists()
        await v.stop()
        await runner.cleanup()

    asyncio.run(run())


def test_favbot_live_against_fake(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The bot places exactly one signed MARKET buy through the real LiveVenue code path."""
    from terminal.favbot import BotCfg, FavBot

    monkeypatch.chdir(tmp_path)
    acct = Account.create()
    fake = FakePredict(acct.address)

    async def run() -> None:
        runner = web.AppRunner(fake.app())
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
        monkeypatch.setattr(venue_live, "BSC_RPC", f"http://127.0.0.1:{port}/rpc")
        s = Settings(api_key=API_KEY, private_key=acct.key.hex(), api_url=f"http://127.0.0.1:{port}",
                     ws_url=f"ws://127.0.0.1:{port}/ws", series=parse_series("btc-updown-5m:300"))
        v = venue_live.LiveVenue(s, lambda m: None)
        await v.start()
        mid = int(time.time()) // 300
        for _ in range(60):
            await asyncio.sleep(0.1)
            if mid in v.books and v.account.balance is not None:
                break
        bot = FavBot(v, BotCfg(live=True, t_from=0, t_to=300), tmp_path)
        await bot.tick()
        await bot.tick()
        buys = [o for o in fake.orders if o["data"]["strategy"] == "MARKET"]
        assert len(buys) == 1
        o = buys[0]["data"]["order"]
        assert o["tokenId"] == f"{mid}1" and o["side"] == 0
        assert int(o["makerAmount"]) / int(o["takerAmount"]) <= 0.63 + 1e-9      # cap = ask 0.62 + 1 tick
        # max spend = stake at the cap price (fills at the book price 0.62 cost <= 1.1)
        assert int(o["makerAmount"]) / 1e18 <= 1.1 * 0.63 / 0.62 + 1e-6
        assert mid in bot.pending and bot.pending[mid][0].outcome == "UP"   # waits for the position to confirm
        await v.stop()
        await runner.cleanup()

    asyncio.run(run())
