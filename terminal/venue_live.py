"""Live predict.fun venue: discovery, books over WS, account, orders."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import aiohttp

from .books import split_book
from .config import Settings
from .markets import MarketInfo, parse_category, slug_for, window_start
from .predict_api import PredictError, PredictRest, PredictWS
from .trading import BuiltOrder, OrderFactory, OrderRejected
from .venue import BookState, OpenOrder, Position, Publish, Venue

BSC_RPC = "https://bsc-dataseed.bnbchain.org/"
USDT_BSC = "0x55d398326f99059fF775485246999027B3197955"


def _wei(v: Any) -> float:
    try:
        return int(str(v)) / 1e18
    except (TypeError, ValueError):
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0


class LiveVenue(Venue):
    name = "predict.fun"

    def __init__(self, s: Settings, publish: Publish) -> None:
        super().__init__(s, publish)
        self.factory = OrderFactory(s.private_key, s.predict_account)
        self.address = self.factory.address
        self.rest = PredictRest(s.api_url, s.api_key, self.address, self.factory.sign_auth_message)
        self.ws = PredictWS(s.ws_url, s.api_key, self._on_ws, self._wallet_topic, self._on_ws_status)
        self._tasks: list[asyncio.Task[Any]] = []
        self._last_fetch: dict[str, float] = {}
        self._account_kick = asyncio.Event()
        self._rpc: aiohttp.ClientSession | None = None
        self._force_fetch: set[str] = set()

    # ------------------------------------------------------------------
    async def start(self) -> None:
        await self.rest.start()
        self._rpc = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=6))
        self.log("info", f"кошелёк для ордеров: {self.address} "
                         f"({'Predict account' if self.s.is_predict_account else 'EOA'}, подписант {self.factory.eoa})")
        try:
            await self.rest.ensure_jwt()
            self.log("info", "авторизация predict.fun: OK")
        except Exception as exc:  # noqa: BLE001
            self.log("error", f"авторизация predict.fun не прошла: {exc}")
        self._tasks = [
            asyncio.create_task(self.ws.run(), name="predict_ws"),
            asyncio.create_task(self._discovery_loop(), name="discovery"),
            asyncio.create_task(self._account_loop(), name="account"),
            asyncio.create_task(self._verify_owner(), name="verify_owner"),
        ]

    async def stop(self) -> None:
        self.ws.stop()
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        await self.rest.close()
        if self._rpc is not None:
            await self._rpc.close()

    def rpm(self) -> int | None:
        return self.rest.rpm

    async def _verify_owner(self) -> None:
        msg = await asyncio.to_thread(self.factory.verify_owner)
        if msg:
            self.log("warning", msg)

    # ---- websocket ------------------------------------------------------
    async def _wallet_topic(self) -> str | None:
        token = await self.rest.ensure_jwt()
        return f"predictWalletEvents/{token}"

    def _on_ws_status(self, up: bool) -> None:
        self.connected = up
        self.publish(self.status())
        if up:
            # books are resent as snapshots on subscribe; mark all stale until then
            pass

    def _on_ws(self, topic: str, data: dict[str, Any]) -> None:
        if topic.startswith("predictOrderbook/"):
            try:
                market_id = int(data.get("marketId") or topic.split("/", 1)[1])
            except ValueError:
                return
            self._apply_book(market_id, data)
        elif topic.startswith("predictWalletEvents/"):
            self._on_wallet_event(data)
        elif topic.startswith("predictMarketChanged/"):
            m = self.markets.get(int(data.get("marketId") or 0))
            if m is not None:
                self._force_fetch.add(m.slug)      # start/end price or status changed: refetch now
        elif topic.startswith("predictTradingStatus/"):
            m = self.markets.get(int(data.get("marketId") or 0))
            st = str(data.get("tradingStatus") or "")
            if m is not None and st and st != m.trading_status:
                m.trading_status = st
                self.push_markets()

    def _apply_book(self, market_id: int, data: dict[str, Any]) -> None:
        m = self.markets.get(market_id)
        precision = m.precision if m else 2
        ts = int(data.get("updateTimestampMs") or 0)
        cur = self.books.get(market_id)
        if cur is not None and ts and ts < cur.ts:
            return
        up, down = split_book(data.get("bids") or [], data.get("asks") or [], precision)
        self.books[market_id] = BookState(up, down, ts, time.time())
        self.dirty_books.add(market_id)

    def _on_wallet_event(self, ev: dict[str, Any]) -> None:
        kind = str(ev.get("type") or "")
        d = ev.get("details") or {}
        side = "BUY" if str(d.get("quoteType", "")).upper() == "BID" else "SELL"
        text = (f"{kind}: {side} {d.get('outcomeName') or d.get('outcome') or ''} "
                f"{d.get('quantityFilled') or d.get('quantity') or ''} @ {d.get('price') or ''} "
                f"({d.get('strategyType') or ''})")
        level = "error" if kind in ("orderNotAccepted", "orderTransactionFailed") else "info"
        if kind in ("orderTransactionSuccess", "orderNotAccepted", "orderTransactionFailed", "orderCancelled",
                    "orderExpired", "orderAccepted"):
            self.log(level, text)
        self.publish({"t": "wallet_event", "event": ev})
        self._account_kick.set()

    # ---- discovery ------------------------------------------------------
    async def _fetch_market(self, series, start_s: int) -> MarketInfo | None:  # type: ignore[no-untyped-def]
        slug = slug_for(series, start_s)
        raw = await self.rest.category(slug)
        if not raw:
            return None
        return parse_category(raw, series)

    async def _discovery_loop(self) -> None:
        while True:
            try:
                await self._discover_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self.log("warning", f"поиск рынков: {exc}")
            await asyncio.sleep(1.0)

    async def _discover_once(self) -> None:
        now = time.time()
        changed = False
        for series in self.s.series:
            cur = window_start(series, now)
            for start in (cur, cur + series.period_s, cur - series.period_s):
                slug = slug_for(series, start)
                known = next((m for m in self.markets.values() if m.slug == slug), None)
                # refresh cadence: current without start price 2 s, current 10 s, next 20 s, previous 5 s
                # REST budget: predict.fun allows 240 req/min per API key (shared with any bot on the same key)
                if known is None:
                    every = 15.0 if start != cur - series.period_s else 120.0
                elif start == cur:
                    every = 3.0 if (known.start_price is None and now - start < 90) else 30.0
                elif start < cur:
                    every = 10.0 if (known.end_price is None and now - start - series.period_s < 180) else 3600.0
                else:
                    every = 60.0
                if slug in self._force_fetch:
                    every = 0.0
                    self._force_fetch.discard(slug)
                if now - self._last_fetch.get(slug, 0) < every:
                    continue
                self._last_fetch[slug] = now
                info = await self._fetch_market(series, start)
                if info is None or not info.market_id:
                    continue
                old = self.markets.get(info.market_id)
                self.markets[info.market_id] = info
                if old is None:
                    changed = True
                    self.log("info", f"рынок {info.title or info.slug} (id {info.market_id})")
                    await self.ws.subscribe(f"predictOrderbook/{info.market_id}")
                    await self.ws.subscribe(f"predictMarketChanged/{info.market_id}")
                    await self.ws.subscribe(f"predictTradingStatus/{info.market_id}")
                    asyncio.create_task(self._snapshot(info.market_id))
                elif (old.start_price, old.end_price, old.trading_status) != (
                        info.start_price, info.end_price, info.trading_status):
                    changed = True
        # forget markets that ended > 10 min ago
        for mid in [mid for mid, m in self.markets.items() if m.end_ms and m.end_ms < now * 1000 - 600_000]:
            self.markets.pop(mid, None)
            self.books.pop(mid, None)
            for topic in ("predictOrderbook", "predictMarketChanged", "predictTradingStatus"):
                await self.ws.unsubscribe(f"{topic}/{mid}")
            changed = True
        if changed:
            self.push_markets()

    async def _snapshot(self, market_id: int) -> None:
        try:
            data = await self.rest.orderbook(market_id)
            if market_id not in self.books:
                self._apply_book(market_id, data)
        except Exception as exc:  # noqa: BLE001
            self.log("warning", f"снимок стакана {market_id}: {exc}")

    # ---- account ------------------------------------------------------------
    async def _usdt_balance(self) -> float | None:
        assert self._rpc is not None
        data = "0x70a08231" + self.address.lower().replace("0x", "").rjust(64, "0")
        body = {"jsonrpc": "2.0", "id": 1, "method": "eth_call",
                "params": [{"to": USDT_BSC, "data": data}, "latest"]}
        async with self._rpc.post(BSC_RPC, json=body) as r:
            res = await r.json(content_type=None)
        val = res.get("result")
        return int(val, 16) / 1e18 if val else None

    async def _account_loop(self) -> None:
        while True:
            try:
                await self.refresh_account()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self.log("warning", f"обновление счёта: {exc}")
            self._account_kick.clear()
            try:
                await asyncio.wait_for(self._account_kick.wait(), 6.0)
                await asyncio.sleep(0.4)       # let the venue index the fill
            except asyncio.TimeoutError:
                pass

    def _outcome_name(self, market_id: int, token_id: str, fallback: str) -> str:
        m = self.markets.get(market_id)
        if m is not None:
            if token_id == m.up.token_id:
                return "UP"
            if token_id == m.down.token_id:
                return "DOWN"
        return (fallback or "?").upper()

    async def refresh_account(self) -> None:
        pos_raw, ord_raw, bal = await asyncio.gather(self.rest.positions(), self.rest.open_orders(),
                                                     self._usdt_balance(), return_exceptions=True)
        acc = self.account
        if isinstance(bal, (int, float)):
            acc.balance = float(bal)
        if isinstance(pos_raw, list):
            positions = []
            for p in pos_raw:
                mk = p.get("market") or {}
                oc = p.get("outcome") or {}
                mid = int(mk.get("id") or 0)
                token = str(oc.get("onChainId") or "")
                shares = _wei(p.get("amount"))
                if shares < 1e-6:
                    continue
                status = str(oc.get("status") or "")
                positions.append(Position(mid, self._outcome_name(mid, token, str(oc.get("name") or "")), token, shares,
                                          float(p.get("averageBuyPriceUsd") or 0), str(mk.get("categorySlug") or ""),
                                          status if status in ("WON", "LOST") else ""))
            acc.positions = positions
        elif isinstance(pos_raw, Exception):
            self.log("warning", f"позиции: {pos_raw}")
        if isinstance(ord_raw, list):
            orders = []
            for o in ord_raw:
                so = o.get("order") or {}
                mid = int(o.get("marketId") or 0)
                side = "BUY" if str(so.get("side")) in ("0", "Bid") else "SELL"
                maker, taker = _wei(so.get("makerAmount")), _wei(so.get("takerAmount"))
                shares, cost = (taker, maker) if side == "BUY" else (maker, taker)
                price = cost / shares if shares else 0.0
                orders.append(OpenOrder(str(so.get("hash") or ""), mid,
                                        self._outcome_name(mid, str(so.get("tokenId") or ""), ""), side,
                                        round(price, 4), shares, _wei(o.get("amountFilled")), str(o.get("id") or "")))
            acc.orders = orders
        elif isinstance(ord_raw, Exception):
            self.log("warning", f"ордера: {ord_raw}")
        acc.updated = time.time()
        self.push_account()

    # ---- trading ------------------------------------------------------------
    async def _send(self, m: MarketInfo, outcome: str, built: BuiltOrder) -> dict[str, Any]:
        t0 = time.perf_counter()
        try:
            res = await self.rest.create_order(built.data)
            ok, msg = True, str(res.get("code") or "OK")
        except PredictError as exc:
            ok, msg = False, str(exc)
            body = exc.body if isinstance(exc.body, dict) else {}
            if body.get("error"):
                msg = f"{body.get('error')}: {body.get('message', '')}"
        except Exception as exc:  # noqa: BLE001
            ok, msg = False, f"{type(exc).__name__}: {exc}"
        latency = (time.perf_counter() - t0) * 1000
        px = f"{built.price:.4f}"
        usdt = built.preview.usdt if built.preview else built.shares * built.price
        text = (f"{'✓' if ok else '✗'} {built.side} {outcome} {built.strategy} {built.shares:.2f} sh @ {px} "
                f"(~{usdt:.2f}$) {msg} [{latency:.0f} ms]")
        self.log("info" if ok else "error", text)
        self.journal.write({"venue": self.name, "market_id": m.market_id, "slug": m.slug, "outcome": outcome,
                            "side": built.side, "strategy": built.strategy, "shares": f"{built.shares:.4f}",
                            "price": px, "usdt": f"{usdt:.4f}", "order_hash": built.order_hash,
                            "latency_ms": f"{latency:.0f}", "result": msg[:200]})
        self._account_kick.set()
        return {"ok": ok, "msg": text, "latency_ms": latency, "hash": built.order_hash}

    async def buy(self, market_id: int, outcome: str, limit_price: float, usdt: float) -> dict[str, Any]:
        m = self.market(market_id)
        built = self.factory.market_buy(m, outcome, self.book(market_id).outcome(outcome), limit_price, usdt,
                                        self.s.slippage_ticks)
        return await self._send(m, outcome, built)

    async def sell(self, market_id: int, outcome: str, limit_price: float, shares: float) -> dict[str, Any]:
        m = self.market(market_id)
        built = self.factory.market_sell(m, outcome, self.book(market_id).outcome(outcome), limit_price, shares,
                                         self.s.slippage_ticks)
        return await self._send(m, outcome, built)

    async def limit(self, market_id: int, outcome: str, side: str, price: float, shares: float,
                    post_only: bool) -> dict[str, Any]:
        m = self.market(market_id)
        built = self.factory.limit(m, outcome, side, price, shares, post_only)
        return await self._send(m, outcome, built)

    async def cancel(self, hashes: list[str]) -> dict[str, Any]:
        if not hashes:
            return {"ok": True, "msg": "нечего отменять"}
        t0 = time.perf_counter()
        try:
            ids = [o.order_id for o in self.account.orders if o.order_hash in hashes and o.order_id]
            res = await self.rest.remove_orders(hashes, ids) or {}
            if "removed" not in res and isinstance(res.get("data"), dict):
                res = res["data"]
            removed = len(res.get("removed") or [])
            rejected = res.get("rejected") or []
            ok = not rejected
            text = f"отмена: снято {removed}, нечего снимать {len(res.get('noop') or [])}" + (
                f", отказ {json.dumps(rejected)[:200]}" if rejected else "")
        except Exception as exc:  # noqa: BLE001
            ok, text = False, f"отмена не прошла: {exc}"
        latency = (time.perf_counter() - t0) * 1000
        self.log("info" if ok else "error", f"{text} [{latency:.0f} ms]")
        self._account_kick.set()
        return {"ok": ok, "msg": text, "latency_ms": latency}


__all__ = ["LiveVenue", "OrderRejected"]


class DataVenue(LiveVenue):
    """Read-only predict.fun: market discovery and order books with the API key alone (no wallet, no orders).
    Used by the recorder and by the DOM bot in paper mode when no private key is configured."""

    name = "predict.fun (только данные)"

    def __init__(self, s: Settings, publish: Publish) -> None:  # noqa: D107 - LiveVenue.__init__ needs a wallet
        Venue.__init__(self, s, publish)
        self.factory = None  # type: ignore[assignment]
        self.address = "(без кошелька)"
        self.rest = PredictRest(s.api_url, s.api_key, "", lambda message: "")
        self.ws = PredictWS(s.ws_url, s.api_key, self._on_ws, None, self._on_ws_status)
        self._tasks: list[asyncio.Task[Any]] = []
        self._last_fetch: dict[str, float] = {}
        self._account_kick = asyncio.Event()
        self._rpc: aiohttp.ClientSession | None = None
        self._force_fetch: set[str] = set()

    async def start(self) -> None:
        await self.rest.start()
        self._tasks = [asyncio.create_task(self.ws.run(), name="predict_ws"),
                       asyncio.create_task(self._discovery_loop(), name="discovery")]

    async def stop(self) -> None:
        self.ws.stop()
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        await self.rest.close()

    async def _no_orders(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise OrderRejected("режим только данных: ордера не отправляются")

    buy = sell = limit = _no_orders  # type: ignore[assignment]

    async def cancel(self, hashes: list[str]) -> dict[str, Any]:
        return {"ok": True, "msg": "режим только данных"}
