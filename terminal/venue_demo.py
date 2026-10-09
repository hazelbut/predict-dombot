"""Offline simulator: same UI and order flow, fake money, fake book driven by the BTC walk."""

from __future__ import annotations

import asyncio
import math
import random
import secrets
import time
from typing import Any

from .books import OutcomeBook, split_book, take_buy, take_sell
from .btc import BtcState
from .config import Settings
from .markets import MarketInfo, Outcome, window_start
from .trading import MIN_ORDER_USDT, OrderRejected
from .venue import BookState, OpenOrder, Position, Publish, Venue


def phi(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


class DemoVenue(Venue):
    name = "demo"

    def __init__(self, s: Settings, publish: Publish, btc: BtcState) -> None:
        super().__init__(s, publish)
        self.btc = btc
        self.address = "0xDEMO000000000000000000000000000000000000"
        self.account.balance = 200.0
        self._tasks: list[asyncio.Task[Any]] = []
        self._next_id = 1_000_000

    async def start(self) -> None:
        self.connected = True
        self._tasks = [asyncio.create_task(self._loop(), name="demo")]
        self.log("info", "ДЕМО-режим: деньги ненастоящие, стакан симулируется")

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    def _ensure_markets(self, now: float) -> bool:
        changed = False
        for series in self.s.series:
            cur = window_start(series, now)
            for start in (cur, cur + series.period_s):
                slug = f"{series.prefix}-{start}"
                if any(m.slug == slug for m in self.markets.values()):
                    continue
                self._next_id += 1
                mid = self._next_id
                self.markets[mid] = MarketInfo(
                    series.prefix, series.label, slug, f"{series.label} {time.strftime('%H:%M', time.localtime(start))}",
                    mid, mid, start * 1000, (start + series.period_s) * 1000, None, None, "OPEN", "REGISTERED",
                    2, 200, False, True, Outcome("Up", f"{mid}1", 1), Outcome("Down", f"{mid}2", 2))
                changed = True
        return changed

    def _gen_book(self, m: MarketInfo, now: float) -> None:
        tte = max(m.end_ms / 1000 - now, 0.5)
        if m.start_price is None or math.isnan(self.btc.mid):
            p = 0.5
        else:
            p = phi((self.btc.mid - m.start_price) / (4.0 * math.sqrt(tte)))
        p = min(max(p + random.gauss(0, 0.004), 0.02), 0.98)
        mid = round(p, 2)
        asks, bids = [], []
        px = round(max(mid + 0.01, 0.02), 2)
        for _ in range(12):
            if px > 0.99:
                break
            asks.append([px, round(random.uniform(15, 450) * (1.6 if random.random() < 0.15 else 1), 2)])
            px = round(px + 0.01, 2)
        px = round(min(mid - 0.0, 0.98), 2)
        if px >= asks[0][0] if asks else False:
            px = round(asks[0][0] - 0.01, 2)
        for _ in range(12):
            if px < 0.01:
                break
            bids.append([px, round(random.uniform(15, 450), 2)])
            px = round(px - 0.01, 2)
        up, down = split_book(bids, asks, m.precision)
        self.books[m.market_id] = BookState(up, down, int(now * 1000), now)
        self.dirty_books.add(m.market_id)

    async def _loop(self) -> None:
        tick = 0
        while True:
            now = time.time()
            changed = self._ensure_markets(now)
            for m in list(self.markets.values()):
                if m.start_price is None and now * 1000 >= m.start_ms and not math.isnan(self.btc.mid):
                    sp = self.btc.price_at(m.start_ms // 1000) or self.btc.mid
                    m.start_price = round(sp, 2)
                    changed = True
                if m.end_price is None and now * 1000 >= m.end_ms and not math.isnan(self.btc.mid):
                    m.end_price = round(self.btc.mid, 2)
                    m.trading_status = "CLOSED"
                    self._settle(m)
                    changed = True
                if m.trading_status == "OPEN" and (tick % 2 == 0 or m.market_id not in self.books):
                    self._gen_book(m, now)
                    self._match_limits(m)
            for mid in [k for k, m in self.markets.items() if m.end_ms < now * 1000 - 600_000]:
                self.markets.pop(mid, None)
                self.books.pop(mid, None)
                changed = True
            if changed:
                self.push_markets()
                self.push_account()
            tick += 1
            await asyncio.sleep(0.25)

    # ---- account simulation ------------------------------------------------
    def _add_position(self, m: MarketInfo, outcome: str, shares: float, price: float) -> None:
        token = m.outcome(outcome).token_id
        for p in self.account.positions:
            if p.token_id == token and not p.resolved:
                total = p.shares + shares
                if total <= 1e-9:
                    self.account.positions.remove(p)
                else:
                    if shares > 0:
                        p.avg = (p.avg * p.shares + price * shares) / total
                    p.shares = total
                return
        if shares > 0:
            self.account.positions.append(Position(m.market_id, outcome, token, shares, price, m.slug))

    def _settle(self, m: MarketInfo) -> None:
        if m.start_price is None or m.end_price is None:
            return
        winner = "UP" if m.end_price > m.start_price else "DOWN"
        for p in self.account.positions:
            if p.market_id == m.market_id and not p.resolved:
                p.resolved = "WON" if p.outcome == winner else "LOST"
                if p.resolved == "WON":
                    self.account.balance = (self.account.balance or 0) + p.shares
                self.log("info", f"расчёт {m.label}: {p.outcome} {p.shares:.2f} sh -> {p.resolved}")
        self.account.orders = [o for o in self.account.orders if o.market_id != m.market_id]

    def _match_limits(self, m: MarketInfo) -> None:
        book = self.books.get(m.market_id)
        if book is None:
            return
        for o in list(self.account.orders):
            if o.market_id != m.market_id:
                continue
            ob = book.outcome(o.outcome)
            if o.side == "BUY" and ob.best_ask is not None and ob.best_ask <= o.price:
                self._add_position(m, o.outcome, o.shares, o.price)
            elif o.side == "SELL" and ob.best_bid is not None and ob.best_bid >= o.price:
                self._add_position(m, o.outcome, -o.shares, o.price)
                self.account.balance = (self.account.balance or 0) + o.shares * o.price
            else:
                continue
            self.account.orders.remove(o)
            self.log("info", f"демо: лимитка исполнена {o.side} {o.outcome} {o.shares:.2f} @ {o.price}")
            self.push_account()

    async def _latency(self) -> float:
        lat = random.uniform(0.08, 0.25)
        await asyncio.sleep(lat)
        return lat * 1000

    def _check_open(self, m: MarketInfo) -> None:
        if m.trading_status != "OPEN" or time.time() * 1000 >= m.end_ms:
            raise OrderRejected("рынок закрыт")

    async def buy(self, market_id: int, outcome: str, limit_price: float, usdt: float) -> dict[str, Any]:
        m = self.market(market_id)
        self._check_open(m)
        fill = take_buy(self.book(market_id).outcome(outcome).asks, limit_price, budget_usdt=usdt)
        if fill.usdt < MIN_ORDER_USDT:
            raise OrderRejected(f"в стакане нет предложений по цене ≤ {limit_price} на {MIN_ORDER_USDT}$+")
        fee = fill.usdt * 0.02 * min(fill.avg, 1 - fill.avg) / max(fill.avg, 1e-9)
        if (self.account.balance or 0) < fill.usdt + fee:
            raise OrderRejected("недостаточно средств")
        lat = await self._latency()
        self.account.balance = (self.account.balance or 0) - fill.usdt - fee
        self._add_position(m, outcome, fill.shares, fill.avg)
        text = f"✓ BUY {outcome} MARKET {fill.shares:.2f} sh @ {fill.avg:.4f} (~{fill.usdt:.2f}$) демо [{lat:.0f} ms]"
        self.log("info", text)
        self.journal.write({"venue": "demo", "market_id": m.market_id, "slug": m.slug, "outcome": outcome,
                            "side": "BUY", "strategy": "MARKET", "shares": f"{fill.shares:.4f}",
                            "price": f"{fill.avg:.4f}", "usdt": f"{fill.usdt:.4f}", "latency_ms": f"{lat:.0f}",
                            "result": "OK"})
        self.push_account()
        return {"ok": True, "msg": text, "latency_ms": lat}

    async def sell(self, market_id: int, outcome: str, limit_price: float, shares: float) -> dict[str, Any]:
        m = self.market(market_id)
        self._check_open(m)
        have = self.account.shares_of(m.outcome(outcome).token_id)
        shares = min(shares, have)
        if shares <= 0:
            raise OrderRejected(f"нет позиции {outcome}")
        fill = take_sell(self.book(market_id).outcome(outcome).bids, limit_price, shares)
        if fill.usdt < MIN_ORDER_USDT:
            raise OrderRejected(f"в стакане нет покупателей по цене ≥ {limit_price} на {MIN_ORDER_USDT}$+")
        lat = await self._latency()
        self.account.balance = (self.account.balance or 0) + fill.usdt * 0.99
        self._add_position(m, outcome, -fill.shares, fill.avg)
        text = f"✓ SELL {outcome} MARKET {fill.shares:.2f} sh @ {fill.avg:.4f} (~{fill.usdt:.2f}$) демо [{lat:.0f} ms]"
        self.log("info", text)
        self.push_account()
        return {"ok": True, "msg": text, "latency_ms": lat}

    async def limit(self, market_id: int, outcome: str, side: str, price: float, shares: float,
                    post_only: bool) -> dict[str, Any]:
        m = self.market(market_id)
        self._check_open(m)
        if shares * price < MIN_ORDER_USDT:
            raise OrderRejected(f"объём меньше {MIN_ORDER_USDT}$")
        if side == "BUY":
            if (self.account.balance or 0) < shares * price:
                raise OrderRejected("недостаточно средств")
            self.account.balance = (self.account.balance or 0) - shares * price
        else:
            have = self.account.shares_of(m.outcome(outcome).token_id)
            if have < shares - 1e-9:
                raise OrderRejected(f"нет столько {outcome} (есть {have:.2f})")
        lat = await self._latency()
        h = "0x" + secrets.token_hex(32)
        self.account.orders.append(OpenOrder(h, market_id, outcome, side, round(price, 2), shares, 0.0))
        text = f"✓ {side} {outcome} LIMIT {shares:.2f} sh @ {price:.2f} демо [{lat:.0f} ms]"
        self.log("info", text)
        self.push_account()
        self._match_limits(m)
        return {"ok": True, "msg": text, "latency_ms": lat, "hash": h}

    async def cancel(self, hashes: list[str]) -> dict[str, Any]:
        lat = await self._latency()
        n = 0
        for o in list(self.account.orders):
            if o.order_hash in hashes:
                if o.side == "BUY":
                    self.account.balance = (self.account.balance or 0) + o.shares * o.price
                self.account.orders.remove(o)
                n += 1
        text = f"отмена: снято {n} [{lat:.0f} ms]"
        self.log("info", text)
        self.push_account()
        return {"ok": True, "msg": text, "latency_ms": lat}
