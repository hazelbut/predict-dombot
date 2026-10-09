"""Venue state shared by the live predict.fun connection and the demo simulator."""

from __future__ import annotations

import csv
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .books import OutcomeBook
from .config import Settings
from .markets import MarketInfo

log = logging.getLogger("venue")

Publish = Callable[[dict[str, Any]], None]


@dataclass(slots=True)
class BookState:
    up: OutcomeBook
    down: OutcomeBook
    ts: int
    received: float

    def outcome(self, name: str) -> OutcomeBook:
        return self.up if name.upper() == "UP" else self.down

    def to_json(self) -> dict[str, Any]:
        return {"up": {"bids": self.up.bids[:60], "asks": self.up.asks[:60]},
                "down": {"bids": self.down.bids[:60], "asks": self.down.asks[:60]},
                "ts": self.ts, "age_ms": int((time.time() - self.received) * 1000)}


@dataclass(slots=True)
class Position:
    market_id: int
    outcome: str        # UP / DOWN / name
    token_id: str
    shares: float
    avg: float
    title: str = ""
    resolved: str = ""  # '' while trading, WON / LOST after resolution

    def to_json(self) -> dict[str, Any]:
        return {"market_id": self.market_id, "outcome": self.outcome, "token_id": self.token_id,
                "shares": self.shares, "avg": self.avg, "title": self.title, "resolved": self.resolved}


@dataclass(slots=True)
class OpenOrder:
    order_hash: str
    market_id: int
    outcome: str
    side: str
    price: float
    shares: float
    filled: float
    order_id: str = ""

    def to_json(self) -> dict[str, Any]:
        return {"hash": self.order_hash, "id": self.order_id, "market_id": self.market_id, "outcome": self.outcome, "side": self.side,
                "price": self.price, "shares": self.shares, "filled": self.filled}


@dataclass
class Account:
    balance: float | None = None
    positions: list[Position] = field(default_factory=list)
    orders: list[OpenOrder] = field(default_factory=list)
    updated: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return {"balance": self.balance, "positions": [p.to_json() for p in self.positions],
                "orders": [o.to_json() for o in self.orders], "updated": self.updated}

    def shares_of(self, token_id: str) -> float:
        return sum(p.shares for p in self.positions if p.token_id == token_id and not p.resolved)


class Venue:
    """Base: holds markets/books/account and pushes changes to the UI hub."""

    name = "base"

    def __init__(self, s: Settings, publish: Publish) -> None:
        self.s = s
        self.publish = publish
        self.markets: dict[int, MarketInfo] = {}      # market_id -> info
        self.books: dict[int, BookState] = {}
        self.account = Account()
        self.address = ""
        self.connected = False
        self.dirty_books: set[int] = set()
        self.journal = Journal(Path("logs") / "trades.csv")
        self.recent_logs: deque[dict[str, Any]] = deque(maxlen=150)

    # ---- UI helpers ---------------------------------------------------------
    def log(self, level: str, msg: str) -> None:
        getattr(log, level if level in ("info", "warning", "error") else "info")(msg)
        entry = {"t": "log", "level": level, "msg": msg, "ts": int(time.time() * 1000)}
        self.recent_logs.append(entry)
        self.publish(entry)

    def push_markets(self) -> None:
        now = time.time() * 1000
        items = [m.to_json() for m in self.markets.values() if m.end_ms > now - 10 * 60_000]
        items.sort(key=lambda d: (d["series"], d["start_ms"]))
        self.publish({"t": "markets", "items": items})

    def push_account(self) -> None:
        self.publish({"t": "account", **self.account.to_json()})

    def status(self) -> dict[str, Any]:
        return {"t": "status", "venue": self.name, "connected": self.connected, "address": self.address,
                "rpm": self.rpm()}

    def rpm(self) -> int | None:
        return None

    def market(self, market_id: int) -> MarketInfo:
        m = self.markets.get(int(market_id))
        if m is None:
            raise KeyError(f"рынок {market_id} не найден")
        return m

    def book(self, market_id: int) -> BookState:
        b = self.books.get(int(market_id))
        if b is None:
            raise KeyError("стакан ещё не получен")
        return b

    # ---- trading API (implemented by subclasses) ------------------------------
    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def buy(self, market_id: int, outcome: str, limit_price: float, usdt: float) -> dict[str, Any]:
        raise NotImplementedError

    async def sell(self, market_id: int, outcome: str, limit_price: float, shares: float) -> dict[str, Any]:
        raise NotImplementedError

    async def limit(self, market_id: int, outcome: str, side: str, price: float, shares: float,
                    post_only: bool) -> dict[str, Any]:
        raise NotImplementedError

    async def cancel(self, hashes: list[str]) -> dict[str, Any]:
        raise NotImplementedError

    async def refresh_account(self) -> None: ...


class Journal:
    FIELDS = ["time", "venue", "market_id", "slug", "outcome", "side", "strategy", "shares", "price", "usdt",
              "order_hash", "latency_ms", "result"]

    def __init__(self, path: Path) -> None:
        self.path = path

    def write(self, row: dict[str, Any]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            new = not self.path.exists()
            with self.path.open("a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=self.FIELDS, extrasaction="ignore")
                if new:
                    w.writeheader()
                w.writerow({"time": time.strftime("%Y-%m-%d %H:%M:%S"), **row})
        except OSError as exc:
            log.warning("journal write failed: %s", exc)
