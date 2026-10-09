"""Favourite bot (live test of the backtest finding).

Rule: in a BTC Up/Down 5m window, between t_from and t_to seconds after the start, the first time the best ask
of one side is inside [lo, hi] -> market-buy that side for `stake` USDT (price cap ask + 1 tick) and hold to
expiry. One entry per window. Winnings are redeemed on-chain periodically.

Risk caps (all in USDT, counted from this bot's own ledger, open positions counted as already lost):
  daily_loss  - no new entries for the rest of the UTC day
  total_loss  - no new entries at all (delete the state file to reset)
  checkpoints - after 300 settled trades stop if ROI < -3%, after 1000 if ROI < +1%
  STOP file   - create logs/favbot.STOP to pause entries
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .books import take_buy
from .markets import MarketInfo, parse_category
from .trading import MIN_ORDER_USDT, OrderRejected

log = logging.getLogger("favbot")


@dataclass
class BotCfg:
    lo: float = 0.60
    hi: float = 0.70
    t_from: float = 30
    t_to: float = 150
    stake: float = 1.1
    extra_ticks: int = 1            # price cap = ask + N ticks ...
    max_price: float | None = None  # ... but never above this (default: hi)
    max_spread: float = 0.05
    daily_loss: float = 10.0
    total_loss: float = 25.0
    # (trades, minimum ROI): stop if the edge does not show up
    checkpoints: tuple = ((300, -0.03), (1000, 0.01))
    min_balance: float = 3.0
    fee_rate: float = 0.02          # taker fee 2% x min(p, 1-p), withheld in shares
    series: str = "btc-updown-5m"
    live: bool = False
    redeem_every: float = 300.0
    redeem: bool = False            # predict.fun credits winnings itself; on-chain redeem only wastes BNB gas
    fill_wait: float = 20.0         # s to wait for a fill / no-fill signal before giving up on the window
    max_attempts: int = 3


def utc_day(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------- ledger
@dataclass
class Entry:
    market_id: int
    slug: str
    outcome: str
    ts: float
    ask: float
    limit: float
    shares: float          # gross shares expected from the book walk
    cost: float            # USDT expected to be spent
    end_ts: float
    start_price: float | None = None
    order_ok: bool = True
    msg: str = ""
    shares_seen: float = 0.0   # net shares observed in positions
    base: float = 0.0          # shares of this outcome already held before the order
    filled: bool = True
    settled: bool = False
    won: bool | None = None
    pnl: float = 0.0


@dataclass
class Ledger:
    path: Path
    entries: dict[str, Entry] = field(default_factory=dict)     # filled trades only
    misses: dict[str, Entry] = field(default_factory=dict)      # orders that did not fill (for analysis)

    @classmethod
    def load(cls, path: Path) -> "Ledger":
        led = cls(path)
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            led.entries = {k: Entry(**v) for k, v in raw.get("entries", {}).items()}
            led.misses = {k: Entry(**v) for k, v in raw.get("misses", {}).items()}
        return led

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"entries": {k: asdict(v) for k, v in self.entries.items()},
                                   "misses": {k: asdict(v) for k, v in self.misses.items()}}, ensure_ascii=False),
                       encoding="utf-8")
        tmp.replace(self.path)

    def has(self, market_id: int) -> bool:
        return str(market_id) in self.entries

    def add(self, e: Entry) -> None:
        self.entries[str(e.market_id)] = e
        self.save()

    def add_miss(self, e: Entry) -> None:
        self.misses[f"{e.market_id}:{len(self.misses)}"] = e
        self.save()

    def miss_stats(self) -> dict[str, int]:
        d = [e for e in self.misses.values() if e.settled and e.won is not None]
        return {"n": len(self.misses), "settled": len(d), "would_win": sum(1 for e in d if e.won)}

    def done(self) -> list[Entry]:
        return [e for e in self.entries.values() if e.order_ok and e.settled]

    def open(self) -> list[Entry]:
        return [e for e in self.entries.values() if e.order_ok and not e.settled]

    def stats(self, day: str | None = None) -> dict[str, float]:
        d = [e for e in self.done() if day is None or utc_day(e.ts) == day]
        cost = sum(e.cost for e in d)
        pnl = sum(e.pnl for e in d)
        return {"n": len(d), "wins": sum(1 for e in d if e.won), "pnl": pnl, "cost": cost,
                "roi": pnl / cost if cost else 0.0}


def risk_gate(cfg: BotCfg, led: Ledger, now: float, balance: float | None, stop_file: Path,
              pending_cost: float = 0.0) -> str:
    """'' if a new entry is allowed, else the reason."""
    if stop_file.exists():
        return f"пауза: есть файл {stop_file.name}"
    open_cost = sum(e.cost for e in led.open()) + pending_cost
    total = led.stats()
    if total["pnl"] - open_cost - cfg.stake < -cfg.total_loss:
        return f"стоп: общий убыток достиг лимита {cfg.total_loss:.0f}$"
    today = led.stats(utc_day(now))
    if today["pnl"] - open_cost - cfg.stake < -cfg.daily_loss:
        return f"пауза до конца суток UTC: дневной лимит убытка {cfg.daily_loss:.0f}$"
    for n, min_roi in cfg.checkpoints:
        if total["n"] >= n and total["roi"] < min_roi:
            return (f"стоп по контрольной точке {n} сделок: ROI {total['roi']:+.1%} < {min_roi:+.0%} — "
                    f"край не подтвердился")
    if cfg.live and balance is not None and balance < max(cfg.min_balance, cfg.stake):
        return f"мало USDT на счёте ({balance:.2f}$)"
    return ""


def signal(cfg: BotCfg, m: MarketInfo, book: Any, now: float) -> tuple[str, float] | None:
    """(outcome, ask) if exactly one side's best ask is inside [lo, hi] in the entry window."""
    t = now - m.start_ms / 1000
    if not (cfg.t_from <= t <= cfg.t_to) or m.trading_status not in ("", "OPEN"):
        return None
    hits = []
    for o in ("UP", "DOWN"):
        ob = book.outcome(o)
        a, b = ob.best_ask, ob.best_bid
        if a is None or b is None or a - b > cfg.max_spread + 1e-9:
            continue
        if cfg.lo - 1e-9 <= a <= cfg.hi + 1e-9:
            hits.append((o, a))
    return hits[0] if len(hits) == 1 else None


def settle_pnl(cfg: BotCfg, e: Entry, won: bool) -> float:
    cut = cfg.fee_rate * min(e.ask, 1 - e.ask) / e.ask if e.ask > 0 else 0.0
    net = e.shares_seen if e.shares_seen > 0 else e.shares * (1 - cut)
    return net * (1.0 if won else 0.0) - e.cost


# ---------------------------------------------------------------------------- bot
class FavBot:
    def __init__(self, venue: Any, cfg: BotCfg, root: Path) -> None:
        self.v = venue
        self.cfg = cfg
        mode = "live" if cfg.live else "paper"
        self.led = Ledger.load(root / "logs" / f"favbot_{mode}.json")
        self.csv = root / "logs" / f"favbot_{mode}.csv"
        self.stop_file = root / "logs" / "favbot.STOP"
        self.attempts: dict[int, int] = {}
        self.pending: dict[int, list] = {}      # market_id -> [entry, sent_ts, no_fill_event]
        self.win: dict[int, dict[str, Any]] = {}  # market_id -> window state: outcome, base, unit, template, blocked
        self.last_gate = ""
        self.redeemed = 0
        self.redeem_fail = 0
        self._builder: Any = None
        self._redeem_tried: dict[str, float] = {}

    def _csv(self, row: dict[str, Any]) -> None:
        new = not self.csv.exists()
        self.csv.parent.mkdir(parents=True, exist_ok=True)
        with self.csv.open("a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)

    # -- entries
    def _pos_shares(self, market_id: int, outcome: str) -> float:
        return sum(p.shares for p in self.v.account.positions
                   if p.market_id == market_id and p.outcome == outcome and not p.resolved)

    NO_FILL = ("orderNotAccepted", "orderCancelled", "orderExpired", "orderTransactionFailed")

    def on_wallet(self, ev: dict[str, Any]) -> None:
        """predict.fun wallet events: a definitive 'not filled' lets the bot retry safely."""
        kind = str(ev.get("type") or "")
        if kind not in self.NO_FILL:
            return
        d = ev.get("details") or {}
        name = str(d.get("outcomeName") or d.get("outcome") or "").upper()
        for p in self.pending.values():
            if not name or p[0].outcome == name:
                p[2] = True

    def _cut(self, ask: float) -> float:
        return self.cfg.fee_rate * min(ask, 1 - ask) / ask if ask > 0 else 0.0

    def _sync_windows(self, now: float) -> None:
        """Positions are the truth: every share bought in a window (incl. late fills of 'missed' orders) is booked
        into one ledger entry per window, cost scaled from the planned price."""
        for mid, w in list(self.win.items()):
            t = w["tpl"]
            if now > t.end_ts + 300:
                del self.win[mid]
                continue
            got = self._pos_shares(mid, w["outcome"]) - w["base"]
            e = self.led.entries.get(str(mid))
            have = e.shares_seen if e else 0.0
            if got <= have + 1e-6 or (e is not None and e.settled):
                continue
            cost = got / (1 - self._cut(t.ask)) * w["unit"]
            if e is None:
                e = Entry(mid, t.slug, w["outcome"], t.ts, t.ask, t.limit, got / (1 - self._cut(t.ask)), cost,
                          t.end_ts, t.start_price, base=w["base"])
                e.shares_seen = got
                late = [k for k, x in self.led.misses.items() if x.market_id == mid]
                for k in late:
                    del self.led.misses[k]
                self.led.add(e)
                log.info("ВХОД %s %s исполнен%s: %.4f шт (ask был %.2f), ≈%.2f$", t.slug, w["outcome"],
                         " с опозданием" if late else "", got, t.ask, cost)
                self._row("entry", e)
            else:
                log.warning("ДОКУПЛЕНО %s %s: позиция выросла %.4f → %.4f шт (позднее исполнение), стоимость ≈%.2f$",
                            t.slug, w["outcome"], have, got, cost)
                e.shares_seen, e.cost = got, cost
                self.led.save()

    def _confirm(self, now: float) -> None:
        self._sync_windows(now)
        for mid, p in list(self.pending.items()):
            e, sent, no_fill = p
            if self.led.has(mid):
                del self.pending[mid]
            elif no_fill:
                del self.pending[mid]
                e.filled = False
                self.led.add_miss(e)
                log.info("НЕ ИСПОЛНЕН %s %s (ask %.2f, лимит %.2f): биржа не нашла встречных заявок", e.slug,
                         e.outcome, e.ask, e.limit)
                self._row("nofill", e)
            elif now - sent > self.cfg.fill_wait:
                del self.pending[mid]
                self.win[mid]["blocked"] = True
                e.filled = False
                e.msg = "нет подтверждения"
                self.led.add_miss(e)
                log.warning("НЕТ ПОДТВЕРЖДЕНИЯ %s %s за %.0f с — в этом окне больше не покупаю, слежу за позицией",
                            e.slug, e.outcome, self.cfg.fill_wait)
                self._row("unknown", e)

    def _row(self, event: str, e: Entry) -> None:
        self._csv({"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "event": event,
                   "market_id": e.market_id, "slug": e.slug, "outcome": e.outcome, "ask": e.ask, "limit": e.limit,
                   "shares": round(e.shares_seen or e.shares, 4), "usdt": round(e.cost, 4),
                   "won": "" if e.won is None else e.won, "pnl": round(e.pnl, 4) if e.settled else "",
                   "msg": e.msg})

    async def tick(self, now: float | None = None) -> None:
        now = now or time.time()
        if self.cfg.live:
            self._confirm(now)
        for m in list(self.v.markets.values()):
            if m.series != self.cfg.series or self.led.has(m.market_id) or m.market_id in self.pending:
                continue
            w = self.win.get(m.market_id)
            if w is not None and w["blocked"]:
                continue
            book = self.v.books.get(m.market_id)
            if book is None:
                continue
            sig = signal(self.cfg, m, book, now)
            if sig is None:
                continue
            gate = risk_gate(self.cfg, self.led, now, self.v.account.balance, self.stop_file,
                             sum(p[0].cost for p in self.pending.values()))
            if gate:
                if gate != self.last_gate:
                    log.warning(gate)
                    self.last_gate = gate
                return
            self.last_gate = ""
            await self.enter(m, sig[0], sig[1], book, now)

    async def enter(self, m: MarketInfo, outcome: str, ask: float, book: Any, now: float) -> None:
        tick = 10 ** -m.precision
        cap = self.cfg.max_price if self.cfg.max_price is not None else self.cfg.hi
        limit = round(min(ask + self.cfg.extra_ticks * tick, cap), m.precision)
        fill = take_buy(book.outcome(outcome).asks, limit, budget_usdt=self.cfg.stake)
        if fill.usdt < MIN_ORDER_USDT:
            return
        e = Entry(m.market_id, m.slug, outcome, now, ask, limit, fill.shares, fill.usdt, m.end_ms / 1000,
                  m.start_price)
        if not self.cfg.live:
            self.led.add(e)
            log.info("ВХОД %s %s ask %.2f лимит %.2f ≈%.2f шт за %.2f$ (бумага)", m.slug, outcome, ask, limit,
                     fill.shares, fill.usdt)
            self._row("entry", e)
            return
        n = self.attempts.get(m.market_id, 0)
        if n >= self.cfg.max_attempts:
            return
        w = self.win.get(m.market_id)
        if w is not None and w["outcome"] != outcome:
            return                                   # never buy both sides of one window
        self.attempts[m.market_id] = n + 1
        if w is None:
            w = {"outcome": outcome, "base": self._pos_shares(m.market_id, outcome),
                 "unit": fill.usdt / fill.shares, "tpl": e, "blocked": False}
            self.win[m.market_id] = w
        e.base = w["base"]
        try:
            res = await self.v.buy(m.market_id, outcome, limit, self.cfg.stake)
            ok, e.msg = bool(res.get("ok")), str(res.get("msg") or "")[:200]
        except OrderRejected as exc:
            ok, e.msg = False, str(exc)
        except Exception as exc:  # noqa: BLE001
            ok, e.msg = False, f"{type(exc).__name__}: {exc}"
        if not ok:
            log.warning("ордер отклонён (попытка %d/%d): %s", n + 1, self.cfg.max_attempts, e.msg)
            self._row("reject", e)
            return
        self.pending[m.market_id] = [e, now, False]
        log.info("ОРДЕР %s %s ask %.2f лимит %.2f ≈%.2f шт за %.2f$ — жду исполнения", m.slug, outcome, ask, limit,
                 fill.shares, fill.usdt)

    # -- settlement
    def _observe_positions(self) -> None:
        if self.cfg.live:
            self._sync_windows(time.time())

    async def _winner(self, e: Entry) -> bool | None:
        for p in self.v.account.positions:
            if p.market_id == e.market_id and p.outcome == e.outcome and p.resolved in ("WON", "LOST"):
                return p.resolved == "WON"
        m = self.v.markets.get(e.market_id)
        if m is None:
            try:
                raw = await self.v.rest.category(e.slug)
            except Exception:  # noqa: BLE001
                raw = None
            if raw:
                series = next((s for s in self.v.s.series if s.prefix == self.cfg.series), None)
                m = parse_category(raw, series) if series else None
        if m is None or m.end_price is None or m.start_price is None:
            return None
        if m.end_price == m.start_price:
            return None          # Flat: wait for the official status
        up = m.end_price > m.start_price
        return up if e.outcome == "UP" else not up

    async def settle(self, now: float | None = None) -> None:
        now = now or time.time()
        self._observe_positions()
        for e in self.led.open():
            if now < e.end_ts + 15:
                continue
            won = await self._winner(e)
            if won is None:
                if now > e.end_ts + 3600:
                    log.warning("рынок %s: исход не получен за час, считаю по позиции позже", e.slug)
                continue
            e.won, e.settled = won, True
            e.pnl = settle_pnl(self.cfg, e, won)
            self.led.save()
            st = self.led.stats()
            log.info("ИТОГ %s %s: %s %+.2f$ | всего %d сделок, %d побед, PnL %+.2f$, ROI %+.1f%%", e.slug, e.outcome,
                     "выиграл" if won else "проиграл", e.pnl, st["n"], st["wins"], st["pnl"], st["roi"] * 100)
            self._row("settle", e)
        for e in self.led.misses.values():         # what the missed orders would have done
            if e.settled or now < e.end_ts + 15:
                continue
            won = await self._winner(e)
            if won is not None:
                e.won, e.settled = won, True
                e.pnl = settle_pnl(self.cfg, e, won)
                self.led.save()

    # -- redemption (on-chain, BNB gas on the signing wallet)
    def _redeem_builder(self) -> Any:
        if self._builder is None:
            from predict_sdk import ChainId, OrderBuilder, OrderBuilderOptions  # type: ignore
            s = self.v.s
            opts = OrderBuilderOptions(predict_account=s.predict_account) if s.predict_account else None
            self._builder = (OrderBuilder.make(ChainId.BNB_MAINNET, s.private_key, opts) if opts
                             else OrderBuilder.make(ChainId.BNB_MAINNET, s.private_key))
        return self._builder

    async def redeem(self) -> None:
        if not self.cfg.live or not self.cfg.redeem:
            return
        try:
            raw = await self.v.rest.positions()
        except Exception as exc:  # noqa: BLE001
            log.warning("выкуп: позиции недоступны: %s", exc)
            return
        now = time.time()
        for p in raw:
            m, o = p.get("market") or {}, p.get("outcome") or {}
            amount = int(str(p.get("amount") or 0)) if str(p.get("amount") or "0").isdigit() else 0
            cid = str(m.get("conditionId") or "")
            if amount <= 0 or not cid or m.get("status") != "RESOLVED" or o.get("status") != "WON":
                continue
            if now - self._redeem_tried.get(cid, 0.0) < 900:
                continue
            self._redeem_tried[cid] = now
            neg = bool(m.get("isNegRisk"))
            try:
                b = await asyncio.to_thread(self._redeem_builder)
                res = await asyncio.to_thread(b.redeem_positions, cid, int(o.get("indexSet") or 1),
                                              amount if neg else None, is_neg_risk=neg,
                                              is_yield_bearing=bool(m.get("isYieldBearing")))
                ok = bool(getattr(res, "success", False))
                cause = "" if ok else str(getattr(res, "cause", ""))[:200]
            except Exception as exc:  # noqa: BLE001
                ok, cause = False, f"{type(exc).__name__}: {exc}"[:200]
            if ok:
                self.redeemed += 1
            else:
                self.redeem_fail += 1
            log.info("ВЫКУП рынок %s %s %.4f шт -> %s %s", m.get("id"), o.get("name"), amount / 1e18,
                     "OK" if ok else "ОШИБКА", cause + (" (нужен BNB на газ у кошелька-подписанта?)"
                                                        if cause and "fund" in cause.lower() else ""))

    def status_line(self) -> str:
        st, today = self.led.stats(), self.led.stats(utc_day(time.time()))
        bal = self.v.account.balance
        parts = [f"[{'LIVE' if self.cfg.live else 'PAPER'}] сделок {st['n']} (побед {st['wins']})",
                 f"PnL {st['pnl']:+.2f}$", f"ROI {st['roi'] * 100:+.1f}%",
                 f"сегодня {today['n']} / {today['pnl']:+.2f}$", f"открыто {len(self.led.open())}"]
        if bal is not None:
            parts.append(f"баланс {bal:.2f}$")
        if self.cfg.live:
            ms = self.led.miss_stats()
            parts.append(f"неисполнено {ms['n']} (из итоговых {ms['settled']} выиграли бы {ms['would_win']})")
            if self.cfg.redeem:
                parts.append(f"выкуплено {self.redeemed}, ошибок выкупа {self.redeem_fail}")
        return " | ".join(parts)

    async def run(self) -> None:
        last_status = last_redeem = 0.0
        while True:
            try:
                await self.tick()
                await self.settle()
                now = time.time()
                if now - last_redeem > self.cfg.redeem_every:
                    last_redeem = now
                    await self.redeem()
                if now - last_status > 60:
                    last_status = now
                    log.info(self.status_line())
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.exception("ошибка цикла: %s", exc)
            await asyncio.sleep(0.25)
