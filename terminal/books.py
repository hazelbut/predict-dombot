"""Order book math. The venue publishes one book per market quoted in the first
(Up / Yes) outcome; the Down book is its complement with sides swapped.

Levels are (price, shares) tuples; asks ascending, bids descending.
"""

from __future__ import annotations

from dataclasses import dataclass

Level = tuple[float, float]


def rnd(price: float, precision: int) -> float:
    return round(price + 0.0, precision)


@dataclass(slots=True)
class OutcomeBook:
    bids: list[Level]
    asks: list[Level]

    @property
    def best_bid(self) -> float | None:
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return self.asks[0][0] if self.asks else None


def normalize(levels: list, descending: bool) -> list[Level]:
    out: list[Level] = []
    for lv in levels or ():
        try:
            p, q = float(lv[0]), float(lv[1])
        except (TypeError, ValueError, IndexError):
            continue
        if p > 0 and q > 0:
            out.append((p, q))
    out.sort(key=lambda x: x[0], reverse=descending)
    return out


def split_book(yes_bids: list, yes_asks: list, precision: int) -> tuple[OutcomeBook, OutcomeBook]:
    """(up_book, down_book) from the raw Yes-quoted book."""
    ub = normalize(yes_bids, True)
    ua = normalize(yes_asks, False)
    down_asks = sorted(((rnd(1 - p, precision), q) for p, q in ub), key=lambda x: x[0])
    down_bids = sorted(((rnd(1 - p, precision), q) for p, q in ua), key=lambda x: x[0], reverse=True)
    return OutcomeBook(ub, ua), OutcomeBook(down_bids, down_asks)


@dataclass(slots=True)
class Fill:
    shares: float
    usdt: float
    worst: float
    levels: list[Level]

    @property
    def avg(self) -> float:
        return self.usdt / self.shares if self.shares > 0 else 0.0


EPS = 1e-12


def take_buy(asks: list[Level], limit_price: float, budget_usdt: float | None = None,
             max_shares: float | None = None) -> Fill:
    """Walk asks up to limit_price, spending at most budget_usdt / buying at most max_shares."""
    shares = cost = 0.0
    worst = 0.0
    used: list[Level] = []
    for p, q in asks:
        if p > limit_price + EPS:
            break
        take = q
        if max_shares is not None:
            take = min(take, max_shares - shares)
        if budget_usdt is not None:
            take = min(take, (budget_usdt - cost) / p)
        if take <= EPS:
            break
        shares += take
        cost += take * p
        worst = p
        used.append((p, take))
    return Fill(shares, cost, worst, used)


def take_sell(bids: list[Level], limit_price: float, max_shares: float) -> Fill:
    """Walk bids down to limit_price selling at most max_shares."""
    shares = proceeds = 0.0
    worst = 0.0
    used: list[Level] = []
    for p, q in bids:
        if p < limit_price - EPS:
            break
        take = min(q, max_shares - shares)
        if take <= EPS:
            break
        shares += take
        proceeds += take * p
        worst = p
        used.append((p, take))
    return Fill(shares, proceeds, worst, used)
