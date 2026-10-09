"""Recurring BTC Up/Down markets. Slugs are deterministic:
``btc-updown-5m-<unix start>``, ``btc-updown-15m-<unix start>``."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from .config import Series


def iso_ms(text: Any) -> int:
    if not text:
        return 0
    try:
        return int(datetime.fromisoformat(str(text).replace("Z", "+00:00")).timestamp() * 1000)
    except ValueError:
        return 0


def fnum(v: Any) -> float | None:
    try:
        return None if v is None or v == "" else float(v)
    except (TypeError, ValueError):
        return None


@dataclass(slots=True)
class Outcome:
    name: str
    token_id: str
    index_set: int


@dataclass(slots=True)
class MarketInfo:
    series: str              # series prefix, e.g. btc-updown-5m
    label: str
    slug: str
    title: str
    category_id: int
    market_id: int
    start_ms: int
    end_ms: int
    start_price: float | None
    end_price: float | None
    trading_status: str
    status: str
    precision: int
    fee_bps: int
    is_neg_risk: bool
    is_yield_bearing: bool
    up: Outcome
    down: Outcome

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["up"] = asdict(self.up)
        d["down"] = asdict(self.down)
        return d

    def outcome(self, name: str) -> Outcome:
        return self.up if name.upper() == "UP" else self.down


def slug_for(series: Series, start_s: int) -> str:
    return f"{series.prefix}-{start_s}"


def window_start(series: Series, now_s: float) -> int:
    return int(now_s // series.period_s) * series.period_s


def parse_category(raw: dict[str, Any], series: Series) -> MarketInfo | None:
    markets = raw.get("markets") or []
    if not markets:
        return None
    m = markets[0]
    outs = m.get("outcomes") or []
    if len(outs) < 2:
        return None
    by_index = {int(o.get("indexSet") or 0): o for o in outs}
    o1 = by_index.get(1) or outs[0]
    o2 = by_index.get(2) or outs[1]
    # index 1 is the Yes/Up side the book is quoted in; names are checked for safety
    if str(o1.get("name", "")).lower().startswith("down") and str(o2.get("name", "")).lower().startswith("up"):
        o1, o2 = o2, o1
    vd = m.get("variantData") or raw.get("variantData") or {}
    return MarketInfo(
        series=series.prefix,
        label=series.label,
        slug=str(raw.get("slug") or m.get("categorySlug") or ""),
        title=str(raw.get("title") or m.get("title") or ""),
        category_id=int(raw.get("id") or 0),
        market_id=int(m.get("id") or 0),
        start_ms=iso_ms(raw.get("startsAt")),
        end_ms=iso_ms(raw.get("endsAt")),
        start_price=fnum(vd.get("startPrice")),
        end_price=fnum(vd.get("endPrice")),
        trading_status=str(m.get("tradingStatus") or ""),
        status=str(m.get("status") or raw.get("status") or ""),
        precision=int(m.get("decimalPrecision") or 2),
        fee_bps=int(m.get("feeRateBps") or 200),
        is_neg_risk=bool(m.get("isNegRisk", raw.get("isNegRisk", False))),
        is_yield_bearing=bool(m.get("isYieldBearing", raw.get("isYieldBearing", False))),
        up=Outcome(str(o1.get("name") or "Up"), str(o1.get("onChainId") or ""), int(o1.get("indexSet") or 1)),
        down=Outcome(str(o2.get("name") or "Down"), str(o2.get("onChainId") or ""), int(o2.get("indexSet") or 2)),
    )
