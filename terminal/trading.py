"""Order construction and signing with the official predict.fun SDK.

Taker click  = MARKET order sized from the live book up to the clicked price
               (fills what is there, the rest is not left on the book).
Limit click  = LIMIT order resting at the clicked price, expires at market end.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import ROUND_DOWN, Decimal
from typing import Any

from eth_account import Account
from eth_account.messages import encode_defunct
from predict_sdk import Book, BuildOrderInput, LimitHelperInput, MarketHelperInput, OrderBuilder, OrderBuilderOptions
from predict_sdk._internal.utils import generate_order_salt
from predict_sdk.constants import ADDRESSES_BY_CHAIN_ID, ChainId, Side
from predict_sdk.logger import Logger

from .books import Fill, OutcomeBook, take_buy, take_sell
from .markets import MarketInfo

WEI = Decimal(10) ** 18
MIN_ORDER_USDT = 0.9      # venue: "order must have a value of at least 0.9 USD"
MIN_SHARES = 0.01         # SDK: quantity_wei >= 1e16


class OrderRejected(Exception):
    """Local pre-trade check failed (nothing was sent)."""


def to_wei(x: float) -> int:
    return int((Decimal(repr(x)) * WEI).to_integral_value(rounding=ROUND_DOWN))


def hex0x(s: str) -> str:
    return s if s.startswith("0x") else "0x" + s


@dataclass(slots=True)
class BuiltOrder:
    data: dict[str, Any]        # body["data"] for POST /v1/orders
    order_hash: str
    preview: Fill | None
    side: str
    strategy: str
    shares: float
    price: float                # avg (taker) or limit price


class OrderFactory:
    def __init__(self, private_key: str, predict_account: str = "") -> None:
        acct = Account.from_key(private_key)
        self.eoa = acct.address
        self.predict_account = predict_account or None
        # built without RPC: signing needs none. Ownership of the Predict account is
        # verified separately (verify_owner) so a dead RPC never blocks trading.
        self.builder = OrderBuilder(
            chain_id=ChainId.BNB_MAINNET,
            precision=18,
            addresses=ADDRESSES_BY_CHAIN_ID[ChainId.BNB_MAINNET],
            generate_salt_fn=generate_order_salt,
            logger=Logger("WARN"),
            signer=acct,
            predict_account=self.predict_account,
        )
        self._acct = acct

    @property
    def address(self) -> str:
        """Maker address: Predict account (deposit address) or the EOA."""
        return self.predict_account or self.eoa

    def verify_owner(self) -> str:
        """Blocking RPC check (run in a thread). Returns '' if ok, else a message."""
        if not self.predict_account:
            return ""
        try:
            b = OrderBuilder.make(ChainId.BNB_MAINNET, self._acct,
                                  OrderBuilderOptions(predict_account=self.predict_account))
            del b
            return ""
        except Exception as exc:  # noqa: BLE001
            name = type(exc).__name__
            if name == "InvalidSignerError":
                return ("PREDICT_PRIVATE_KEY не является владельцем PREDICT_ACCOUNT: экспортируй ключ Privy "
                        "из настроек именно этого аккаунта")
            return f"не удалось проверить владельца аккаунта через BSC RPC ({name}); торговля всё равно доступна"

    def sign_auth_message(self, message: str) -> str:
        if self.predict_account:
            return hex0x(self.builder.sign_predict_account_message(message))
        signed = self._acct.sign_message(encode_defunct(text=message))
        return hex0x(signed.signature.hex())

    # ------------------------------------------------------------------
    def _finish(self, strategy: str, side: Side, m: MarketInfo, token_id: str, maker: int, taker: int,
                price_per_share: int, expires_at: datetime | None) -> tuple[dict[str, Any], str]:
        order = self.builder.build_order(strategy, BuildOrderInput(  # type: ignore[arg-type]
            side=side, token_id=token_id, maker_amount=str(maker), taker_amount=str(taker),
            fee_rate_bps=m.fee_bps, expires_at=expires_at,
            maker=self.predict_account, signer=self.predict_account,
        ))
        typed = self.builder.build_typed_data(order, is_neg_risk=m.is_neg_risk, is_yield_bearing=m.is_yield_bearing)
        order_hash = hex0x(self.builder.build_typed_data_hash(typed))
        signed = self.builder.sign_typed_data_order(typed)
        body = {
            "salt": str(signed.salt),
            "maker": signed.maker,
            "signer": signed.signer,
            "taker": signed.taker,
            "tokenId": str(signed.token_id),
            "makerAmount": str(signed.maker_amount),
            "takerAmount": str(signed.taker_amount),
            "expiration": int(signed.expiration),
            "nonce": str(signed.nonce),
            "feeRateBps": str(signed.fee_rate_bps),
            "side": int(signed.side),
            "signatureType": int(signed.signature_type),
            "hash": order_hash,
            "signature": hex0x(signed.signature),
        }
        return {"order": body, "pricePerShare": str(price_per_share), "strategy": strategy}, order_hash

    def market_buy(self, m: MarketInfo, outcome: str, book: OutcomeBook, limit_price: float, budget_usdt: float,
                   extra_ticks: int = 0) -> BuiltOrder:
        fill = take_buy(book.asks, limit_price, budget_usdt=budget_usdt)
        if fill.shares < MIN_SHARES:
            raise OrderRejected(f"в стакане нет предложений по цене ≤ {limit_price:.{m.precision}f}")
        if fill.usdt < MIN_ORDER_USDT:
            raise OrderRejected(f"объём {fill.usdt:.2f}$ меньше минимума {MIN_ORDER_USDT}$")
        tick = 10 ** -m.precision
        cap = min(limit_price + extra_ticks * tick, 0.999)
        slip = max(0, math.floor((cap / fill.worst - 1) * 10_000)) if fill.worst > 0 else 0
        sdk_book = Book(m.market_id, 0, asks=list(fill.levels), bids=[])
        amounts = self.builder.get_market_order_amounts(
            MarketHelperInput(side=Side.BUY, quantity_wei=to_wei(fill.shares), slippage_bps=slip), sdk_book)
        token = m.outcome(outcome).token_id
        data, h = self._finish("MARKET", Side.BUY, m, token, amounts.maker_amount, amounts.taker_amount,
                               amounts.price_per_share, None)
        return BuiltOrder(data, h, fill, "BUY", "MARKET", fill.shares, fill.avg)

    def market_sell(self, m: MarketInfo, outcome: str, book: OutcomeBook, limit_price: float, shares: float,
                    extra_ticks: int = 0) -> BuiltOrder:
        fill = take_sell(book.bids, limit_price, shares)
        if fill.shares < MIN_SHARES:
            raise OrderRejected(f"в стакане нет покупателей по цене ≥ {limit_price:.{m.precision}f}")
        if fill.usdt < MIN_ORDER_USDT:
            raise OrderRejected(f"объём {fill.usdt:.2f}$ меньше минимума {MIN_ORDER_USDT}$")
        tick = 10 ** -m.precision
        floor_px = max(limit_price - extra_ticks * tick, tick)
        slip = max(0, math.floor((1 - floor_px / fill.worst) * 10_000)) if fill.worst > 0 else 0
        sdk_book = Book(m.market_id, 0, asks=[], bids=list(fill.levels))
        amounts = self.builder.get_market_order_amounts(
            MarketHelperInput(side=Side.SELL, quantity_wei=to_wei(fill.shares), slippage_bps=slip), sdk_book)
        token = m.outcome(outcome).token_id
        data, h = self._finish("MARKET", Side.SELL, m, token, amounts.maker_amount, amounts.taker_amount,
                               amounts.price_per_share, None)
        return BuiltOrder(data, h, fill, "SELL", "MARKET", fill.shares, fill.avg)

    def limit(self, m: MarketInfo, outcome: str, side: str, price: float, shares: float,
              post_only: bool = False) -> BuiltOrder:
        tick = 10 ** -m.precision
        if not (tick - 1e-12 <= price <= 1 - tick + 1e-12):
            raise OrderRejected("цена вне диапазона")
        if shares < MIN_SHARES or shares * price < MIN_ORDER_USDT:
            raise OrderRejected(f"объём {shares * price:.2f}$ меньше минимума {MIN_ORDER_USDT}$")
        expires = datetime.fromtimestamp(m.end_ms / 1000, timezone.utc) if m.end_ms else None
        if expires is not None and expires.timestamp() <= datetime.now(timezone.utc).timestamp() + 1:
            raise OrderRejected("рынок уже закончился")
        s = Side.BUY if side.upper() == "BUY" else Side.SELL
        amounts = self.builder.get_limit_order_amounts(
            LimitHelperInput(side=s, price_per_share_wei=to_wei(round(price, m.precision)), quantity_wei=to_wei(shares)))
        token = m.outcome(outcome).token_id
        data, h = self._finish("LIMIT", s, m, token, amounts.maker_amount, amounts.taker_amount,
                               amounts.price_per_share, expires)
        if post_only:
            data["isPostOnly"] = True
        return BuiltOrder(data, h, None, s.name, "LIMIT", amounts.amount / 1e18, amounts.price_per_share / 1e18)
