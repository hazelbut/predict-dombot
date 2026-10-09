"""Settings from .env (KEY=VALUE) with real environment overriding the file."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def load_dotenv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, _, v = s.partition("=")
        k, v = k.strip(), v.strip()
        if k.startswith("export "):
            k = k[7:].strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
            v = v[1:-1]
        else:
            i = v.find(" #")
            if i >= 0:
                v = v[:i].rstrip()
        out[k] = v
    return out


@dataclass(frozen=True)
class Series:
    """A family of recurring markets, e.g. btc-updown-5m with a 300 s period."""

    prefix: str
    period_s: int
    label: str


def parse_series(spec: str) -> list[Series]:
    out: list[Series] = []
    for part in (p.strip() for p in spec.split(",")):
        if not part:
            continue
        prefix, _, period = part.partition(":")
        period_s = int(period) if period else 300
        label = prefix.replace("-updown-", " ").upper()
        out.append(Series(prefix.strip(), period_s, label))
    return out


@dataclass(frozen=True)
class Settings:
    api_key: str = ""
    predict_account: str = ""        # deposit address of an email / Privy "Predict account" (empty = EOA mode)
    private_key: str = ""            # Privy exported key (Predict account) or the EOA key
    api_url: str = "https://api.predict.fun"
    ws_url: str = "wss://ws.predict.fun/ws"
    binance_ws_url: str = "wss://stream.binance.com:9443/stream?streams=btcusdt@bookTicker/btcusdt@kline_1s/btcusdt@aggTrade"
    binance_depth_ws_url: str = "wss://stream.binance.com:9443/ws/btcusdt@depth@100ms"
    depth_range_usd: float = 1000.0          # order profile: buckets within ±this of the mid
    binance_rest_url: str = "https://api.binance.com"
    host: str = "127.0.0.1"
    port: int = 8765
    series: list[Series] = field(default_factory=lambda: parse_series("btc-updown-5m:300,btc-updown-15m:900"))
    max_order_usdt: float = 50.0     # hard cap per click
    default_size_usdt: float = 5.0
    size_presets: str = "2,5,10,25,50"
    slippage_ticks: int = 0          # extra ticks allowed past the clicked price for taker orders
    chart_minutes: int = 20
    demo: bool = False

    @property
    def is_predict_account(self) -> bool:
        return bool(self.predict_account)

    @property
    def can_trade(self) -> bool:
        return self.demo or bool(self.api_key and self.private_key)


def load_settings(env_file: str = ".env", demo: bool = False, port: int | None = None,
                  require_wallet: bool = True) -> Settings:
    merged = load_dotenv(Path(env_file))
    merged.update({k: v for k, v in os.environ.items() if k.startswith(("PREDICT_", "TERMINAL_", "BINANCE_"))})

    def g(key: str, default: str) -> str:
        return merged.get(key, default) or default

    s = Settings(
        api_key=merged.get("PREDICT_API_KEY", ""),
        predict_account=merged.get("PREDICT_ACCOUNT", "").strip(),
        private_key=merged.get("PREDICT_PRIVATE_KEY", "").strip(),
        api_url=g("PREDICT_API_URL", "https://api.predict.fun").rstrip("/"),
        ws_url=g("PREDICT_WS_URL", "wss://ws.predict.fun/ws"),
        binance_ws_url=g("BINANCE_WS_URL", Settings.binance_ws_url),
        binance_rest_url=g("BINANCE_REST_URL", "https://api.binance.com").rstrip("/"),
        host=g("TERMINAL_HOST", "127.0.0.1"),
        port=port or int(g("TERMINAL_PORT", "8765")),
        series=parse_series(g("TERMINAL_MARKETS", "btc-updown-5m:300,btc-updown-15m:900")),
        max_order_usdt=float(g("TERMINAL_MAX_ORDER_USDT", "50")),
        default_size_usdt=float(g("TERMINAL_DEFAULT_SIZE_USDT", "5")),
        size_presets=g("TERMINAL_SIZE_PRESETS", "2,5,10,25,50"),
        slippage_ticks=int(g("TERMINAL_SLIPPAGE_TICKS", "0")),
        chart_minutes=int(g("TERMINAL_CHART_MINUTES", "20")),
        binance_depth_ws_url=g("BINANCE_DEPTH_WS_URL", Settings.binance_depth_ws_url),
        depth_range_usd=float(g("TERMINAL_DEPTH_RANGE_USD", "1000")),
        demo=demo,
    )
    if not s.demo:
        need = [("PREDICT_API_KEY", s.api_key)] + ([("PREDICT_PRIVATE_KEY", s.private_key)] if require_wallet else [])
        missing = [k for k, v in need if not v]
        if missing:
            raise SystemExit(f"missing in .env: {', '.join(missing)}  (or start with --demo)")
    return s
