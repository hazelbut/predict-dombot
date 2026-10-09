"""DOM bot: cheap outcomes (0.10-0.30) of predict.fun BTC Up/Down, confirmed by the Binance BTCUSDT order book.

    python run_dombot.py                  # PAPER: real books, signals, candidates and results, no orders
    python run_dombot.py --live           # real orders (stake 1.1 USDT, risk caps below)
    python run_dombot.py --demo           # offline simulator (no keys, no network) - smoke test only
    python run_dombot.py --report         # calibration from logs/dombot_candidates.csv

Logs: logs/dombot.log, trades logs/dombot_<mode>.csv, state logs/dombot_<mode>.json,
every in-band situation logs/dombot_candidates.csv. Pause: create logs/dombot.STOP.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import logging
import os
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from terminal.btc import BinanceFeed, BtcState, DemoBtcFeed  # noqa: E402
from terminal.config import load_settings, parse_series  # noqa: E402
from terminal.depth import BinanceDepthFeed, DemoDepthFeed, SpotDepthBook  # noqa: E402
from terminal.dombot import DomBot, DomCfg, SigCfg, report  # noqa: E402
from terminal.venue import Journal  # noqa: E402

PERIOD = {"5m": 300, "15m": 900}


async def amain(a: argparse.Namespace) -> int:
    period = PERIOD[a.series]
    prefix = f"btc-updown-{a.series}"
    s = load_settings(str(ROOT / ".env"), demo=a.demo, require_wallet=a.live)
    s = dataclasses.replace(s, series=parse_series(f"{prefix}:{period}"))
    cfg = DomCfg(lo=a.lo, hi=a.hi, max_price=a.hi, stake=a.stake, daily_loss=a.daily_loss, total_loss=a.total_loss,
                 live=a.live, extra_ticks=a.extra_ticks, series=prefix, min_edge=a.min_edge,
                 require_signal=not a.no_signal, kinds=tuple(k.strip() for k in a.kinds.split(",") if k.strip()),
                 sig_ttl=a.sig_ttl, shift_bps=a.shift_bps, k=a.k,
                 tte_min=a.tte_min, tte_max=a.tte_max if a.tte_max is not None else period - 30)
    sig = SigCfg(wall_min_usd=a.wall_min_usd)
    log = logging.getLogger("dombot")

    btc = BtcState(65)                      # 60 min of 1 s prices for the volatility estimate
    depth = SpotDepthBook()
    if a.demo:
        from terminal.venue_demo import DemoVenue
        venue = DemoVenue(s, lambda msg: None, btc)
        btc_feed = DemoBtcFeed(btc)
        depth_feed = DemoDepthFeed(depth, lambda: btc.mid)
    else:
        from terminal.venue_live import DataVenue, LiveVenue
        venue = LiveVenue(s, lambda msg: None) if s.private_key else DataVenue(s, lambda msg: None)
        btc_feed = BinanceFeed(s.binance_ws_url, s.binance_rest_url, btc, 65)
        depth_feed = BinanceDepthFeed(s.binance_depth_ws_url, s.binance_rest_url, depth)
    venue.journal = Journal(ROOT / "logs" / "dombot_orders.csv")

    bot = DomBot(venue, cfg, ROOT, btc, depth, sig)
    orig_trade = btc.on_trade

    def trade_hook(ts_ms: int, price: float, qty: float, sell: bool) -> None:
        orig_trade(ts_ms, price, qty, sell)
        bot.on_btc_trade(ts_ms, price, qty, sell)
    btc.on_trade = trade_hook                       # type: ignore[method-assign]
    if not a.demo and s.private_key:
        orig_wallet = venue._on_wallet_event

        def wallet_hook(ev):  # type: ignore[no-untyped-def]
            orig_wallet(ev)
            bot.on_wallet(ev)
        venue._on_wallet_event = wallet_hook

    tasks = [asyncio.create_task(btc_feed.run()), asyncio.create_task(depth_feed.run())]
    await venue.start()
    for _ in range(40):
        if venue.markets and (a.demo or not s.private_key or venue.account.balance is not None):
            break
        await asyncio.sleep(0.5)
    log.info("режим %s | %s | счёт %s | баланс USDT %s", "DEMO" if a.demo else
             ("LIVE (реальные ордера)" if cfg.live else "PAPER (без ордеров)"), prefix, venue.address,
             f"{venue.account.balance:.2f}" if venue.account.balance is not None else "?")
    log.info("правило: аск %.2f-%.2f, до конца %d-%d с, сигналы %s за %d с%s, край ≥ %.3f, сдвиг %.1f б.п., "
             "ставка %.2f$, лимиты: день -%.0f$, всего -%.0f$", cfg.lo, cfg.hi, cfg.tte_min, cfg.tte_max,
             "/".join(cfg.kinds), cfg.sig_ttl, "" if cfg.require_signal else " (НЕ обязательны)", cfg.min_edge,
             cfg.shift_bps, cfg.stake, cfg.daily_loss, cfg.total_loss)
    if not a.demo:
        log.info("волатильности нужно ≥15 мин истории BTC: первые входы возможны после загрузки свечей")
    tasks.append(asyncio.create_task(bot.run()))
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sg in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sg, stop.set)
        except NotImplementedError:
            pass
    if a.run_for:
        loop.call_later(a.run_for, stop.set)
    await stop.wait()
    btc_feed.stop()
    depth_feed.stop()
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    log.info(bot.status_line())
    await venue.stop()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true", help="реальные ордера (без флага — бумажный режим)")
    ap.add_argument("--demo", action="store_true", help="офлайн-симулятор, без ключей и сети")
    ap.add_argument("--report", action="store_true", help="отчёт-калибровка по записанным кандидатам")
    ap.add_argument("--series", choices=sorted(PERIOD), default="5m")
    ap.add_argument("--lo", type=float, default=0.10)
    ap.add_argument("--hi", type=float, default=0.30, help="диапазон аска и жёсткий потолок цены входа")
    ap.add_argument("--stake", type=float, default=1.1)
    ap.add_argument("--min-edge", type=float, default=0.02, help="модель − цена с комиссией, не меньше")
    ap.add_argument("--no-signal", action="store_true", help="входить и без сигнала стакана (только модель)")
    ap.add_argument("--kinds", default="eaten,pulled,imbalance", help="какие сигналы стакана считать")
    ap.add_argument("--sig-ttl", type=float, default=20, help="сигнал свежий N секунд")
    ap.add_argument("--shift-bps", type=float, default=0.0,
                    help="сдвинуть цену BTC в сторону сигнала на N б.п. перед моделью (0 = без догадок)")
    ap.add_argument("--k", type=float, default=1.0, help="масштаб волатильности модели")
    ap.add_argument("--tte-min", type=float, default=30, help="не входить позже, чем за N с до конца")
    ap.add_argument("--tte-max", type=float, default=None, help="не входить раньше, чем за N с до конца")
    ap.add_argument("--wall-min-usd", type=float, default=150_000, help="минимальная плотность BTCUSDT, $")
    ap.add_argument("--daily-loss", type=float, default=10.0)
    ap.add_argument("--total-loss", type=float, default=25.0)
    ap.add_argument("--extra-ticks", type=int, default=1, help="потолок цены = аск + N центов (но ≤ --hi)")
    ap.add_argument("--tp-mult", type=float, default=2.0, help="для отчёта: какой тейк-профит проверить")
    ap.add_argument("--run-for", type=float, default=0, help=argparse.SUPPRESS)
    a = ap.parse_args()
    os.chdir(ROOT)
    (ROOT / "logs").mkdir(exist_ok=True)
    if a.report:
        print(report(ROOT, a.tp_mult))
        return 0
    if a.live and a.demo:
        ap.error("--live и --demo вместе нельзя")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.FileHandler(ROOT / "logs" / "dombot.log", encoding="utf-8"),
                                  logging.StreamHandler(sys.stdout)])
    for noisy in ("venue", "predict_api", "aiohttp", "web3", "urllib3", "btc", "depth"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return asyncio.run(amain(a))


if __name__ == "__main__":
    sys.exit(main())
