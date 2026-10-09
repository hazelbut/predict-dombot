"""Favourite bot for BTC Up/Down 5m on predict.fun (live test of the backtest).

    python run_favbot.py                 # PAPER: signals and results on real books, no orders
    python run_favbot.py --live          # real orders, stake 1.1 USDT, risk caps below

Default rule: between 30 and 150 s after the window start, the first time one side's best ask is in
0.60-0.70 -> buy it (cap ask + 1 tick), hold to expiry, one entry per window.
Logs: logs/favbot.log, logs/favbot_live.csv (or _paper), state logs/favbot_live.json.
Pause: create logs/favbot.STOP. Reset the ledger: delete logs/favbot_live.json.
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

from terminal.config import load_settings, parse_series  # noqa: E402
from terminal.favbot import BotCfg, FavBot  # noqa: E402
from terminal.venue import Journal  # noqa: E402
from terminal.venue_live import BSC_RPC, LiveVenue  # noqa: E402


async def bnb_balance(venue: LiveVenue, address: str) -> float | None:
    try:
        body = {"jsonrpc": "2.0", "id": 1, "method": "eth_getBalance", "params": [address, "latest"]}
        assert venue._rpc is not None
        async with venue._rpc.post(BSC_RPC, json=body) as r:
            res = await r.json(content_type=None)
        return int(res["result"], 16) / 1e18
    except Exception:  # noqa: BLE001
        return None


def audit_positions(raw: list[dict]) -> dict[str, float]:
    """Split predict.fun positions into redeemable winnings, losses and still-open positions."""
    out = {"won_n": 0, "won_usd": 0.0, "lost_n": 0, "lost_cost": 0.0, "open_n": 0, "open_cost": 0.0, "max_sh": 0.0}
    for p in raw:
        try:
            sh = int(str(p.get("amount") or 0)) / 1e18
        except ValueError:
            sh = 0.0
        if sh <= 1e-9:
            continue
        m, o = p.get("market") or {}, p.get("outcome") or {}
        avg = float(p.get("averageBuyPriceUsd") or 0)
        st = str(o.get("status") or "")
        if st == "WON":
            out["won_n"] += 1
            out["won_usd"] += sh
        elif st == "LOST":
            out["lost_n"] += 1
            out["lost_cost"] += sh * avg
        else:
            out["open_n"] += 1
            out["open_cost"] += sh * avg
        out["max_sh"] = max(out["max_sh"], sh)
    return out


async def audit(a: argparse.Namespace) -> int:
    from terminal.favbot import Ledger
    s = load_settings(str(ROOT / ".env"))
    venue = LiveVenue(s, lambda msg: None)
    await venue.rest.start()
    import aiohttp
    venue._rpc = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=8))
    try:
        bal = await venue._usdt_balance()
        raw = await venue.rest.positions()
    finally:
        await venue.rest.close()
        await venue._rpc.close()
    r = audit_positions(raw)
    equity = (bal or 0.0) + r["won_usd"] + r["open_cost"]
    led = Ledger.load(ROOT / "logs" / "favbot_live.json")
    st = led.stats()
    print(f"USDT на счёте:                 {bal:.2f}$" if bal is not None else "USDT на счёте: ?")
    print(f"выигрыши ещё не зачислены:     {r['won_n']} позиций, {r['won_usd']:.2f}$ (обычно 0: зачисляются за минуту)")
    print(f"открытые позиции (по цене):    {r['open_n']} шт, {r['open_cost']:.2f}$")
    print(f"проигравшие (ещё в списке):    {r['lost_n']}")
    print(f"ИТОГО капитал:                 {equity:.2f}$")
    if a.start:
        print(f"РЕАЛЬНЫЙ результат от старта {a.start:.2f}$: {equity - a.start:+.2f}$")
    print(f"по учёту бота:                 {st['n']} сделок, PnL {st['pnl']:+.2f}$, ROI {st['roi'] * 100:+.1f}%")
    print("(если на этом счёте ты ещё торговал руками — разница с учётом бота из-за этого)")
    return 0


async def amain(a: argparse.Namespace) -> int:
    s = load_settings(str(ROOT / ".env"))
    s = dataclasses.replace(s, series=parse_series("btc-updown-5m:300"))
    cfg = BotCfg(lo=a.lo, hi=a.hi, t_from=a.t_from, t_to=a.t_to, stake=a.stake, daily_loss=a.daily_loss,
                 total_loss=a.total_loss, live=a.live, extra_ticks=a.extra_ticks,
                 max_price=a.max_price, redeem=a.redeem)
    log = logging.getLogger("favbot")
    venue = LiveVenue(s, lambda msg: None)
    venue.journal = Journal(ROOT / "logs" / "favbot_orders.csv")
    await venue.start()
    for _ in range(40):
        if venue.account.balance is not None and venue.markets:
            break
        await asyncio.sleep(0.5)
    bnb = await bnb_balance(venue, venue.factory.eoa)
    log.info("режим %s | счёт %s | баланс USDT %s | BNB на газ у подписанта %s: %s",
             "LIVE (реальные ордера)" if cfg.live else "PAPER (без ордеров)", venue.address,
             f"{venue.account.balance:.2f}" if venue.account.balance is not None else "?", venue.factory.eoa,
             f"{bnb:.5f}" if bnb is not None else "?")
    log.info("потолок цены: аск + %d ц., но не выше %.2f", cfg.extra_ticks, cfg.max_price or cfg.hi)
    log.info("правило: %.2f-%.2f, %d-%d с от старта окна, ставка %.2f$, лимиты: день -%.0f$, всего -%.0f$, "
             "контроль: %s", cfg.lo, cfg.hi, cfg.t_from, cfg.t_to, cfg.stake, cfg.daily_loss, cfg.total_loss,
             ", ".join(f"{n} сделок ROI≥{r:+.0%}" for n, r in cfg.checkpoints))
    if cfg.live and cfg.redeem and bnb is not None and bnb < 0.0005:
        log.warning("мало BNB для выкупа выигрышей (%.5f). Пополни кошелёк-подписант %s на ~0.002 BNB, иначе выигрыши "
                    "останутся невыкупленными и USDT на счёте будет убывать", bnb, venue.factory.eoa)
    bot = FavBot(venue, cfg, ROOT)
    orig_wallet = venue._on_wallet_event

    def wallet_hook(ev):  # type: ignore[no-untyped-def]
        orig_wallet(ev)
        bot.on_wallet(ev)
    venue._on_wallet_event = wallet_hook
    task = asyncio.create_task(bot.run())
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass
    await stop.wait()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    log.info(bot.status_line())
    await venue.stop()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true", help="реальные ордера (без флага — бумажный режим)")
    ap.add_argument("--stake", type=float, default=1.1)
    ap.add_argument("--lo", type=float, default=0.60)
    ap.add_argument("--hi", type=float, default=0.70)
    ap.add_argument("--t-from", type=float, default=30)
    ap.add_argument("--t-to", type=float, default=150)
    ap.add_argument("--daily-loss", type=float, default=10.0)
    ap.add_argument("--total-loss", type=float, default=25.0)
    ap.add_argument("--extra-ticks", type=int, default=1, help="потолок цены = аск + N центов")
    ap.add_argument("--max-price", type=float, default=0.70, help="жёсткий потолок цены входа")
    ap.add_argument("--redeem", action="store_true", help="выкупать выигрыши самому (обычно не нужно)")
    ap.add_argument("--audit", action="store_true", help="сверка: реальный капитал по данным биржи")
    ap.add_argument("--start", type=float, default=0.0, help="капитал на старте теста, $ (для --audit)")
    a = ap.parse_args()
    os.chdir(ROOT)
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.FileHandler(ROOT / "logs" / "favbot.log", encoding="utf-8"),
                                  logging.StreamHandler(sys.stdout)])
    for noisy in ("venue", "predict_api", "aiohttp", "web3", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return asyncio.run(audit(a) if a.audit else amain(a))


if __name__ == "__main__":
    sys.exit(main())
