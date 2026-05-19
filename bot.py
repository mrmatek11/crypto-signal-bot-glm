#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║          Crypto Signal Bot — NWO + Stoch(7,3,2) + CVD → Discord       ║
║                                                                        ║
║  Neural Weight Oscillator (Zeiierman) + Stochastic + CVD              ║
║  Monitoruje pary krypto i indeksy (SP500/US100) na żywo              ║
║  Wysyła alerty na Discord z AI news sentiment i position tracking     ║
║                                                                        ║
║  v2: + AI sentiment filter, + YFinance (SP500/US100),                 ║
║      + position tracking (SQLite), + trend filter alert mode           ║
║                                                                        ║
║  Użycie:                                                              ║
║    python bot.py --test --scan                                        ║
║    python bot.py --webhook URL --strategy nwo_stoch_cvd               ║
║    python bot.py --config aggressive --sentiment                      ║
║    python bot.py --market both --symbols BTC/USDT,SP500              ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import time
import signal
import argparse
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

# Dodaj ścieżkę projektu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import BotConfig, config_aggressive, config_conservative, config_scalping
from signal_detector import SignalDetector, Signal
from data_fetcher import DataFetcher
from discord_notifier import DiscordNotifier
from custom_strategy import STRATEGY_REGISTRY, get_sentiment_engine, set_sentiment_enabled

# ─── Logging ──────────────────────────────────────────────────────────────────

def setup_logging(level: str = "INFO"):
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=fmt, datefmt=datefmt)
    return logging.getLogger("StochBot")

logger = setup_logging()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN BOT CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class StochSignalBot:
    """
    Główna klasa bota sygnałowego.
    Pętla: pobierz dane → wykryj sygnały → wyślij na Discord → position tracking → czekaj.
    """

    def __init__(self, config: BotConfig, test_mode: bool = False):
        self.config = config
        self.test_mode = test_mode
        self._running = False
        self._scan_count = 0
        self._signals_sent = 0
        self._errors = 0
        self._cooldowns: Dict[str, float] = {}  # key -> timestamp ostatniego sygnału
        self._last_status_time = 0

        # Inicjalizuj detektor
        self.detector = SignalDetector(
            stoch_k_length=config.stoch_k_length,
            stoch_k_smooth=config.stoch_k_smooth,
            stoch_d_smooth=config.stoch_d_smooth,
            oversold_threshold=config.oversold_threshold,
            overbought_threshold=config.overbought_threshold,
            require_crossover=config.require_crossover,
            rsi_filter=config.rsi_filter,
            rsi_oversold=config.rsi_oversold,
            rsi_overbought=config.rsi_overbought,
            volume_filter=config.volume_filter,
            volume_mult=config.volume_mult,
        )

        # ─── Data fetcher (crypto / stocks / unified) ─────────────────
        self._yf_fetcher = None
        self._unified_fetcher = None

        if config.market_source in ("stocks", "both"):
            try:
                from data_fetcher_yfinance import YFinanceDataFetcher, UnifiedDataFetcher
                if config.market_source == "both":
                    self._unified_fetcher = UnifiedDataFetcher()
                    self.fetcher = self._unified_fetcher
                    logger.info("📦 Unified data fetcher: Crypto (Binance) + Stocks (YFinance)")
                else:
                    self._yf_fetcher = YFinanceDataFetcher()
                    self.fetcher = self._yf_fetcher
                    logger.info("📦 Stock data fetcher: YFinance (SPY, QQQ, etc.)")
            except ImportError:
                logger.warning("yfinance nie zainstalowany! Fallback do Binance. pip install yfinance")
                self.fetcher = DataFetcher(
                    exchange_id=config.exchange,
                    rate_limit_ms=config.rate_limit_ms,
                    cache_ttl_seconds=config.cache_ttl,
                    candles_per_fetch=config.candles_per_fetch,
                )
        else:
            self.fetcher = DataFetcher(
                exchange_id=config.exchange,
                rate_limit_ms=config.rate_limit_ms,
                cache_ttl_seconds=config.cache_ttl,
                candles_per_fetch=config.candles_per_fetch,
            )

        # ─── Discord notifier ─────────────────────────────────────────
        if not test_mode and config.discord_webhook_url:
            self.notifier = DiscordNotifier(
                webhook_url=config.discord_webhook_url,
                bot_name=config.discord_bot_name,
                avatar_url=config.discord_avatar_url,
                mention_role_id=config.discord_role_id,
                mention_on_long=config.mention_on_long,
                mention_on_short=config.mention_on_short,
                quiet_hours=config.quiet_hours,
            )
        else:
            self.notifier = None

        # ─── Position tracker ─────────────────────────────────────────
        self.position_tracker = None
        if config.use_position_tracking:
            try:
                from position_tracker import PositionTracker
                self.position_tracker = PositionTracker(
                    db_path=config.position_db_path,
                    timeout_hours=config.position_timeout_hours,
                    default_size_usd=config.default_position_size_usd,
                    max_open=config.max_open_positions,
                )
                logger.info(f"📊 Position tracker: ENABLED (db={config.position_db_path})")
            except Exception as e:
                logger.warning(f"Position tracker init failed: {e}")

        # ─── Sentiment engine ─────────────────────────────────────────
        self.sentiment_engine = None
        if config.use_sentiment:
            try:
                from news_sentiment import SentimentEngine
                self.sentiment_engine = SentimentEngine(
                    cryptopanic_key=config.cryptopanic_api_key,
                    finnhub_key=config.finnhub_api_key,
                    newsapi_key=config.newsapi_key,
                    refresh_interval=config.sentiment_refresh_interval,
                )
                logger.info(f"📰 AI Sentiment filter: ENABLED")
                logger.info(f"   CryptoPanic: {'✅' if config.cryptopanic_api_key else '❌'}")
                logger.info(f"   Finnhub: {'✅' if config.finnhub_api_key else '❌'}")
                logger.info(f"   RSS feeds: ✅ (always available)")
            except Exception as e:
                logger.warning(f"Sentiment engine init failed: {e}")

        # ─── Custom strategy ──────────────────────────────────────────
        self.custom_strategy_fn = None  # Domyślnie używa SignalDetector

        # ─── Apply trend filter mode to custom_strategy ────────────────
        if config.trend_filter_mode:
            try:
                import custom_strategy
                custom_strategy.TREND_FILTER_MODE = config.trend_filter_mode
                logger.info(f"⚠️ Trend filter mode: {config.trend_filter_mode}")
            except Exception:
                pass
        
        # ─── Apply sentiment enabled/disabled to custom_strategy ──────
        set_sentiment_enabled(config.use_sentiment)
        if config.use_sentiment:
            logger.info("📰 Sentiment filter in custom_strategy: ENABLED")
        else:
            logger.info("📰 Sentiment filter in custom_strategy: DISABLED (signals pass through)")
        
        # ─── AI Analyst (GLM) ────────────────────────────────────────
        self.ai_analyst = None
        if config.glm_api_key:
            try:
                from ai_analyst import AIAnalyst
                self.ai_analyst = AIAnalyst(
                    api_key=config.glm_api_key,
                    model=config.glm_model,
                    enabled=True,
                    signal_scoring=config.ai_signal_scoring,
                    daily_briefing=config.ai_daily_briefing,
                    regime_detection=config.ai_regime_detection,
                )
            except Exception as e:
                logger.warning(f"AI Analyst init failed: {e}")
        else:
            logger.info("🧠 AI Analyst: DISABLED (set GLM_API_KEY to enable)")
        
        # Track daily briefing time
        self._last_briefing_time = 0
        
        # Per-symbol indicator cache (for AI analysis)
        self._indicator_cache: Dict[str, dict] = {}

    def run(self):
        """Uruchom główną pętlę bota."""
        self._running = True

        # Rejestruj handler na SIGINT/SIGTERM
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        logger.info("=" * 60)
        logger.info("  CRYPTO STOCH SIGNAL BOT v2")
        logger.info(f"  Stochastic ({self.config.stoch_k_length}, {self.config.stoch_k_smooth}, {self.config.stoch_d_smooth})")
        logger.info(f"  Oversold: < {self.config.oversold_threshold} | Overbought: > {self.config.overbought_threshold}")
        logger.info(f"  Pary: {', '.join(self.config.symbols)}")
        logger.info(f"  Timeframes: {', '.join(self.config.timeframes)}")
        logger.info(f"  Interval: {self.config.scan_interval}s")
        logger.info(f"  Market: {self.config.market_source}")
        logger.info(f"  Trend filter: {self.config.trend_filter_mode}")
        logger.info(f"  Position tracking: {'✅' if self.position_tracker else '❌'}")
        logger.info(f"  Mode: {'⚠️ AUTO-TRADE' if self.config.auto_open_positions else '📡 ALERT ONLY (no execution)'}")
        logger.info(f"  AI Sentiment: {'✅' if self.sentiment_engine else '❌'}")
        logger.info(f"  Discord: {'TEST MODE (brak wysyłki)' if self.test_mode else '✅'}")
        logger.info("=" * 60)

        # Wyślij wiadomość powitalną na Discord
        if self.notifier and not self.test_mode:
            try:
                self.notifier.send_startup_message({
                    "symbols": self.config.symbols,
                    "timeframes": self.config.timeframes,
                    "stoch_k_length": self.config.stoch_k_length,
                    "stoch_k_smooth": self.config.stoch_k_smooth,
                    "stoch_d_smooth": self.config.stoch_d_smooth,
                    "oversold_threshold": self.config.oversold_threshold,
                    "overbought_threshold": self.config.overbought_threshold,
                    "scan_interval": self.config.scan_interval,
                    "require_crossover": self.config.require_crossover,
                    "strategy_name": "NWO + Stoch + CVD v2",
                    "use_training": True,
                })
            except Exception as e:
                logger.warning(f"Nie można wysłać wiadomości powitalnej: {e}")

        # Główna pętla
        while self._running:
            try:
                self._scan_cycle()
            except KeyboardInterrupt:
                break
            except Exception as e:
                self._errors += 1
                logger.error(f"Błąd w cyklu skanowania: {e}")
                if self.notifier and self._errors <= 3:
                    try:
                        self.notifier.send_error(str(e))
                    except Exception:
                        pass

            # Czekaj do następnego skanu
            if self._running:
                wait_until_next = self._calculate_wait()
                if wait_until_next > 0:
                    # Sprawdzaj co 5s czy bot nie został zatrzymany
                    waited = 0
                    while waited < wait_until_next and self._running:
                        time.sleep(min(5, wait_until_next - waited))
                        waited += 5

        logger.info("Bot zatrzymany.")

    def _scan_cycle(self):
        """Pojedynczy cykl skanowania wszystkich par i timeframe'ów."""
        self._scan_count += 1
        cycle_start = time.time()
        total_signals = 0

        logger.info(f"{'─'*50}")
        logger.info(f"Skan #{self._scan_count} — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")

        # ─── Position tracker: check SL/TP closes ─────────────────────
        if self.position_tracker:
            try:
                current_prices = self._get_current_prices()
                closed_positions = self.position_tracker.check_closes(current_prices)
                for pos in closed_positions:
                    emoji = "🟢" if pos.pnl and pos.pnl > 0 else "🔴"
                    logger.info(f"  {emoji} POSITION CLOSED: {pos.direction} {pos.symbol} @ ${pos.close_price:,.2f} | PnL: ${pos.pnl:+,.2f} ({pos.pnl_pct:+.2f}%) | {pos.close_reason}")
                    # Notify on Discord about position close
                    if self.notifier and not self.test_mode:
                        try:
                            self._send_position_close_notification(pos)
                        except Exception:
                            pass
            except Exception as e:
                logger.debug(f"Position check error: {e}")

        # ─── Scan symbols ──────────────────────────────────────────────
        # Combine crypto + stock symbols if needed
        all_symbols = list(self.config.symbols)
        if self.config.market_source in ("stocks", "both") and self.config.stock_symbols:
            for sym in self.config.stock_symbols:
                if sym not in all_symbols:
                    all_symbols.append(sym)

        for timeframe in self.config.timeframes:
            logger.info(f"  TF: {timeframe}")

            for symbol in all_symbols:
                try:
                    signals = self._check_symbol(symbol, timeframe)
                    if signals:
                        for sig in signals:
                            # Sprawdź cooldown
                            cooldown_key = f"{sig.symbol}_{sig.timeframe}_{sig.signal_type}"
                            if self._is_on_cooldown(cooldown_key):
                                logger.debug(f"    {symbol}: sygnał {sig.signal_type} na cooldown")
                                continue

                            # ─── AI Signal Scoring (GLM) ────────────────
                            if self.ai_analyst and self.ai_analyst.signal_scoring:
                                try:
                                    indicator_ctx = self._indicator_cache.get(f"{symbol}_{sig.timeframe}", {})
                                    signal_data = {
                                        "signal_type": sig.signal_type,
                                        "symbol": sig.symbol,
                                        "timeframe": sig.timeframe,
                                        "price": sig.price,
                                        "source": sig.extra_data.get("source", ""),
                                        "confidence": sig.extra_data.get("confidence", "MEDIUM"),
                                        "sl": sig.extra_data.get("sl", 0),
                                        "tp": sig.extra_data.get("tp", 0),
                                    }
                                    quality = self.ai_analyst.score_signal(signal_data, indicator_ctx)
                                    if quality:
                                        sig.extra_data["ai_score"] = quality.score
                                        sig.extra_data["ai_confidence"] = quality.confidence
                                        sig.extra_data["ai_analysis"] = quality.analysis
                                        sig.extra_data["ai_recommendation"] = quality.recommendation
                                        sig.extra_data["ai_risks"] = quality.risks
                                        sig.extra_data["ai_key_factors"] = quality.key_factors
                                        score_tag = f" AI:{quality.score}/10({quality.recommendation})"
                                        logger.info(f"    🧠 AI Score: {quality.score}/10 — {quality.analysis[:80]}")
                                except Exception as e:
                                    logger.debug(f"AI scoring error: {e}")

                            # Wyślij na Discord
                            sent = False
                            if self.notifier and not self.test_mode:
                                sent = self.notifier.send_signal(sig)
                            elif self.test_mode:
                                sent = True  # W test mode traktuj jako wysłany

                            if sent:
                                self._signals_sent += 1
                                self._cooldowns[cooldown_key] = time.time()
                                total_signals += 1

                                # Log signal
                                risk_tag = " ⚠️RISKY" if sig.extra_data.get("against_trend") else ""
                                logger.info(f"    {sig.emoji} {symbol}: {sig.signal_type}{risk_tag} @ ${sig.price:,.2f} | K={sig.k_value} D={sig.d_value} | {sig.reason}")

                                # ─── Auto-open position (ONLY if auto_open_positions=True) ───
                                if self.position_tracker and self.config.auto_open_positions and not sig.extra_data.get("against_trend", False):
                                    try:
                                        self.position_tracker.open_position(
                                            symbol=sig.symbol,
                                            direction=sig.signal_type,
                                            entry_price=sig.price,
                                            sl=sig.extra_data.get("sl", 0),
                                            tp=sig.extra_data.get("tp", 0),
                                            timeframe=sig.timeframe,
                                            strategy=sig.strategy_name,
                                            signal_reason=sig.reason,
                                            atr=sig.extra_data.get("atr", 0),
                                            risk_level=sig.extra_data.get("risk_level", "NORMAL"),
                                        )
                                    except Exception as e:
                                        logger.debug(f"Position open error: {e}")
                            else:
                                logger.debug(f"    {symbol}: sygnał zablokowany (spam/quiet)")

                except Exception as e:
                    logger.error(f"    ❌ {symbol} {timeframe}: {e}")
                    continue

        elapsed = time.time() - cycle_start
        logger.info(f"  Skan zakończony w {elapsed:.1f}s — sygnałów: {total_signals}")

        # ─── Clean stale cooldowns ──────────────────────────────────────
        if self._scan_count % 20 == 0:
            now = time.time()
            stale = [k for k, v in self._cooldowns.items() if now - v > self.config.cooldown_per_signal]
            for k in stale:
                del self._cooldowns[k]

        # ─── Periodic stats ────────────────────────────────────────────
        if self.position_tracker and self._scan_count % 10 == 0:
            try:
                stats = self.position_tracker.get_stats()
                if stats["total_trades"] > 0:
                    logger.info(f"  📊 Positions: {stats['total_trades']} trades | WR: {stats['win_rate']}% | PnL: ${stats['total_pnl']:+,.2f} | PF: {stats['profit_factor']}")
            except Exception:
                pass

        # Status update na Discord
        if self.notifier and not self.test_mode:
            if time.time() - self._last_status_time > self.config.status_interval:
                self._send_status_update()
                self._last_status_time = time.time()
        
        # ─── Daily Market Briefing (GLM) ────────────────────────────
        if self.ai_analyst and self.ai_analyst.daily_briefing and self.notifier and not self.test_mode:
            try:
                # Wyślij briefing co 6 godzin lub jeśli nowy dzień
                now = time.time()
                if now - self._last_briefing_time > 21600:  # 6 godzin
                    self._send_daily_briefing()
                    self._last_briefing_time = now
            except Exception as e:
                logger.debug(f"Daily briefing error: {e}")

    def _check_symbol(self, symbol: str, timeframe: str) -> List[Signal]:
        """Sprawdź jedną parę na jednym timeframe i zwróć sygnały."""
        try:
            df = self.fetcher.fetch_ohlcv(symbol, timeframe)
            if df.empty:
                return []
        except Exception as e:
            logger.debug(f"Błąd pobierania {symbol}: {e}")
            return []

        # Użyj custom strategy lub domyślnego detektora
        if self.custom_strategy_fn:
            signals = self.custom_strategy_fn(df, symbol, timeframe)
        else:
            signals = self.detector.detect(df, symbol, timeframe)
        
        # ─── Cache indicator context for AI analysis ──────────────
        if self.ai_analyst:
            try:
                from custom_strategy import get_current_nwo_state
                state = get_current_nwo_state(df, symbol, timeframe)
                if state:
                    cache_key = f"{symbol}_{timeframe}"
                    self._indicator_cache[cache_key] = state
            except Exception:
                pass
        
        return signals

    def _get_current_prices(self) -> Dict[str, float]:
        """Pobierz aktualne ceny dla otwartych pozycji."""
        prices = {}

        if not self.position_tracker:
            return prices

        open_positions = self.position_tracker.get_open_positions()
        symbols_needed = list(set(p.symbol for p in open_positions))

        for symbol in symbols_needed:
            try:
                price = self.fetcher.get_latest_price(symbol) if hasattr(self.fetcher, 'get_latest_price') else None
                if price is not None:
                    prices[symbol] = price
                else:
                    # Fallback: pobierz ostatnią świecę
                    df = self.fetcher.fetch_ohlcv(symbol, self.config.timeframes[0])
                    if not df.empty:
                        prices[symbol] = df['close'].iloc[-1]
            except Exception:
                pass

        return prices

    def _send_position_close_notification(self, pos):
        """Wyślij powiadomienie o zamknięciu pozycji na Discord."""
        if not self.notifier:
            return

        emoji = "🟢" if pos.pnl and pos.pnl > 0 else "🔴"
        pnl_str = f"${pos.pnl:+,.2f}" if pos.pnl else "N/A"
        pnl_pct_str = f"({pos.pnl_pct:+.2f}%)" if pos.pnl_pct else ""

        embed = {
            "title": f"{emoji} Position Closed — {pos.symbol}",
            "color": 0x00E676 if pos.pnl and pos.pnl > 0 else 0xFF1744,
            "fields": [
                {"name": "Direction", "value": pos.direction, "inline": True},
                {"name": "Entry", "value": f"${pos.entry_price:,.2f}", "inline": True},
                {"name": "Exit", "value": f"${pos.close_price:,.2f}" if pos.close_price else "N/A", "inline": True},
                {"name": "PnL", "value": f"{emoji} {pnl_str} {pnl_pct_str}", "inline": True},
                {"name": "Reason", "value": pos.close_reason or "N/A", "inline": True},
                {"name": "Holding", "value": f"{pos.holding_time_hours:.1f}h" if pos.holding_time_hours else "N/A", "inline": True},
            ],
            "footer": {"text": "Position Tracker | Crypto Signal Bot"},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        payload = {
            "username": self.notifier.bot_name,
            "embeds": [embed],
        }

        self.notifier.send_custom_embed(embed)

    def _is_on_cooldown(self, key: str) -> bool:
        """Sprawdź czy sygnał jest na cooldown (anti-spam)."""
        if key not in self._cooldowns:
            return False
        elapsed = time.time() - self._cooldowns[key]
        return elapsed < self.config.cooldown_per_signal

    def _calculate_wait(self) -> float:
        """Oblicz ile sekund czekać do następnego skanu."""
        return float(self.config.scan_interval)

    def _send_status_update(self):
        """Wyślij status bota na Discord."""
        logger.info("  Wysyłam status na Discord...")
        for symbol in self.config.symbols[:3]:  # Tylko top 3 żeby nie spamować
            for timeframe in self.config.timeframes[:1]:
                try:
                    df = self.fetcher.fetch_ohlcv(symbol, timeframe)
                    if df.empty:
                        continue
                    values = self.detector.get_current_values(df)
                    if values and self.notifier:
                        self.notifier.send_status(
                            symbol, timeframe, values,
                            detector_config={
                                "k_length": self.config.stoch_k_length,
                                "k_smooth": self.config.stoch_k_smooth,
                                "d_smooth": self.config.stoch_d_smooth,
                            }
                        )
                except Exception:
                    pass
    
    def _send_daily_briefing(self):
        """Wyślij daily market briefing przez GLM na Discord."""
        if not self.ai_analyst or not self.notifier:
            return
        
        logger.info("  🧠 Generuję daily market briefing...")
        
        # Zbierz dane rynkowe z cache
        market_snapshot = {"pairs": [], "regimes": {}}
        for symbol in self.config.symbols:
            # Użyj 1h jako głównego TF
            cache_key = f"{symbol}_1h"
            state = self._indicator_cache.get(cache_key)
            if state:
                pair_data = {"symbol": symbol}
                pair_data.update(state)
                market_snapshot["pairs"].append(pair_data)
        
        briefing = self.ai_analyst.generate_daily_briefing(market_snapshot)
        
        if briefing:
            # Build Discord embed
            bias_emoji = {"bullish": "🟢📈", "bearish": "🔴📉", "neutral": "⚪↔️", "mixed": "🟡🔀"}.get(briefing.overall_bias, "❓")
            
            fields = [
                {
                    "name": "Overall Bias",
                    "value": f"{bias_emoji} **{briefing.overall_bias.upper()}**",
                    "inline": True
                },
                {
                    "name": "Summary",
                    "value": briefing.summary or "N/A",
                    "inline": False
                },
            ]
            
            # Key pairs
            if briefing.key_pairs:
                pairs_text = ""
                for kp in briefing.key_pairs[:5]:
                    kp_emoji = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}.get(kp.get("bias", ""), "⚪")
                    pairs_text += f"{kp_emoji} **{kp.get('symbol', '?')}** — {kp.get('comment', 'N/A')}\n"
                fields.append({
                    "name": "Key Pairs",
                    "value": pairs_text.strip(),
                    "inline": False
                })
            
            # Risk events
            if briefing.risk_events:
                risks_text = "\n".join([f"⚠️ {r}" for r in briefing.risk_events[:3]])
                fields.append({
                    "name": "Risk Events",
                    "value": risks_text,
                    "inline": False
                })
            
            # Watchlist
            if briefing.watchlist:
                fields.append({
                    "name": "👀 Watchlist",
                    "value": ", ".join(briefing.watchlist[:5]),
                    "inline": False
                })
            
            embed = {
                "title": "🧠 AI Daily Market Briefing",
                "color": 0x9C27B0,  # Purple for AI
                "fields": fields,
                "footer": {"text": f"GLM AI Analyst | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC"},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            
            self.notifier.send_custom_embed(embed)
            logger.info(f"  🧠 Briefing wysłany: bias={briefing.overall_bias}")

    def _handle_signal(self, signum, frame):
        """Handler sygnałów systemowych (Ctrl+C)."""
        logger.info("Otrzymano sygnał zatrzymania...")
        self._running = False

    def stop(self):
        """Zatrzymaj bota."""
        self._running = False

    @property
    def stats(self) -> dict:
        result = {
            "running": self._running,
            "scans": self._scan_count,
            "signals_sent": self._signals_sent,
            "errors": self._errors,
            "cooldowns_active": len(self._cooldowns),
            "fetcher": self.fetcher.stats if hasattr(self.fetcher, 'stats') else {},
        }
        if self.position_tracker:
            result["positions"] = self.position_tracker.get_stats()
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# ONE-SHOT SCAN (bez pętli, pojedynczy skan)
# ═══════════════════════════════════════════════════════════════════════════════

def run_single_scan(config: BotConfig, test_mode: bool = True, strategy: str = "nwo_stoch_cvd"):
    """Uruchom pojedynczy skan i wyświetl wyniki (bez pętli, bez Discord)."""
    from custom_strategy import get_current_nwo_state, strategy_nwo_stoch_cvd, STRATEGY_REGISTRY

    use_nwo = strategy == "nwo_stoch_cvd"

    logger.info("Jednorazowy skan — sprawdzam sygnały...")
    if use_nwo:
        logger.info("Strategia: NWO + Stoch(7,3,2) + CVD")

    # Data fetcher
    if config.market_source in ("stocks", "both"):
        try:
            from data_fetcher_yfinance import UnifiedDataFetcher
            fetcher = UnifiedDataFetcher()
        except ImportError:
            fetcher = DataFetcher(
                exchange_id=config.exchange,
                rate_limit_ms=config.rate_limit_ms,
                cache_ttl_seconds=config.cache_ttl,
                candles_per_fetch=config.candles_per_fetch,
            )
    else:
        fetcher = DataFetcher(
            exchange_id=config.exchange,
            rate_limit_ms=config.rate_limit_ms,
            cache_ttl_seconds=config.cache_ttl,
            candles_per_fetch=config.candles_per_fetch,
        )

    detector = SignalDetector(
        stoch_k_length=config.stoch_k_length,
        stoch_k_smooth=config.stoch_k_smooth,
        stoch_d_smooth=config.stoch_d_smooth,
        oversold_threshold=config.oversold_threshold,
        overbought_threshold=config.overbought_threshold,
        require_crossover=config.require_crossover,
    )

    notifier = None
    if not test_mode and config.discord_webhook_url:
        notifier = DiscordNotifier(webhook_url=config.discord_webhook_url, bot_name=config.discord_bot_name)

    # Combine symbols
    all_symbols = list(config.symbols)
    if config.market_source in ("stocks", "both") and config.stock_symbols:
        for sym in config.stock_symbols:
            if sym not in all_symbols:
                all_symbols.append(sym)

    for timeframe in config.timeframes:
        if use_nwo:
            logger.info(f"\n  Timeframe: {timeframe}")
            logger.info(f"  {'Symbol':<12} {'Price':>10} {'NWO':>6} {'K':>6} {'D':>6} {'CVD':>6} {'Risk':<7} {'Signal'}")
            logger.info(f"  {'─'*12} {'─'*10} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*7} {'─'*25}")
        else:
            logger.info(f"\n  Timeframe: {timeframe}")
            logger.info(f"  {'Symbol':<12} {'Price':>10} {'K':>6} {'D':>6} {'Zone':<12} {'Signal'}")
            logger.info(f"  {'─'*12} {'─'*10} {'─'*6} {'─'*6} {'─'*12} {'─'*20}")

        for symbol in all_symbols:
            try:
                df = fetcher.fetch_ohlcv(symbol, timeframe)
                if df.empty:
                    continue

                if use_nwo:
                    nwo_state = get_current_nwo_state(df, symbol, timeframe)
                    signals = strategy_nwo_stoch_cvd(df, symbol, timeframe)

                    if nwo_state:
                        signal_str = ""
                        for s in signals:
                            risk_tag = "⚠️" if s.extra_data.get("against_trend") else ""
                            signal_str += f"{s.emoji} {s.signal_type}{risk_tag} ({s.extra_data.get('source','')})"
                            if notifier:
                                notifier.send_signal(s)

                        risk = "HIGH" if any(s.extra_data.get("against_trend") for s in signals) else "OK"

                        logger.info(
                            f"  {symbol:<12} ${nwo_state['price']:>9,.2f} "
                            f"{nwo_state['osc']:>5.1f} "
                            f"{nwo_state['stoch_k'] or 0:>5.1f} {nwo_state['stoch_d'] or 0:>5.1f} "
                            f"{nwo_state['cvd'] or 0:>+5.2f} "
                            f"{risk:<7} "
                            f"{signal_str or '—'}"
                        )
                else:
                    values = detector.get_current_values(df)
                    if values:
                        zone = values['zone']
                        signals = detector.detect(df, symbol, timeframe)

                        signal_str = ""
                        for s in signals:
                            signal_str += f"{s.emoji} {s.signal_type}"
                            if notifier:
                                notifier.send_signal(s)

                        logger.info(
                            f"  {symbol:<12} ${values['price']:>9,.2f} "
                            f"{values['stoch_k']:>5.1f} {values['stoch_d']:>5.1f} "
                            f"{zone:<12} {signal_str or '—'}"
                        )

            except Exception as e:
                logger.error(f"  {symbol}: Błąd — {e}")

    # Show position stats if tracking enabled
    if config.use_position_tracking:
        try:
            from position_tracker import PositionTracker
            tracker = PositionTracker(db_path=config.position_db_path)
            stats = tracker.get_stats()
            if stats["total_trades"] > 0:
                logger.info(f"\n  📊 Position Stats:")
                logger.info(f"     Total trades: {stats['total_trades']}")
                logger.info(f"     Win rate: {stats['win_rate']}%")
                logger.info(f"     Total PnL: ${stats['total_pnl']:+,.2f}")
                logger.info(f"     Profit factor: {stats['profit_factor']}")
                logger.info(f"     Best trade: ${stats['best_trade']:+,.2f}")
                logger.info(f"     Worst trade: ${stats['worst_trade']:+,.2f}")
            tracker.close()
        except:
            pass

    logger.info("\nSkan zakończony.")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Crypto Stoch Signal Bot — Discord Notifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Przykłady:
  python bot.py --webhook https://discord.com/api/webhooks/...   Uruchom bota
  python bot.py --test                                            Test (bez Discord)
  python bot.py --scan                                            Pojedynczy skan
  python bot.py --config aggressive                               Preset agresywny
  python bot.py --symbols BTC/USDT,ETH/USDT                      Tylko te pary
  python bot.py --sentiment --cryptopanic-key KEY                 AI sentiment
  python bot.py --market both                                     Crypto + SP500/US100
  python bot.py --trend-filter block                              Blokuj against trend
  python bot.py --position-size 200                               Rozmiar pozycji $200
        """
    )

    parser.add_argument('--webhook', '-w', type=str, default=None,
                        help='Discord Webhook URL')
    parser.add_argument('--test', '-t', action='store_true',
                        help='Tryb testowy (bez wysyłki na Discord)')
    parser.add_argument('--scan', action='store_true',
                        help='Pojedynczy skan (bez pętli)')
    parser.add_argument('--config', '-c', type=str, default='default',
                        choices=['default', 'aggressive', 'conservative', 'scalping'],
                        help='Preset konfiguracji')
    parser.add_argument('--symbols', type=str, default=None,
                        help='Lista par oddzielona przecinkami (np. BTC/USDT,ETH/USDT)')
    parser.add_argument('--timeframes', '-tf', type=str, default=None,
                        help='Timeframe\'y oddzielone przecinkami (np. 5m,15m,1h)')
    parser.add_argument('--oversold', type=float, default=None,
                        help='Próg oversold Stochastic (domyślnie 20)')
    parser.add_argument('--overbought', type=float, default=None,
                        help='Próg overbought Stochastic (domyślnie 80)')
    parser.add_argument('--no-crossover', action='store_true',
                        help='Nie wymagaj crossoveru K/D')
    parser.add_argument('--interval', type=int, default=None,
                        help='Interwał skanowania w sekundach')
    parser.add_argument('--exchange', type=str, default=None,
                        help='Giełda (domyślnie binance)')
    parser.add_argument('--role-id', type=str, default=None,
                        help='ID roli Discord do @mention')
    parser.add_argument('--strategy', type=str, default='nwo_stoch_cvd',
                        choices=list(STRATEGY_REGISTRY.keys()),
                        help='Strategia sygnałowa')

    # ─── v2 options ──────────────────────────────────────────────────
    parser.add_argument('--sentiment', action='store_true',
                        help='Włącz AI news sentiment filter')
    parser.add_argument('--no-sentiment', action='store_true',
                        help='Wyłącz AI news sentiment filter')
    parser.add_argument('--cryptopanic-key', type=str, default='',
                        help='CryptoPanic API key (PŁATNY, opcjonalny)')
    parser.add_argument('--finnhub-key', type=str, default='',
                        help='Finnhub API key (darmowy)')
    parser.add_argument('--market', type=str, default=None,
                        choices=['crypto', 'stocks', 'both'],
                        help='Rynek: crypto (Binance), stocks (YFinance), both')
    parser.add_argument('--trend-filter', type=str, default=None,
                        choices=['alert', 'block', 'off'],
                        help='Trend filter mode: alert (default), block, off')
    parser.add_argument('--position-size', type=float, default=None,
                        help='Domyślny rozmiar pozycji w USD (domyślnie 100)')
    parser.add_argument('--no-positions', action='store_true',
                        help='Wyłącz position tracking')
    parser.add_argument('--auto-trade', action='store_true',
                        help='⚠️ AUTO-OTWIERAJ pozycje (domyslnie OFF = alert only)')
    parser.add_argument('--log', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Poziom logowania')

    args = parser.parse_args()

    # Wybierz preset konfiguracji
    if args.config == 'aggressive':
        config = config_aggressive()
    elif args.config == 'conservative':
        config = config_conservative()
    elif args.config == 'scalping':
        config = config_scalping()
    else:
        config = BotConfig()

    # Nadpisz konfigurację z argumentów CLI
    if args.webhook:
        config.discord_webhook_url = args.webhook
    if args.symbols:
        config.symbols = [s.strip() for s in args.symbols.split(',')]
    if args.timeframes:
        config.timeframes = [tf.strip() for tf in args.timeframes.split(',')]
    if args.oversold is not None:
        config.oversold_threshold = args.oversold
    if args.overbought is not None:
        config.overbought_threshold = args.overbought
    if args.no_crossover:
        config.require_crossover = False
    if args.interval:
        config.scan_interval = args.interval
    if args.exchange:
        config.exchange = args.exchange
    if args.role_id:
        config.discord_role_id = args.role_id
    config.log_level = args.log

    # ─── v2 config overrides ─────────────────────────────────────────
    # Sentiment
    if args.sentiment:
        config.use_sentiment = True
    if args.no_sentiment:
        config.use_sentiment = False

    # API keys
    if args.cryptopanic_key:
        config.cryptopanic_api_key = args.cryptopanic_key
    if args.finnhub_key:
        config.finnhub_api_key = args.finnhub_key

    # Market source
    if args.market:
        config.market_source = args.market

    # Trend filter
    if args.trend_filter:
        config.trend_filter_mode = args.trend_filter

    # Position tracking
    if args.no_positions:
        config.use_position_tracking = False
    if args.position_size:
        config.default_position_size_usd = args.position_size
    if args.auto_trade:
        config.auto_open_positions = True

    # Re-konfiguruj logging
    setup_logging(args.log)

    # Walidacja
    if not args.test and not args.scan:
        errors = config.validate()
        if errors:
            for err in errors:
                logger.error(f"Błąd konfiguracji: {err}")
            logger.error("Użyj --test do testowania bez Discord, lub podaj --webhook URL")
            sys.exit(1)

    # Wyświetl konfigurację
    logger.info(f"\nKonfiguracja:\n{config.summary()}\n")

    # Uruchom
    if args.scan:
        run_single_scan(config, test_mode=args.test, strategy=args.strategy)
    else:
        bot = StochSignalBot(config, test_mode=args.test)

        # Ustaw custom strategy jeśli wybrana
        if args.strategy != 'stoch_7_3_2' and STRATEGY_REGISTRY[args.strategy]['fn']:
            bot.custom_strategy_fn = STRATEGY_REGISTRY[args.strategy]['fn']
            logger.info(f"Strategia: {STRATEGY_REGISTRY[args.strategy]['name']}")

        try:
            bot.run()
        except KeyboardInterrupt:
            bot.stop()


if __name__ == "__main__":
    main()
