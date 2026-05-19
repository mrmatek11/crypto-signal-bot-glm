"""
Configuration Module
Wszystkie ustawienia bota w jednym miejscu.

v2: dodano AI news sentiment, YFinance (SP500/US100), position tracking, trend filter mode
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class BotConfig:
    """Główna konfiguracja bota sygnałowego."""

    # ─── Discord ───────────────────────────────────────────────────────────
    discord_webhook_url: str = ""  # Also reads from DISCORD_WEBHOOK env var
    discord_bot_name: str = "📊 Crypto Stoch Bot"
    discord_avatar_url: str = "https://cdn-icons-png.flaticon.com/512/6001/6001527.png"
    discord_role_id: Optional[str] = None        # ID roli do @mention
    mention_on_long: bool = True
    mention_on_short: bool = True
    quiet_hours: Optional[dict] = None           # {"start": 23, "end": 7} UTC

    # ─── Exchange ──────────────────────────────────────────────────────────
    exchange: str = "binance"
    rate_limit_ms: int = 500                      # ms między requestami
    cache_ttl: int = 30                           # sekundy cache'u

    # ─── Watchlist ─────────────────────────────────────────────────────────
    symbols: List[str] = field(default_factory=lambda: [
        "BTC/USDT",
        "ETH/USDT",
        "SOL/USDT",
        "BNB/USDT",
        "XRP/USDT",
        "ADA/USDT",
        "DOGE/USDT",
        "AVAX/USDT",
        "DOT/USDT",
        "LINK/USDT",
    ])

    timeframes: List[str] = field(default_factory=lambda: [
        "5m",
        "15m",
        "1h",
    ])

    # ─── Stochastic Settings ───────────────────────────────────────────────
    stoch_k_length: int = 7
    stoch_k_smooth: int = 3
    stoch_d_smooth: int = 2
    oversold_threshold: float = 20.0
    overbought_threshold: float = 80.0

    # ─── Signal Filters ────────────────────────────────────────────────────
    require_crossover: bool = True               # K musi przeciąć D (nie tylko być w strefie)
    rsi_filter: bool = False                     # Dodatkowy filtr RSI
    rsi_oversold: float = 35.0
    rsi_overbought: float = 65.0
    volume_filter: bool = False                  # Filtr wolumenu
    volume_mult: float = 1.5

    # ─── Trend Filter ─────────────────────────────────────────────────────
    trend_filter_mode: str = "alert"              # "alert", "block", "off"
    # alert = alertuj z ostrzezeniem (default)
    # block = blokuj sygnal w ogole
    # off = brak filtra trendu

    # ─── Scanning ──────────────────────────────────────────────────────────
    scan_interval: int = 60                      # Sekundy między skanami
    candles_per_fetch: int = 100                 # Ile świec pobierać
    cooldown_per_signal: int = 300               # Sekundy cooldown dla tego samego sygnału

    # ─── AI News Sentiment ──────────────────────────────────────────────────
    use_sentiment: bool = False                  # Wlacz AI news sentiment filter
    cryptopanic_api_key: str = ""                # Also reads from CRYPTOPANIC_KEY env var
    finnhub_api_key: str = ""                    # Also reads from FINNHUB_KEY env var
    newsapi_key: str = ""                        # Also reads from NEWSAPI_KEY env var
    sentiment_refresh_interval: int = 300        # Sekundy miedzy refresh sentimentu
    sentiment_block_threshold: float = 0.5       # |score| > tego → blokuj sygnal

    # ─── Market Data Source ────────────────────────────────────────────────
    market_source: str = "crypto"                 # "crypto", "stocks", "both"
    # crypto = tylko Binance (ccxt)
    # stocks = tylko YFinance (SPY, QQQ, akcje)
    # both = auto-wybierz zrodl

    # ─── Stock/ETF Symbols (YFinance) ───────────────────────────────────────
    stock_symbols: List[str] = field(default_factory=lambda: [
        "SP500",      # SPY ETF
        "US100",      # QQQ ETF
    ])

    # ─── Position Tracking ──────────────────────────────────────────────────
    use_position_tracking: bool = True           # Sledzenie pozycji (INFO, nie egzekucja!)
    auto_open_positions: bool = False             # ⚠️ False = ALERT ONLY, True = auto-otwieraj
    position_db_path: str = ""                   # Auto-detected: /app/data/positions.db in Docker, positions.db locally
    default_position_size_usd: float = 100       # Domyslny rozmiar pozycji (USD)
    max_open_positions: int = 10                  # Max otwartych pozycji
    position_timeout_hours: float = 72           # Max czas otwartej pozycji (godziny)

    # ─── Status Updates ────────────────────────────────────────────────────
    status_interval: int = 3600                  # Co ile sekund wysyłać status na Discord
    log_level: str = "INFO"                      # DEBUG, INFO, WARNING, ERROR
    
    # ─── GLM AI Analyst ────────────────────────────────────────────────────
    glm_api_key: str = ""                        # Also reads from GLM_API_KEY env var
    glm_model: str = "glm-4-flash"               # glm-4-flash (fast/cheap) or glm-4-plus (quality)
    ai_signal_scoring: bool = True               # LLM ocenia jakość sygnału (1-10)
    ai_daily_briefing: bool = True               # Generuj daily briefing co 6h
    ai_regime_detection: bool = True             # AI klasyfikuje reżim rynkowy
    ai_min_score_filter: int = 0                 # 0 = pokaż wszystkie, >0 = filtruj sygnały z AI score < N

    def __post_init__(self):
        """Resolve env vars for secrets and auto-detect DB path."""
        if not self.discord_webhook_url:
            self.discord_webhook_url = os.getenv("DISCORD_WEBHOOK", "")
        if not self.cryptopanic_api_key:
            self.cryptopanic_api_key = os.getenv("CRYPTOPANIC_KEY", "")
        if not self.finnhub_api_key:
            self.finnhub_api_key = os.getenv("FINNHUB_KEY", "")
        if not self.newsapi_key:
            self.newsapi_key = os.getenv("NEWSAPI_KEY", "")
        if not self.glm_api_key:
            self.glm_api_key = os.getenv("GLM_API_KEY", "")
        if not self.position_db_path:
            if os.path.isdir("/app/data"):
                self.position_db_path = "/app/data/positions.db"
            else:
                self.position_db_path = "positions.db"

    def validate(self) -> List[str]:
        """Waliduj konfigurację. Zwraca listę błędów."""
        errors = []
        if not self.discord_webhook_url:
            errors.append("discord_webhook_url jest wymagany!")
        elif not self.discord_webhook_url.startswith("https://discord.com/api/webhooks/") and \
           not self.discord_webhook_url.startswith("https://discordapp.com/api/webhooks/"):
            errors.append(f"discord_webhook_url nie wygląda na prawidłowy webhook Discord")
        if self.scan_interval < 10:
            errors.append("scan_interval musi być >= 10 sekund")
        if not self.symbols:
            errors.append("symbols nie może być puste")
        if not self.timeframes:
            errors.append("timeframes nie może być puste")
        if self.trend_filter_mode not in ("alert", "block", "off"):
            errors.append(f"trend_filter_mode musi być 'alert', 'block' lub 'off', jest '{self.trend_filter_mode}'")
        if self.market_source not in ("crypto", "stocks", "both"):
            errors.append(f"market_source musi być 'crypto', 'stocks' lub 'both', jest '{self.market_source}'")
        return errors

    def summary(self) -> str:
        """Zwraca podsumowanie konfiguracji."""
        trend_icon = {"alert": "⚠️", "block": "🚫", "off": "❌"}.get(self.trend_filter_mode, "?")
        return (
            f"Exchange: {self.exchange}\n"
            f"Symbols: {', '.join(self.symbols)}\n"
            f"Timeframes: {', '.join(self.timeframes)}\n"
            f"Stochastic: ({self.stoch_k_length}, {self.stoch_k_smooth}, {self.stoch_d_smooth})\n"
            f"Oversold: < {self.oversold_threshold} | Overbought: > {self.overbought_threshold}\n"
            f"Crossover required: {self.require_crossover}\n"
            f"Trend filter: {trend_icon} {self.trend_filter_mode}\n"
            f"Scan interval: {self.scan_interval}s\n"
            f"Market: {self.market_source}\n"
            f"AI Sentiment: {'✅' if self.use_sentiment else '❌'}\n"
            f"Position tracking: {'✅' if self.use_position_tracking else '❌'}\n"
            f"Discord: {'✅' if self.discord_webhook_url else '❌'}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PRESET CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def config_aggressive() -> BotConfig:
    """Agresywna konfiguracja — więcej sygnałów, luźniejsze filtry."""
    return BotConfig(
        oversold_threshold=25.0,
        overbought_threshold=75.0,
        require_crossover=False,
        scan_interval=30,
        timeframes=["5m", "15m"],
        cooldown_per_signal=180,
    )


def config_conservative() -> BotConfig:
    """Konserwatywna konfiguracja — mniej sygnałów, mocniejsze filtry."""
    return BotConfig(
        oversold_threshold=15.0,
        overbought_threshold=85.0,
        require_crossover=True,
        rsi_filter=True,
        rsi_oversold=30.0,
        rsi_overbought=70.0,
        volume_filter=True,
        scan_interval=120,
        timeframes=["1h", "4h"],
        cooldown_per_signal=600,
    )


def config_scalping() -> BotConfig:
    """Konfiguracja pod skalping — niskie TF, szybkie skanowanie."""
    return BotConfig(
        symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        timeframes=["1m", "5m"],
        scan_interval=15,
        candles_per_fetch=50,
        require_crossover=True,
        cooldown_per_signal=60,
    )
