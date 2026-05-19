"""
Crypto Stoch Signal Bot — Discord Notifier

Monitoruje pary krypto na żywo, wykrywa sygnały Stochastic (7,3,2)
i wysyła alerty na Discord przez Webhook.
"""

from .config import BotConfig
from .signal_detector import SignalDetector, Signal
from .data_fetcher import DataFetcher
from .discord_notifier import DiscordNotifier
from .custom_strategy import STRATEGY_REGISTRY
