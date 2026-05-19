"""
Data Fetcher — Yahoo Finance (SP500, US100, akcje)
═════════════════════════════════════════════════════════════════════════

Alternatywny provider danych do data_fetcher.py (który używa ccxt/Binance).

Obsługuje:
  - SPY (SP500 ETF)
  - QQQ (NASDAQ 100 ETF)
  - Dow Jones (DIA)
  - Dowolne tickery akcji (AAPL, TSLA, itd.)

Dane:
  - 1h, 4h, 1d: darmowe, miesiące/lat danych
  - 15m: darmowe, max 60 dni
  - 5m, 1m: darmowe, max 7-30 dni

Instalacja:
  pip install yfinance

Użycie:
  from data_fetcher_yfinance import YFinanceDataFetcher
  
  fetcher = YFinanceDataFetcher()
  df = fetcher.fetch_ohlcv("SPY", "1h")
"""

import time
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List
from collections import OrderedDict


# Mapowanie symboli krypto → Yahoo Finance tickery
CRYPTO_TO_YF = {
    "BTC/USDT": "BTC-USD",
    "ETH/USDT": "ETH-USD",
    "SOL/USDT": "SOL-USD",
    "BNB/USDT": "BNB-USD",
    "XRP/USDT": "XRP-USD",
    "ADA/USDT": "ADA-USD",
    "DOGE/USDT": "DOGE-USD",
    "AVAX/USDT": "AVAX-USD",
    "DOT/USDT": "DOT-USD",
    "LINK/USDT": "LINK-USD",
}

# Mapowanie indeksów → Yahoo Finance tickery
INDEX_TICKERS = {
    "SP500": "SPY",
    "US100": "QQQ",
    "NASDAQ": "QQQ",
    "DOW": "DIA",
    "DAX": "^GDAXI",
    "FTSE": "^FTSE",
    "NIKKEI": "^N225",
}

# Mapowanie timeframe → Yahoo Finance interval
TF_TO_YF = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "1d": "1d",
    "1w": "1wk",
    "1M": "1mo",
}

# Max historii per timeframe (YFinance limits)
TF_MAX_PERIOD = {
    "1m": "7d",
    "5m": "60d",
    "15m": "60d",
    "30m": "60d",
    "1h": "730d",
    "2h": "730d",
    "4h": "730d",
    "1d": "max",
    "1w": "max",
    "1M": "max",
}


class YFinanceDataFetcher:
    """
    Pobiera dane OHLCV z Yahoo Finance.
    
    Zalety vs ccxt:
    - Darmowe, bez API key
    - Indeksy giełdowe (SP500, NASDAQ)
    - Akcje (AAPL, TSLA)
    - Krypto też działa (BTC-USD)
    
    Wady:
    - Mniej danych intraday (max 60 dni na 15m)
    - Brak danych tick-level
    - Ograniczenia rate limit (2000 req/h)
    """
    
    def __init__(
        self,
        cache_ttl_seconds: int = 60,
    ):
        self.cache_ttl = cache_ttl_seconds
        self._cache: OrderedDict = OrderedDict()
        self._request_count = 0
        self._yf = None
    
    def _get_yfinance(self):
        """Lazy init yfinance."""
        if self._yf is not None:
            return self._yf
        
        try:
            import yfinance
            self._yf = yfinance
            return self._yf
        except ImportError:
            raise RuntimeError(
                "yfinance nie jest zainstalowany. Uruchom: pip install yfinance"
            )
    
    def _resolve_ticker(self, symbol: str) -> str:
        """Przekonwertuj symbol na Yahoo Finance ticker."""
        # Sprawdź indeksy
        if symbol.upper() in INDEX_TICKERS:
            return INDEX_TICKERS[symbol.upper()]
        
        # Sprawdź krypto
        if symbol in CRYPTO_TO_YF:
            return CRYPTO_TO_YF[symbol]
        
        # Sprawdź czy to już ticker YF (zawiera - lub .)
        if "-" in symbol or "." in symbol:
            return symbol
        
        # Default: traktuj jako stock ticker
        return symbol
    
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        period: str = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Pobierz dane OHLCV z Yahoo Finance.
        
        Args:
            symbol: Ticker (np. "SPY", "BTC/USDT", "AAPL") lub alias ("SP500", "US100")
            timeframe: Interwał ("1m", "5m", "15m", "1h", "4h", "1d")
            period: Okres danych (np. "1mo", "6mo", "1y", "max"). Jeśli None, auto-detect.
            force_refresh: Wymuś odświeżenie cache.
        
        Returns:
            DataFrame z kolumnami [open, high, low, close, volume] i DatetimeIndex
        """
        yf_ticker = self._resolve_ticker(symbol)
        yf_interval = TF_TO_YF.get(timeframe, "1h")
        
        cache_key = f"{yf_ticker}_{yf_interval}"
        
        # Check cache
        if not force_refresh and cache_key in self._cache:
            cached_df, cached_time = self._cache[cache_key]
            age = time.time() - cached_time
            if age < self.cache_ttl:
                return cached_df
        
        yf = self._get_yfinance()
        
        # Determine period
        if period is None:
            period = TF_MAX_PERIOD.get(timeframe, "1y")
        
        try:
            ticker = yf.Ticker(yf_ticker)
            df = ticker.history(period=period, interval=yf_interval)
            
            if df.empty:
                print(f"[YFinance] Brak danych dla {yf_ticker} {yf_interval}")
                if cache_key in self._cache:
                    return self._cache[cache_key][0]
                return pd.DataFrame()
            
            # Standaryzuj kolumny (YF używa Title Case)
            df.columns = [c.lower().replace(' ', '_') for c in df.columns]
            
            # Wybierz tylko OHLCV
            keep_cols = []
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    keep_cols.append(col)
            df = df[keep_cols]
            
            # Konwertuj timezone na UTC
            if df.index.tz is not None:
                df.index = df.index.tz_convert('UTC')
            else:
                df.index = df.index.tz_localize('UTC')
            
            # Usuń duplikaty
            df = df[~df.index.duplicated(keep='first')]
            df = df.sort_index()
            
            # Cast to float
            for col in df.columns:
                df[col] = df[col].astype(float)
            
            # Cache
            self._cache[cache_key] = (df, time.time())
            self._request_count += 1
            
            # Limit cache size
            if len(self._cache) > 50:
                self._cache.popitem(last=False)
            
            return df
            
        except Exception as e:
            print(f"[YFinance] Błąd pobierania {yf_ticker} {yf_interval}: {e}")
            if cache_key in self._cache:
                return self._cache[cache_key][0]
            raise
    
    def fetch_extended(
        self,
        symbol: str,
        timeframe: str,
        total_bars: int = 1500,
    ) -> pd.DataFrame:
        """
        Pobierz więcej barów niż default.
        
        Dla 1d/1w: używa period="max"
        Dla 1h/4h: używa period="2y" (dostarcza ~1500+ barów na 1h)
        Dla <1h: limited do 60 dni przez YFinance
        """
        # Map total_bars to period
        if timeframe in ("1d", "1w", "1M"):
            period = "max"
        elif timeframe in ("1h", "2h", "4h"):
            # 1h * 1500 bars = ~62 days, YFinance allows 730d
            period = "2y"
        else:
            # Intraday: YFinance limits to 60d
            period = "60d"
        
        return self.fetch_ohlcv(symbol, timeframe, period=period, force_refresh=True)
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Pobierz aktualną cenę (last close z najnowszej świecy)."""
        # Spróbuj z cache (szybko)
        yf_ticker = self._resolve_ticker(symbol)
        for interval in ["1m", "5m", "15m", "1h"]:
            cache_key = f"{yf_ticker}_{TF_TO_YF.get(interval, interval)}"
            if cache_key in self._cache:
                df, _ = self._cache[cache_key]
                if not df.empty:
                    return float(df['close'].iloc[-1])

        # Fallback: pobierz 1m dane
        try:
            df = self.fetch_ohlcv(symbol, "1m", period="1d")
            if not df.empty:
                return float(df['close'].iloc[-1])
        except Exception:
            pass

        # Ostateczny fallback: 1h dane
        try:
            df = self.fetch_ohlcv(symbol, "1h", period="5d")
            if not df.empty:
                return float(df['close'].iloc[-1])
        except Exception:
            pass

        return None

    @property
    def stats(self) -> dict:
        return {
            "source": "yfinance",
            "requests_made": self._request_count,
            "cache_size": len(self._cache),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED DATA FETCHER — auto-wybiera Binance lub YFinance
# ═══════════════════════════════════════════════════════════════════════════════

class UnifiedDataFetcher:
    """
    Automatycznie wybiera źródło danych:
    - Krypto (BTC/USDT, ETH/USDT) → Binance (ccxt)
    - Indeksy/akcje (SPY, SP500, AAPL) → Yahoo Finance
    
    Jeden interfejs, dwa backendy.
    """
    
    # Krypto symbols (use Binance)
    CRYPTO_PATTERNS = ["/USDT", "/USD", "/BTC", "/ETH", "/BNB", "/BUSD"]
    
    def __init__(
        self,
        binance_cache_ttl: int = 30,
        yf_cache_ttl: int = 60,
    ):
        self._binance = None
        self._yfinance = YFinanceDataFetcher(cache_ttl_seconds=yf_cache_ttl)
        self._yf_cache_ttl = yf_cache_ttl
        self._binance_cache_ttl = binance_cache_ttl
    
    def _get_binance(self):
        """Lazy init Binance fetcher."""
        if self._binance is None:
            from data_fetcher import DataFetcher
            self._binance = DataFetcher(
                exchange_id='binance',
                candles_per_fetch=500,
                cache_ttl_seconds=self._binance_cache_ttl,
            )
        return self._binance
    
    def _is_crypto(self, symbol: str) -> bool:
        """Czy to krypto (dla Binance)?"""
        # Check known crypto patterns
        for pattern in self.CRYPTO_PATTERNS:
            if pattern in symbol.upper():
                return True
        
        # Check known index aliases
        if symbol.upper() in INDEX_TICKERS:
            return False
        
        # Check if it's a YF-style ticker (contains - or .)
        if "-" in symbol or "." in symbol:
            return False
        
        # Default: jeśli jest / to krypto, inaczej stock
        return "/" in symbol
    
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Pobierz dane — auto-wybierz źródło."""
        if self._is_crypto(symbol):
            fetcher = self._get_binance()
        else:
            fetcher = self._yfinance
        
        return fetcher.fetch_ohlcv(symbol, timeframe, force_refresh=force_refresh)

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Pobierz aktualną cenę — deleguj do odpowiedniego backendu."""
        if self._is_crypto(symbol):
            fetcher = self._get_binance()
            if hasattr(fetcher, 'get_latest_price'):
                return fetcher.get_latest_price(symbol)
            return None
        else:
            return self._yfinance.get_latest_price(symbol)

    @property
    def stats(self) -> dict:
        result = {"mode": "unified"}
        if self._binance:
            result["binance"] = self._binance.stats
        result["yfinance"] = self._yfinance.stats
        return result
