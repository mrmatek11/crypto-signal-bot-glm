"""
AI Analyst Module — GLM-powered analysis for trading signals
═══════════════════════════════════════════════════════════════

Komponenty:
  1. SignalQualityScorer — ocenia jakość sygnału (1-10) z kontekstem
  2. DailyMarketBriefing — poranny raport AI (bias, levels, watchlist)
  3. MarketRegimeDetector — klasyfikuje rynek (trend/range/volatile)
  4. MultiTFConfluence — analiza konfluencji na wielu TF
  5. EndOfDaySummary — wieczorny raport wyników

Używa GLM API (ZhipuAI) — glm-4-flash (fast/cheap) lub glm-4-plus (quality)

Użycie:
  from ai_analyst import AIAnalyst
  analyst = AIAnalyst(api_key="your-glm-key")
  quality = analyst.score_signal(signal, indicator_context)
  briefing = analyst.daily_briefing(market_data)
"""

import os
import time
import json
import requests
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# GLM API CLIENT
# ═══════════════════════════════════════════════════════════════════════════════

class GLMClient:
    """
    Klient GLM API (ZhipuAI).
    
    Modele:
      - glm-4-flash: szybki, tani, dobry do prostych analiz
      - glm-4-plus: wolniejszy, lepszy, do złożonych analiz
      - glm-4-long: bardzo długi kontekst, do obszernych raportów
    
    Endpoint: https://open.bigmodel.cn/api/paas/v4/chat/completions
    """
    
    API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    
    def __init__(self, api_key: str = "", model: str = "glm-4-flash"):
        self.api_key = api_key or os.getenv("GLM_API_KEY", "")
        self.model = model
        self._last_call_time = 0
        self._min_interval = 0.5  # min 0.5s między wywołaniami
        self._call_count = 0
        self._error_count = 0
    
    def chat(self, system_prompt: str, user_prompt: str, 
             temperature: float = 0.1, max_tokens: int = 500,
             json_mode: bool = False) -> Optional[str]:
        """
        Wyślij zapytanie do GLM API.
        
        Returns:
            Odpowiedź tekstowa lub None jeśli błąd.
        """
        if not self.api_key:
            logger.warning("GLM API key nie ustawiony!")
            return None
        
        # Rate limiting
        now = time.time()
        elapsed = now - self._last_call_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        
        try:
            resp = requests.post(
                self.API_URL,
                headers=headers,
                json=payload,
                timeout=30,
            )
            
            self._last_call_time = time.time()
            self._call_count += 1
            
            if resp.status_code == 200:
                data = resp.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                return content.strip()
            elif resp.status_code == 429:
                # Rate limited
                retry_after = float(resp.headers.get("Retry-After", "2"))
                logger.warning(f"GLM rate limited, waiting {retry_after}s")
                time.sleep(retry_after)
                return self.chat(system_prompt, user_prompt, temperature, max_tokens, json_mode)
            else:
                self._error_count += 1
                logger.warning(f"GLM API error {resp.status_code}: {resp.text[:200]}")
                return None
                
        except requests.exceptions.Timeout:
            self._error_count += 1
            logger.warning("GLM API timeout")
            return None
        except Exception as e:
            self._error_count += 1
            logger.warning(f"GLM API error: {e}")
            return None
    
    def chat_json(self, system_prompt: str, user_prompt: str, 
                  temperature: float = 0.1, max_tokens: int = 500) -> Optional[dict]:
        """
        Wyślij zapytanie i sparsuj odpowiedź jako JSON.
        Automatycznie wyciąga JSON z odpowiedzi (nawet w markdown code blocks).
        """
        response = self.chat(system_prompt, user_prompt, temperature, max_tokens, json_mode=True)
        if response is None:
            return None
        
        # Próba bezpośredniego parsowania
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Wyciągnij JSON z markdown code blocks
        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            else:
                # Szukaj pierwszego { i ostatniego }
                start = response.find("{")
                end = response.rfind("}") + 1
                if start >= 0 and end > start:
                    json_str = response[start:end]
                else:
                    return None
            return json.loads(json_str.strip())
        except (json.JSONDecodeError, IndexError):
            return None
    
    @property
    def stats(self) -> dict:
        return {
            "calls": self._call_count,
            "errors": self._error_count,
            "model": self.model,
            "has_key": bool(self.api_key),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SignalQuality:
    """Wynik oceny jakości sygnału przez AI."""
    score: int              # 1-10 (10 = najlepszy)
    confidence: str         # HIGH, MEDIUM, LOW
    analysis: str           # Krótka analiza (1-2 zdania)
    key_factors: List[str]  # Co za sygnałem przemawia
    risks: List[str]        # Zagrożenia
    recommendation: str     # np. "TAKE", "WATCH", "SKIP"
    
    @property
    def emoji(self) -> str:
        if self.score >= 8:
            return "🟢🔥"
        elif self.score >= 6:
            return "🟢"
        elif self.score >= 4:
            return "🟡"
        else:
            return "🔴"


@dataclass
class MarketRegime:
    """Klasyfikacja obecnego reżimu rynkowego."""
    regime: str         # "trending_up", "trending_down", "ranging", "volatile", "quiet"
    strength: str       # "strong", "moderate", "weak"
    confidence: float   # 0.0-1.0
    description: str    # Opis w 1 zdaniu
    bias: str           # "bullish", "bearish", "neutral"
    
    @property
    def emoji(self) -> str:
        return {
            "trending_up": "📈",
            "trending_down": "📉",
            "ranging": "↔️",
            "volatile": "⚡",
            "quiet": "😴",
        }.get(self.regime, "❓")


@dataclass
class DailyBriefing:
    """Poranny raport rynkowy."""
    overall_bias: str            # "bullish", "bearish", "neutral", "mixed"
    key_pairs: List[dict]        # [{"symbol": "BTC/USDT", "bias": "bullish", "comment": "..."}]
    risk_events: List[str]       # Wydarzenia ryzyka
    watchlist: List[str]         # Pary do obserwacji
    summary: str                 # Podsumowanie dnia
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


# ═══════════════════════════════════════════════════════════════════════════════
# AI ANALYST — główna klasa
# ═══════════════════════════════════════════════════════════════════════════════

class AIAnalyst:
    """
    AI Analyst — GLM-powered trading signal analysis.
    
    Features:
      - Signal quality scoring (1-10)
      - Market regime detection
      - Daily market briefing
      - Multi-TF confluence analysis
      - End-of-day summary
    """
    
    # Cache dla regime detection (nie odpytuj AI co minutę)
    _regime_cache: Dict[str, Tuple[MarketRegime, float]] = {}
    _regime_cache_ttl = 900  # 15 minut
    
    def __init__(
        self,
        api_key: str = "",
        model: str = "glm-4-flash",
        enabled: bool = True,
        signal_scoring: bool = True,
        daily_briefing: bool = True,
        regime_detection: bool = True,
    ):
        self.client = GLMClient(api_key=api_key, model=model)
        self.enabled = enabled and bool(api_key)
        self.signal_scoring = signal_scoring
        self.daily_briefing = daily_briefing
        self.regime_detection = regime_detection
        
        if self.enabled:
            logger.info(f"🧠 AI Analyst: ENABLED (model={model})")
            logger.info(f"   Signal scoring: {'✅' if signal_scoring else '❌'}")
            logger.info(f"   Daily briefing: {'✅' if daily_briefing else '❌'}")
            logger.info(f"   Regime detection: {'✅' if regime_detection else '❌'}")
        else:
            logger.info("🧠 AI Analyst: DISABLED (no API key)")
    
    # ─── 1. SIGNAL QUALITY SCORER ──────────────────────────────────────
    
    def score_signal(self, signal_data: dict, indicator_context: dict) -> Optional[SignalQuality]:
        """
        Oceń jakość sygnału tradingowego przez GLM.
        
        Args:
            signal_data: dict z polami signal_type, symbol, timeframe, price, k_value, d_value, source, confidence
            indicator_context: dict z osc, histogram, cvd, atr, trend, stoch_k, stoch_d, nwo_direction
        
        Returns:
            SignalQuality lub None jeśli błąd
        """
        if not self.enabled or not self.signal_scoring:
            return None
        
        system_prompt = """You are a professional crypto trading signal analyst. Evaluate trading signals based on technical indicator confluence. Respond ONLY with valid JSON."""

        user_prompt = f"""Analyze this crypto trading signal and rate its quality.

SIGNAL:
- Direction: {signal_data.get('signal_type', '?')} on {signal_data.get('symbol', '?')} ({signal_data.get('timeframe', '?')})
- Price: ${signal_data.get('price', 0):,.2f}
- Source: {signal_data.get('source', '?')} (signal trigger level)
- Confidence: {signal_data.get('confidence', '?')}

INDICATOR CONTEXT:
- Stochastic K: {indicator_context.get('stoch_k', '?')} / D: {indicator_context.get('stoch_d', '?')}
- NWO Oscillator: {indicator_context.get('osc', '?')}
- NWO Histogram: {indicator_context.get('histogram', '?')} (direction: {indicator_context.get('nwo_direction', '?')})
- CVD z-score: {indicator_context.get('cvd', '?')}
- ATR: {indicator_context.get('atr', '?')}
- Trend: {indicator_context.get('trend', '?')}
- Against trend: {indicator_context.get('against_trend', False)}
- SL: ${signal_data.get('sl', 0):,.2f} | TP: ${signal_data.get('tp', 0):,.2f}

Respond with JSON:
{{
  "score": <integer 1-10>,
  "confidence": "<HIGH|MEDIUM|LOW>",
  "analysis": "<1-2 sentence analysis>",
  "key_factors": ["<factor1>", "<factor2>"],
  "risks": ["<risk1>", "<risk2>"],
  "recommendation": "<TAKE|WATCH|SKIP>"
}}

Scoring guide:
- 8-10: Strong confluence, multiple indicators align, with trend → TAKE
- 5-7: Decent signal but some mixed signals → WATCH
- 1-4: Weak signal, against trend, conflicting indicators → SKIP"""

        result = self.client.chat_json(system_prompt, user_prompt, temperature=0.1, max_tokens=300)
        
        if result is None:
            return None
        
        try:
            return SignalQuality(
                score=max(1, min(10, int(result.get("score", 5)))),
                confidence=result.get("confidence", "MEDIUM"),
                analysis=result.get("analysis", ""),
                key_factors=result.get("key_factors", []),
                risks=result.get("risks", []),
                recommendation=result.get("recommendation", "WATCH"),
            )
        except Exception as e:
            logger.warning(f"Signal scoring parse error: {e}")
            return None
    
    # ─── 2. MARKET REGIME DETECTION ────────────────────────────────────
    
    def detect_regime(self, symbol: str, market_data: dict) -> Optional[MarketRegime]:
        """
        Wykryj obecny reżim rynkowy przez GLM.
        
        Args:
            symbol: np. "BTC/USDT"
            market_data: dict z timeframe'ami jako kluczami, wartości to dict z indicator context
        
        Returns:
            MarketRegime lub None
        """
        if not self.enabled or not self.regime_detection:
            return None
        
        # Check cache
        cache_key = symbol
        if cache_key in self._regime_cache:
            cached_regime, cached_time = self._regime_cache[cache_key]
            if time.time() - cached_time < self._regime_cache_ttl:
                return cached_regime
        
        system_prompt = """You are a market regime classifier. Classify the current market state based on multi-timeframe technical indicators. Respond ONLY with valid JSON."""
        
        # Build multi-TF context
        tf_context_lines = []
        for tf, data in market_data.items():
            tf_context_lines.append(
                f"  {tf}: K={data.get('stoch_k', '?')} D={data.get('stoch_d', '?')} "
                f"osc={data.get('osc', '?')} hist={data.get('histogram', '?')} "
                f"cvd={data.get('cvd', '?')} trend={data.get('trend', '?')} "
                f"nwo={data.get('nwo_direction', '?')}"
            )
        tf_context = "\n".join(tf_context_lines)
        
        user_prompt = f"""Classify the current market regime for {symbol}.

MULTI-TIMEFRAME DATA:
{tf_context}

Respond with JSON:
{{
  "regime": "<trending_up|trending_down|ranging|volatile|quiet>",
  "strength": "<strong|moderate|weak>",
  "confidence": <float 0.0-1.0>,
  "description": "<1 sentence description>",
  "bias": "<bullish|bearish|neutral>"
}}

Classification guide:
- trending_up: EMAs aligned up, NWO bullish, CVD positive across TFs
- trending_down: EMAs aligned down, NWO bearish, CVD negative across TFs
- ranging: Mixed signals, oscillators around 50, no clear direction
- volatile: High ATR, conflicting signals, rapid reversals
- quiet: Low ATR, compressing ranges, potential breakout coming"""

        result = self.client.chat_json(system_prompt, user_prompt, temperature=0.1, max_tokens=200)
        
        if result is None:
            return None
        
        try:
            regime = MarketRegime(
                regime=result.get("regime", "ranging"),
                strength=result.get("strength", "moderate"),
                confidence=float(result.get("confidence", 0.5)),
                description=result.get("description", ""),
                bias=result.get("bias", "neutral"),
            )
            
            # Cache
            self._regime_cache[cache_key] = (regime, time.time())
            return regime
            
        except Exception as e:
            logger.warning(f"Regime detection parse error: {e}")
            return None
    
    # ─── 3. DAILY MARKET BRIEFING ──────────────────────────────────────
    
    def generate_daily_briefing(self, market_snapshot: dict) -> Optional[DailyBriefing]:
        """
        Generuj poranny raport rynkowy przez GLM.
        
        Args:
            market_snapshot: dict z {'pairs': [{'symbol': ..., 'price': ..., 'trend': ..., 'osc': ..., ...}], 'regimes': {...}}
        
        Returns:
            DailyBriefing lub None
        """
        if not self.enabled or not self.daily_briefing:
            return None
        
        system_prompt = """You are a professional crypto market analyst. Generate concise, actionable daily market briefings. Respond ONLY with valid JSON."""
        
        # Build market context
        pairs_lines = []
        for pair in market_snapshot.get("pairs", [])[:10]:
            pairs_lines.append(
                f"  {pair.get('symbol', '?')}: ${pair.get('price', 0):,.2f} "
                f"trend={pair.get('trend', '?')} osc={pair.get('osc', '?')} "
                f"K={pair.get('stoch_k', '?')} D={pair.get('stoch_d', '?')} "
                f"CVD={pair.get('cvd', '?')} NWO={pair.get('nwo_direction', '?')}"
            )
        pairs_text = "\n".join(pairs_lines)
        
        regimes_text = ""
        for sym, regime in market_snapshot.get("regimes", {}).items():
            if isinstance(regime, MarketRegime):
                regimes_text += f"  {sym}: {regime.regime} ({regime.strength}) bias={regime.bias}\n"
            elif isinstance(regime, dict):
                regimes_text += f"  {sym}: {regime.get('regime', '?')} bias={regime.get('bias', '?')}\n"
        
        user_prompt = f"""Generate a daily market briefing for crypto.

CURRENT MARKET DATA:
{pairs_text}

MARKET REGIMES:
{regimes_text if regimes_text else "Not available"}

Respond with JSON:
{{
  "overall_bias": "<bullish|bearish|neutral|mixed>",
  "key_pairs": [
    {{"symbol": "<pair>", "bias": "<bullish|bearish|neutral>", "comment": "<1 sentence>"}}
  ],
  "risk_events": ["<risk1>", "<risk2>"],
  "watchlist": ["<pair1>", "<pair2>"],
  "summary": "<2-3 sentence market overview>"
}}"""

        result = self.client.chat_json(system_prompt, user_prompt, temperature=0.3, max_tokens=500)
        
        if result is None:
            return None
        
        try:
            return DailyBriefing(
                overall_bias=result.get("overall_bias", "neutral"),
                key_pairs=result.get("key_pairs", []),
                risk_events=result.get("risk_events", []),
                watchlist=result.get("watchlist", []),
                summary=result.get("summary", ""),
            )
        except Exception as e:
            logger.warning(f"Daily briefing parse error: {e}")
            return None
    
    # ─── 4. MULTI-TF CONFLUENCE ANALYSIS ───────────────────────────────
    
    def analyze_confluence(self, symbol: str, multi_tf_data: dict) -> Optional[dict]:
        """
        Analizuj konfluencję sygnałów na wielu timeframe'ach.
        
        Args:
            symbol: np. "BTC/USDT"
            multi_tf_data: {tf: {signal_type, source, confidence, stoch_k, ...}, ...}
        
        Returns:
            {"confluence_score": int, "direction": str, "analysis": str, "best_tf": str}
        """
        if not self.enabled:
            return None
        
        system_prompt = """You are a multi-timeframe confluence analyst. Evaluate how signals align across timeframes. Respond ONLY with valid JSON."""
        
        tf_lines = []
        active_signals = []
        for tf, data in multi_tf_data.items():
            tf_lines.append(
                f"  {tf}: signal={data.get('signal_type', 'none')} source={data.get('source', 'none')} "
                f"K={data.get('stoch_k', '?')} D={data.get('stoch_d', '?')} "
                f"osc={data.get('osc', '?')} hist={data.get('histogram', '?')} "
                f"cvd={data.get('cvd', '?')} trend={data.get('trend', '?')}"
            )
            if data.get('signal_type') and data['signal_type'] != 'none':
                active_signals.append(f"{tf}:{data['signal_type']}")
        
        tf_text = "\n".join(tf_lines)
        active_text = ", ".join(active_signals) if active_signals else "none"
        
        user_prompt = f"""Analyze multi-timeframe confluence for {symbol}.

TIMEFRAME DATA:
{tf_text}

ACTIVE SIGNALS: {active_text}

Respond with JSON:
{{
  "confluence_score": <integer 1-10>,
  "direction": "<bullish|bearish|neutral|mixed>",
  "analysis": "<1-2 sentence analysis>",
  "best_tf": "<timeframe with strongest signal or 'none'>"
}}"""

        result = self.client.chat_json(system_prompt, user_prompt, temperature=0.1, max_tokens=200)
        return result
    
    # ─── 5. END-OF-DAY SUMMARY ─────────────────────────────────────────
    
    def generate_eod_summary(self, signals_sent: int, positions_data: dict, 
                              market_snapshot: dict) -> Optional[str]:
        """
        Generuj podsumowanie dnia.
        
        Args:
            signals_sent: ile sygnałów wysłano
            positions_data: {"total_trades": N, "win_rate": %, "pnl": $, "by_direction": {...}}
            market_snapshot: aktualny stan rynku
        
        Returns:
            Tekst podsumowania lub None
        """
        if not self.enabled:
            return None
        
        system_prompt = """You are a trading performance analyst. Write concise end-of-day summaries. Be direct and actionable."""
        
        user_prompt = f"""Write a brief end-of-day trading summary.

STATS:
- Signals sent today: {signals_sent}
- Positions: {json.dumps(positions_data) if positions_data else 'N/A'}

MARKET STATE:
{json.dumps(market_snapshot, default=str)[:500] if market_snapshot else 'N/A'}

Write a 3-4 sentence summary covering: performance, key observations, and one actionable tip for tomorrow. Do NOT use markdown or JSON, just plain text."""

        result = self.client.chat(system_prompt, user_prompt, temperature=0.3, max_tokens=300)
        return result
    
    @property
    def stats(self) -> dict:
        return {
            "enabled": self.enabled,
            "signal_scoring": self.signal_scoring,
            "daily_briefing": self.daily_briefing,
            "regime_detection": self.regime_detection,
            "glm_client": self.client.stats,
        }
