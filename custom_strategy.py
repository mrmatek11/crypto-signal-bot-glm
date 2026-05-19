"""
Custom Strategy v3: NWO (filtr) + Stoch(7,3,2) (trigger) + CVD (potwierdzenie)
═══════════════════════════════════════════════════════════════════════════════

v3 CHANGES vs v2:
  1. MULTI-LEVEL signal hierarchy (4 poziomy zamiast 2):
     ⚡ CONFLUENCE  → Stoch(relaxed) + NWO + CVD (najsilniejszy)
     📊 STOCH+NWO   → Stoch(relaxed) + NWO histogram (główny trigger + filtr)
     🔵 STOCH-ONLY  → Stoch zone (K wchodzi/wychodzi z OB/OS) (najsłabszy)
     🟡 STOCH STRICT → Stoch strict (K<20/K>80) + NWO (oryginalny v2)
  2. RELAXED Stochastic thresholds: K<30/K>70 (zamiast K<20/K>80)
  3. Sentiment filter OPCJONALNY — wywoływany TYLKO gdy use_sentiment=True
  4. CVD threshold ±0.5 dla STOCH+NWO level (zamiast ±1.0)

v2 FIXES (zachowane):
  - NWO filter = HISTOGRAM direction (spójne z backtest.py!)
  - Trend filter = ALERT mode (default)
  - SL/TP = 3x ATR / 4.5x ATR

Hierarchia sygnałów (od najsilniejszego):
  ⚡ CONFLUENCE   → Stoch(relaxed) + NWO histogram + CVD (wszystko się zgadza)
  📊 STOCH+NWO    → Stoch(relaxed) + NWO histogram (główny trigger + filtr)
  🔵 STOCH STRICT → Stoch(strict, K<20/80) + NWO (oryginalny v2, wysoka pewność)
  🟡 STOCH-ONLY   → Stoch zone signal (K wchodzi/wychodzi z OB/OS, brak NWO/CVD)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import List, Optional, Dict

from signal_detector import Signal
from neural_weight_oscillator import NeuralWeightOscillator, NWOConfig, calc_atr, ema

# Global sentiment engine (lazy init)
_sentiment_engine = None
_use_sentiment = False  # Domyślnie OFF — włącza config.use_sentiment


def get_sentiment_engine():
    """Get or create the global SentimentEngine."""
    global _sentiment_engine
    if _sentiment_engine is None:
        from news_sentiment import SentimentEngine
        _sentiment_engine = SentimentEngine()
    return _sentiment_engine


def set_sentiment_enabled(enabled: bool):
    """Enable or disable sentiment filter (called from bot.py based on config)."""
    global _use_sentiment
    _use_sentiment = enabled


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY v3: Multi-level NWO histogram filter + Stoch trigger + CVD confirm
# ═══════════════════════════════════════════════════════════════════════════════

# Per-symbol NWO instances
_nwo_instances: Dict[str, NeuralWeightOscillator] = {}
_nwo_config: Optional[NWOConfig] = None

# SL/TP configuration (spójne z backtest.py BacktestConfig)
SL_ATR_MULT = 3.0
TP_ATR_MULT = 4.5

# Trend filter mode:
#   "alert" = alertuj z ostrzeżeniem ⚠️ (default)
#   "block" = całkowita blokada — nie alertuj w ogóle
#   "off"   = brak filtra trendu
TREND_FILTER_MODE = "alert"


def get_nwo_instance(symbol: str, timeframe: str, config: NWOConfig = None) -> NeuralWeightOscillator:
    """Get or create persistent NWO instance PER symbol+timeframe."""
    global _nwo_config
    if config is None:
        config = NWOConfig()
    
    key = f"{symbol}_{timeframe}"
    
    if _nwo_config != config:
        _nwo_instances.clear()
        _nwo_config = config
    
    if key not in _nwo_instances:
        _nwo_instances[key] = NeuralWeightOscillator(config)
    
    return _nwo_instances[key]


def strategy_nwo_stoch_cvd(df: pd.DataFrame, symbol: str, timeframe: str) -> List[Signal]:
    """
    Strategia v3: NWO histogram jako filtr + Stoch jako trigger + CVD jako potwierdzenie.
    
    Hierarchia sygnałów (od najsilniejszego):
      ⚡ CONFLUENCE   → Stoch(relaxed) + NWO + CVD all align
      📊 STOCH+NWO    → Stoch(relaxed) + NWO histogram align  
      🔵 STOCH STRICT → Stoch(strict K<20/80) + NWO (oryginalny v2)
      🟡 STOCH-ONLY   → Stoch zone signal (K enters/exits OB/OS)
    """
    signals = []
    
    if len(df) < 120:
        return signals
    
    # Inicjalizuj NWO per-symbol
    nwo = get_nwo_instance(symbol, timeframe)
    result = nwo.compute(df)
    
    i = len(df) - 1
    
    # Sprawdź czy mamy poprawne dane
    osc_val = result["osc"].iloc[i]
    if pd.isna(osc_val):
        return signals
    
    current_price = df['close'].iloc[i]
    stoch_k = result["stoch_k"].iloc[i]
    stoch_d = result["stoch_d"].iloc[i]
    cvd_val = result["cvd"].iloc[i]
    histogram_val = result["histogram"].iloc[i]
    
    # Oblicz ATR dla SL/TP
    atr_series = calc_atr(df['high'], df['low'], df['close'], 14)
    current_atr = atr_series.iloc[i] if not pd.isna(atr_series.iloc[i]) else 0
    
    if pd.isna(stoch_k) or pd.isna(cvd_val) or pd.isna(histogram_val):
        return signals
    
    # ─── Trend Filter: EMA20 vs EMA100 ────────────────────────────────
    ema20 = ema(df['close'], 20)
    ema100 = ema(df['close'], 100)
    uptrend = ema20.iloc[i] > ema100.iloc[i] if not pd.isna(ema20.iloc[i]) and not pd.isna(ema100.iloc[i]) else None
    downtrend = ema20.iloc[i] < ema100.iloc[i] if uptrend is not None else None
    
    # ─── NWO Direction Filter (histogram-based) ─────────────────────
    nwo_bullish = histogram_val > 0
    nwo_bearish = histogram_val < 0
    
    # ─── CVD Confirmation ──────────────────────────────────────────────
    cvd_bull_confirm_strict = cvd_val > 1.0
    cvd_bear_confirm_strict = cvd_val < -1.0
    cvd_bull_confirm_relaxed = cvd_val > 0.5   # v3: niższy threshold dla STOCH+NWO
    cvd_bear_confirm_relaxed = cvd_val < -0.5
    
    # ─── Stoch signal levels ──────────────────────────────────────────
    stoch_bull_strict = bool(result["stoch_bull"].iloc[i])          # K cross D + K<20 (oryginalny)
    stoch_bear_strict = bool(result["stoch_bear"].iloc[i])          # K cross D + K>80 (oryginalny)
    stoch_bull_relaxed = bool(result["stoch_bull_relaxed"].iloc[i]) # K cross D + K<30 (v3)
    stoch_bear_relaxed = bool(result["stoch_bear_relaxed"].iloc[i]) # K cross D + K>70 (v3)
    stoch_bull_zone = bool(result["stoch_bull_zone"].iloc[i])       # K enters/exits OB zone (v3)
    stoch_bear_zone = bool(result["stoch_bear_zone"].iloc[i])       # K enters/exits OS zone (v3)
    
    # ─── Pre-compute SL/TP ────────────────────────────────────────────
    if current_atr > 0:
        sl_long = current_price - current_atr * SL_ATR_MULT
        tp_long = current_price + current_atr * TP_ATR_MULT
        sl_short = current_price + current_atr * SL_ATR_MULT
        tp_short = current_price - current_atr * TP_ATR_MULT
    else:
        sl_long = tp_long = sl_short = tp_short = 0
    
    # ═════════════════════════════════════════════════════════════════════
    # LONG SIGNALS (hierarchia — najsilniejszy sygnał wygrywa)
    # ═════════════════════════════════════════════════════════════════════
    
    go_long = False
    source = ""
    reason = ""
    confidence = "MEDIUM"  # confidence level for display
    
    # Priority 1: CONFLUENCE (Stoch relaxed + NWO + CVD all align) — NAJSILNIEJSZY
    if stoch_bull_relaxed and nwo_bullish and cvd_bull_confirm_relaxed:
        go_long = True
        source = "CONFLUENCE"
        confidence = "HIGH"
        reason = (f"CONFLUENCE LONG: Stoch({stoch_k:.1f}) x D({stoch_d:.1f}) | "
                 f"NWO hist={histogram_val:.2f}(>0) | CVD={cvd_val:.2f}(>+0.5)")
    
    # Priority 2: STOCH+NWO (Stoch relaxed + NWO filter) — GŁÓWNY TRIGGER
    elif stoch_bull_relaxed and nwo_bullish:
        go_long = True
        source = "STOCH+NWO"
        confidence = "MEDIUM"
        reason = (f"STOCH+NWO LONG: Stoch({stoch_k:.1f}) x D({stoch_d:.1f}) | "
                 f"NWO hist={histogram_val:.2f}(>0 bullish)")
    
    # Priority 3: STOCH STRICT + NWO (oryginalny v2 — wysoka pewność, rzadszy)
    elif stoch_bull_strict and nwo_bullish:
        go_long = True
        source = "STOCH STRICT+NWO"
        confidence = "HIGH"
        reason = (f"STOCH STRICT+NWO LONG: Stoch({stoch_k:.1f}) x D({stoch_d:.1f}) K<20 | "
                 f"NWO hist={histogram_val:.2f}(>0)")
    
    # Priority 4: STOCH-ONLY (zone signal — najsłabszy, ale najczęstszy)
    elif stoch_bull_zone:
        go_long = True
        source = "STOCH-ONLY"
        confidence = "LOW"
        reason = (f"STOCH-ONLY LONG: Stoch({stoch_k:.1f}) D({stoch_d:.1f}) oversold zone | "
                 f"NWO hist={histogram_val:.2f} | CVD={cvd_val:+.2f}")
    
    # Trend filter
    long_against_trend = go_long and downtrend
    if long_against_trend:
        if TREND_FILTER_MODE == "block":
            go_long = False
            reason += " [BLOCKED: downtrend]"
        elif TREND_FILTER_MODE == "alert":
            reason += " | ⚠️ AGAINST DOWNTREND — AVOID"
    
    # Trend annotation
    if go_long:
        if uptrend:
            reason += " | Trend: UP ✅"
    
    if go_long:
        signals.append(Signal(
            symbol=symbol, timeframe=timeframe, signal_type="LONG",
            strategy_name=source,
            reason=reason,
            price=current_price,
            k_value=round(stoch_k, 2), d_value=round(stoch_d, 2),
            timestamp=df.index[i] if isinstance(df.index[i], datetime) else pd.Timestamp(df.index[i]).to_pydatetime(),
            extra_data={
                "atr": round(current_atr, 2),
                "osc": round(osc_val, 2),
                "cvd": round(cvd_val, 2),
                "histogram": round(histogram_val, 4),
                "source": source,
                "confidence": confidence,
                "trend": "UP" if uptrend else ("DOWN" if downtrend else "?"),
                "sl": round(sl_long, 2),
                "tp": round(tp_long, 2),
                "sl_atr_mult": SL_ATR_MULT,
                "tp_atr_mult": TP_ATR_MULT,
                "against_trend": long_against_trend,
                "risk_level": "HIGH" if long_against_trend else ("MEDIUM" if confidence == "LOW" else "NORMAL"),
            }
        ))
    
    # ═════════════════════════════════════════════════════════════════════
    # SHORT SIGNALS (hierarchia — najsilniejszy sygnał wygrywa)
    # ═════════════════════════════════════════════════════════════════════
    
    go_short = False
    source = ""
    reason = ""
    confidence = "MEDIUM"
    
    # Priority 1: CONFLUENCE (Stoch relaxed + NWO + CVD all align) — NAJSILNIEJSZY
    if stoch_bear_relaxed and nwo_bearish and cvd_bear_confirm_relaxed:
        go_short = True
        source = "CONFLUENCE"
        confidence = "HIGH"
        reason = (f"CONFLUENCE SHORT: Stoch({stoch_k:.1f}) x D({stoch_d:.1f}) | "
                 f"NWO hist={histogram_val:.2f}(<0) | CVD={cvd_val:.2f}(<-0.5)")
    
    # Priority 2: STOCH+NWO (Stoch relaxed + NWO filter) — GŁÓWNY TRIGGER
    elif stoch_bear_relaxed and nwo_bearish:
        go_short = True
        source = "STOCH+NWO"
        confidence = "MEDIUM"
        reason = (f"STOCH+NWO SHORT: Stoch({stoch_k:.1f}) x D({stoch_d:.1f}) | "
                 f"NWO hist={histogram_val:.2f}(<0 bearish)")
    
    # Priority 3: STOCH STRICT + NWO (oryginalny v2 — wysoka pewność, rzadszy)
    elif stoch_bear_strict and nwo_bearish:
        go_short = True
        source = "STOCH STRICT+NWO"
        confidence = "HIGH"
        reason = (f"STOCH STRICT+NWO SHORT: Stoch({stoch_k:.1f}) x D({stoch_d:.1f}) K>80 | "
                 f"NWO hist={histogram_val:.2f}(<0)")
    
    # Priority 4: STOCH-ONLY (zone signal — najsłabszy, ale najczęstszy)
    elif stoch_bear_zone:
        go_short = True
        source = "STOCH-ONLY"
        confidence = "LOW"
        reason = (f"STOCH-ONLY SHORT: Stoch({stoch_k:.1f}) D({stoch_d:.1f}) overbought zone | "
                 f"NWO hist={histogram_val:.2f} | CVD={cvd_val:+.2f}")
    
    # Trend filter
    short_against_trend = go_short and uptrend
    if short_against_trend:
        if TREND_FILTER_MODE == "block":
            go_short = False
            reason += " [BLOCKED: uptrend]"
        elif TREND_FILTER_MODE == "alert":
            reason += " | ⚠️ AGAINST UPTREND — AVOID"
    
    # Trend annotation
    if go_short:
        if downtrend:
            reason += " | Trend: DOWN ✅"
    
    if go_short:
        signals.append(Signal(
            symbol=symbol, timeframe=timeframe, signal_type="SHORT",
            strategy_name=source,
            reason=reason,
            price=current_price,
            k_value=round(stoch_k, 2), d_value=round(stoch_d, 2),
            timestamp=df.index[i] if isinstance(df.index[i], datetime) else pd.Timestamp(df.index[i]).to_pydatetime(),
            extra_data={
                "atr": round(current_atr, 2),
                "osc": round(osc_val, 2),
                "cvd": round(cvd_val, 2),
                "histogram": round(histogram_val, 4),
                "source": source,
                "confidence": confidence,
                "trend": "UP" if uptrend else ("DOWN" if downtrend else "?"),
                "sl": round(sl_short, 2),
                "tp": round(tp_short, 2),
                "sl_atr_mult": SL_ATR_MULT,
                "tp_atr_mult": TP_ATR_MULT,
                "against_trend": short_against_trend,
                "risk_level": "HIGH" if short_against_trend else ("MEDIUM" if confidence == "LOW" else "NORMAL"),
            }
        ))
    
    # ─── Sentiment Filter (OPCJONALNY — TYLKO gdy use_sentiment=True) ─
    if _use_sentiment:
        try:
            engine = get_sentiment_engine()
            sentiment = engine.get_sentiment(symbol)
            
            filtered_signals = []
            for sig in signals:
                should_block, sentiment_reason = engine.should_filter_signal(sig.signal_type, sentiment)
                if should_block:
                    sig.reason += f" [SENTIMENT BLOCK: {sentiment_reason}]"
                else:
                    sig.extra_data["sentiment_score"] = round(sentiment.score, 2)
                    sig.extra_data["sentiment_label"] = sentiment.label
                    sig.extra_data["sentiment_summary"] = sentiment.summary
                    if "caution" in sentiment_reason.lower():
                        sig.reason += f" | 📰 {sentiment_reason}"
                    filtered_signals.append(sig)
            
            signals = filtered_signals
        except Exception:
            # Sentiment failure nie powinien blokować sygnałów
            pass
    
    return signals


def get_current_nwo_state(df: pd.DataFrame, symbol: str = "", timeframe: str = "") -> Optional[dict]:
    """Get current NWO + Stoch + CVD state for status display."""
    nwo = get_nwo_instance(symbol, timeframe)
    result = nwo.compute(df)
    i = len(df) - 1
    
    osc_val = result["osc"].iloc[i]
    if pd.isna(osc_val):
        return None
    
    stoch_k = result["stoch_k"].iloc[i]
    stoch_d = result["stoch_d"].iloc[i]
    cvd_val = result["cvd"].iloc[i]
    histogram_val = result["histogram"].iloc[i]
    
    # Oblicz ATR
    atr_series = calc_atr(df['high'], df['low'], df['close'], 14)
    current_atr = atr_series.iloc[i] if not pd.isna(atr_series.iloc[i]) else None
    
    # Trend
    ema20 = ema(df['close'], 20)
    ema100 = ema(df['close'], 100)
    trend = "UP" if ema20.iloc[i] > ema100.iloc[i] else "DOWN"
    
    # NWO direction
    nwo_dir = "BULLISH" if (not pd.isna(histogram_val) and histogram_val > 0) else ("BEARISH" if not pd.isna(histogram_val) else "?")
    
    # Determine zone
    if osc_val < 20:
        zone = "oversold_extreme"
    elif osc_val < 30:
        zone = "oversold"
    elif osc_val < 70:
        zone = "neutral"
    elif osc_val < 80:
        zone = "overbought"
    else:
        zone = "overbought_extreme"
    
    return {
        "price": df['close'].iloc[i],
        "osc": round(osc_val, 2),
        "signal_line": round(result["signal_line"].iloc[i], 2) if not pd.isna(result["signal_line"].iloc[i]) else None,
        "histogram": round(histogram_val, 4) if not pd.isna(histogram_val) else None,
        "stoch_k": round(stoch_k, 2) if not pd.isna(stoch_k) else None,
        "stoch_d": round(stoch_d, 2) if not pd.isna(stoch_d) else None,
        "cvd": round(cvd_val, 2) if not pd.isna(cvd_val) else None,
        "atr": round(current_atr, 2) if current_atr is not None else None,
        "zone": zone,
        "trend": trend,
        "nwo_direction": nwo_dir,
        "bwm_trend": round(result["bwm_trend"], 4),
        "bwm_mean": round(result["bwm_mean"], 4),
        "bwm_momentum": round(result["bwm_momentum"], 4),
        "training_loss": result["last_loss"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

STRATEGY_REGISTRY = {
    "stoch_7_3_2": {
        "name": "Stochastic (7,3,2)",
        "description": "Prosty Stochastic (7,3,2) — oversold/overbought crossover",
        "fn": None,  # Używa SignalDetector (domyślna)
    },
    "nwo_stoch_cvd": {
        "name": "NWO(filtr) + Stoch(trigger) + CVD v3",
        "description": "v3: Multi-level signals (CONFLUENCE→STOCH+NWO→STOCH STRICT→STOCH-ONLY) + relaxed thresholds + optional sentiment",
        "fn": strategy_nwo_stoch_cvd,
    },
    "custom_pinescript": {
        "name": "Custom PineScript",
        "description": "Template pod Twój własny skrypt PineScript",
        "fn": None,  # Placeholder
    },
}
