"""
Neural Weight Oscillator (Zeiierman) — Python Port
════════════════════════════════════════════════════

Pełne tłumaczenie PineScript → Python z dodatkami:
  - Stochastic (7,3,2) jako komponent sygnałowy
  - CVD (Cumulative Volume Delta) jako komponent sygnałowy
  - Integracja z Discord signal bot

Oryginał: © Zeiierman — Neural Weight Oscillator
Licencja: CC BY-NC-SA 4.0
"""

import pandas as pd
import numpy as np
import math
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from collections import deque


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS (PineScript odpowiedniki)
# ═══════════════════════════════════════════════════════════════════════════════

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(x, hi))


def normalize(x: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 50.0
    return clamp((x - lo) / (hi - lo) * 100.0, 0.0, 100.0)


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(window=length, min_periods=length).mean()


def rma(series: pd.Series, length: int) -> pd.Series:
    alpha = 1.0 / length
    return series.ewm(alpha=alpha, adjust=False, min_periods=length).mean()


def calc_rsi(source: pd.Series, length: int = 14) -> pd.Series:
    delta = source.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = rma(gain, length)
    avg_loss = rma(loss, length)
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return rma(true_range, length)


def calc_stdev(source: pd.Series, length: int) -> pd.Series:
    return source.rolling(window=length, min_periods=length).std(ddof=0)


def calc_stoch(high: pd.Series, low: pd.Series, close: pd.Series,
               k_length: int = 7, k_smooth: int = 3, d_smooth: int = 2) -> Tuple[pd.Series, pd.Series]:
    lowest_low = low.rolling(window=k_length, min_periods=k_length).min()
    highest_high = high.rolling(window=k_length, min_periods=k_length).max()
    k_raw = 100.0 * (close - lowest_low) / (highest_high - lowest_low)
    k_raw = k_raw.replace([np.inf, -np.inf], np.nan)
    k_line = k_raw.rolling(window=k_smooth, min_periods=k_smooth).mean()
    d_line = k_line.rolling(window=d_smooth, min_periods=d_smooth).mean()
    return k_line, d_line


def calc_cvd(close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series,
             length: int = 20) -> pd.Series:
    """
    Cumulative Volume Delta (approximated from OHLCV data).
    
    CVD ≈ cumulative sum of directional volume.
    Direction estimated via: if close > open → buy volume, else sell volume.
    More precise: uses (2*close - high - low) / (high - low) as directional multiplier.
    
    Returns normalized CVD z-score for oscillator integration.
    """
    # Directional volume estimation
    hl_range = high - low
    hl_range = hl_range.replace(0, np.nan)
    
    # Proximity of close to high vs low: +1 at high, -1 at low
    directional = (2.0 * close - high - low) / hl_range
    directional = directional.fillna(0)
    
    # Volume delta = directional * volume
    vol_delta = directional * volume
    
    # Cumulative sum
    cvd_raw = vol_delta.cumsum()
    
    # Normalize: z-score over rolling window
    cvd_mean = sma(cvd_raw, length)
    cvd_std = calc_stdev(cvd_raw, length)
    cvd_zscore = (cvd_raw - cvd_mean) / cvd_std.replace(0, np.nan)
    cvd_zscore = cvd_zscore.fillna(0)
    
    return cvd_zscore


def barssince(condition: pd.Series) -> pd.Series:
    """PineScript ta.barssince() — bars since last true condition."""
    result = pd.Series(np.nan, index=condition.index)
    count = np.nan
    for i in range(len(condition)):
        if condition.iloc[i]:
            count = 0
        elif not np.isnan(count):
            count += 1
        result.iloc[i] = count
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# BWM (Best-Worst Method) WEIGHT MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def criterion_index(name: str) -> int:
    return {"Trend": 0, "Mean Reversion": 1, "Momentum": 2}[name]


def bwm_solve(bo: List[float], ow: List[float], best_idx: int, worst_idx: int) -> List[float]:
    """
    BWM weight solver — PineScript bwmSolve() equivalent.
    Computes normalized weights from Best-to-Others and Relative-to-Worst comparisons.
    """
    a_bw = bo[worst_idx]
    weights = [0.0, 0.0, 0.0]
    
    for i in range(3):
        bo_val = bo[i]
        ow_val = ow[i]
        rel_weight = math.sqrt((a_bw / bo_val) * ow_val)
        weights[i] = rel_weight
    
    # Normalize
    total = sum(weights)
    if total > 0:
        weights = [w / total for w in weights]
    
    return weights


def bwm_objective(weights: List[float], bo: List[float], ow: List[float],
                  best_idx: int, worst_idx: int) -> float:
    """BWM consistency check — max error."""
    w_best = weights[best_idx]
    w_worst = weights[worst_idx]
    max_err = 0.0
    
    for i in range(3):
        wi = weights[i]
        bo_target = bo[i]
        ow_target = ow[i]
        
        bo_err = abs((w_best / wi) - bo_target) if wi != 0 else 0
        ow_err = abs((wi / w_worst) - ow_target) if w_worst != 0 else 0
        
        max_err = max(max_err, max(bo_err, ow_err))
    
    return max_err


# ═══════════════════════════════════════════════════════════════════════════════
# ADAM OPTIMIZER (for adaptive training)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AdamState:
    """State for Adam optimizer (momentum + velocity for each parameter)."""
    m: float = 0.0
    v: float = 0.0
    
    def update(self, weight: float, grad: float, lr: float = 0.01,
               beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8,
               step: int = 1) -> float:
        self.m = beta1 * self.m + (1.0 - beta1) * grad
        self.v = beta2 * self.v + (1.0 - beta2) * grad * grad
        m_hat = self.m / (1.0 - beta1 ** step)
        v_hat = self.v / (1.0 - beta2 ** step)
        return weight - lr * m_hat / (math.sqrt(v_hat) + eps)


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING SAMPLE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainSample:
    score: float
    trend: float
    mean: float
    momentum: float
    target: float
    idx: int = 0


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN OSCILLATOR CLASS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class NWOConfig:
    """Konfiguracja Neural Weight Oscillator."""
    # Core
    len_fast: int = 20
    len_slow: int = 100
    smooth_len: int = 5
    
    # BWM
    best_criterion: str = "Trend"
    worst_criterion: str = "Momentum"
    bo_trend: float = 1.0
    bo_mean: float = 3.0
    bo_momentum: float = 6.0
    ow_trend: float = 6.0
    ow_mean: float = 3.0
    ow_momentum: float = 1.0
    
    # Training
    use_training: bool = True
    learn_influence: float = 0.30
    line_influence: float = 0.25
    
    # Signal
    signal_len: int = 9
    hist_smooth: int = 3
    sweep_lookback: int = 10
    
    # Stochastic (7,3,2)
    stoch_k_length: int = 7
    stoch_k_smooth: int = 3
    stoch_d_smooth: int = 2
    stoch_oversold: float = 20.0
    stoch_overbought: float = 80.0
    
    # CVD
    cvd_length: int = 20
    cvd_threshold: float = 1.0  # z-score threshold for CVD signal (v2: lowered from 1.5)
    
    # Internal
    len_rsi: int = 14
    len_atr: int = 14
    len_momentum: int = 20
    warmup_bars: int = 100
    target_len: int = 3
    memory_size: int = 150
    batch_size: int = 20
    lr: float = 0.01
    huber_d: float = 0.01
    ai_center_len: int = 100


class NeuralWeightOscillator:
    """
    Neural Weight Oscillator (Zeiierman) — pełna implementacja Python.
    
    Komponenty:
      1. Trend — EMA spread + slope
      2. Mean Reversion — RSI exhaustion + z-score
      3. Momentum — ROC + RSI + EMA velocity
      4. Stochastic (7,3,2) — dodatkowy komponent
      5. CVD — Cumulative Volume Delta (dodatkowy komponent)
    
    System wag: BWM (Best-Worst Method)
    Warstwa adaptacyjna: Adam optimizer z Huber loss
    """
    
    def __init__(self, config: Optional[NWOConfig] = None):
        self.config = config or NWOConfig()
        
        # BWM weights (pre-computed, static)
        bo = [self.config.bo_trend, self.config.bo_mean, self.config.bo_momentum]
        ow = [self.config.ow_trend, self.config.ow_mean, self.config.ow_momentum]
        
        best_idx = criterion_index(self.config.best_criterion)
        worst_idx = criterion_index(self.config.worst_criterion)
        
        bo[best_idx] = 1.0
        ow[worst_idx] = 1.0
        
        self.bwm_weights = bwm_solve(bo, ow, best_idx, worst_idx)
        self.bwm_trend = self.bwm_weights[0]
        self.bwm_mean = self.bwm_weights[1]
        self.bwm_momentum = self.bwm_weights[2]
        self.bwm_error = bwm_objective(self.bwm_weights, bo, ow, best_idx, worst_idx)
        
        # Training state (persistent across calls — like PineScript var)
        self._memory: deque = deque(maxlen=self.config.memory_size)
        self._tw_trend = 0.01
        self._tw_mean = 0.01
        self._tw_momentum = 0.01
        self._t_bias = 0.0
        self._adam_trend = AdamState()
        self._adam_mean = AdamState()
        self._adam_momentum = AdamState()
        self._adam_bias = AdamState()
        self._step = 0
        self._last_loss = None
        
        # Cache for feature values (for training targets)
        self._prev_trend_features = deque(maxlen=self.config.target_len + 1)
        self._prev_mean_features = deque(maxlen=self.config.target_len + 1)
        self._prev_momentum_features = deque(maxlen=self.config.target_len + 1)
        self._prev_atr = deque(maxlen=self.config.target_len + 1)
        self._prev_close = deque(maxlen=self.config.target_len + 1)
    
    def compute(self, df: pd.DataFrame) -> Dict:
        """
        Oblicz pełny oscylator na DataFrame OHLCV.
        
        Returns dict with:
            osc, signal_line, histogram, bull_signal, bear_signal,
            stoch_k, stoch_d, cvd,
            trend_score, mean_score, momentum_score,
            bwm_weights, training_weights
        """
        cfg = self.config
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df.get('volume', pd.Series(0, index=df.index))
        
        n = len(df)
        if n < cfg.warmup_bars:
            return self._empty_result(n)
        
        # ─── Market Components ─────────────────────────────────────────────
        ema_fast = ema(close, cfg.len_fast)
        ema_slow = ema(close, cfg.len_slow)
        atr = calc_atr(high, low, close, cfg.len_atr)
        rsi = calc_rsi(close, cfg.len_rsi)
        
        # Trend
        trend_spread = (ema_fast - ema_slow) / atr
        trend_slope = (ema_fast - ema_fast.shift(1)) / atr
        trend_raw = trend_spread + trend_slope
        trend_score = trend_raw.apply(lambda x: normalize(x, -2.5, 2.5))
        
        # Mean Reversion
        basis = sma(close, cfg.len_momentum)
        dev = calc_stdev(close, cfg.len_momentum)
        z_score = pd.Series(
            np.where(dev == 0, 0, (close - basis) / dev),
            index=df.index
        )
        
        rsi_reversion = 100 - rsi
        z_reversion = (-z_score).apply(lambda x: normalize(x, -2.5, 2.5))
        mean_score = rsi_reversion * 0.5 + z_reversion * 0.5
        
        # Momentum
        roc = close / close.shift(cfg.len_momentum) - 1.0
        roc_norm = roc.apply(lambda x: normalize(x, -0.05, 0.05))
        
        rsi_momentum = rsi
        ema_momentum_raw = (ema_fast - ema_fast.shift(1)) / atr
        ema_momentum = ema_momentum_raw.apply(lambda x: normalize(x, -0.5, 0.5))
        
        momentum_score = roc_norm * 0.45 + rsi_momentum * 0.35 + ema_momentum * 0.20
        
        # Features (centered around 0)
        trend_feature = (trend_score - 50.0) / 50.0
        mean_feature = (mean_score - 50.0) / 50.0
        momentum_feature = (momentum_score - 50.0) / 50.0
        
        # ─── Stochastic (7,3,2) ────────────────────────────────────────────
        stoch_k, stoch_d = calc_stoch(high, low, close,
                                       cfg.stoch_k_length,
                                       cfg.stoch_k_smooth,
                                       cfg.stoch_d_smooth)
        
        # ─── CVD ───────────────────────────────────────────────────────────
        cvd = calc_cvd(close, high, low, volume, cfg.cvd_length)
        
        # ─── Training Model (bar-by-bar simulation) ────────────────────────
        # In PineScript, training happens on every bar with var state.
        # Here we simulate the bar-by-bar training loop.
        osc_values = np.full(n, np.nan)
        signal_values = np.full(n, np.nan)
        
        # Running EMA for oscillator smoothing and signal line
        raw_osc_ema = np.full(n, np.nan)
        
        # Pre-compute ai prediction components
        ai_pred_raw_full = np.full(n, np.nan)
        
        for i in range(n):
            # Store features for delayed training
            self._prev_trend_features.append(trend_feature.iloc[i] if not pd.isna(trend_feature.iloc[i]) else 0)
            self._prev_mean_features.append(mean_feature.iloc[i] if not pd.isna(mean_feature.iloc[i]) else 0)
            self._prev_momentum_features.append(momentum_feature.iloc[i] if not pd.isna(momentum_feature.iloc[i]) else 0)
            self._prev_atr.append(atr.iloc[i] if not pd.isna(atr.iloc[i]) else 0)
            self._prev_close.append(close.iloc[i])
            
            # Training: check if we have enough delayed data
            if (cfg.use_training and i > cfg.warmup_bars and
                len(self._prev_trend_features) > cfg.target_len):
                
                old_trend = self._prev_trend_features[0]  # features from targetLen bars ago
                old_mean = self._prev_mean_features[0]
                old_momentum = self._prev_momentum_features[0]
                
                target = close.iloc[i] / self._prev_close[0] - 1.0
                target_direction = 1.0 if target > 0 else (-1.0 if target < 0 else 0.0)
                
                quality_vol = max(abs(self._prev_atr[0] / self._prev_close[0]), 0.000001)
                sample_score = abs(target) / quality_vol
                
                if not (np.isnan(old_trend) or np.isnan(old_mean) or 
                        np.isnan(old_momentum) or np.isnan(target_direction)):
                    self._memory.append(TrainSample(
                        score=sample_score,
                        trend=old_trend,
                        mean=old_mean,
                        momentum=old_momentum,
                        target=target_direction,
                        idx=i
                    ))
            
            # Run training batch
            if cfg.use_training and len(self._memory) >= cfg.batch_size:
                sorted_memory = sorted(self._memory, key=lambda s: s.score, reverse=True)
                batch_n = min(cfg.batch_size, len(sorted_memory))
                
                for j in range(batch_n):
                    s = sorted_memory[j]
                    
                    pred = (self._tw_trend * s.trend + 
                           self._tw_mean * s.mean + 
                           self._tw_momentum * s.momentum + 
                           self._t_bias)
                    err = pred - s.target
                    
                    abs_err = abs(err)
                    if abs_err <= cfg.huber_d:
                        loss = 0.5 * err * err
                        grad_err = err
                    else:
                        loss = cfg.huber_d * (abs_err - 0.5 * cfg.huber_d)
                        grad_err = cfg.huber_d * np.sign(err)
                    
                    g_trend = grad_err * s.trend
                    g_mean = grad_err * s.mean
                    g_momentum = grad_err * s.momentum
                    g_bias = grad_err
                    
                    self._step += 1
                    
                    self._tw_trend = self._adam_trend.update(self._tw_trend, g_trend, cfg.lr, step=self._step)
                    self._tw_mean = self._adam_mean.update(self._tw_mean, g_mean, cfg.lr, step=self._step)
                    self._tw_momentum = self._adam_momentum.update(self._tw_momentum, g_momentum, cfg.lr, step=self._step)
                    self._t_bias = self._adam_bias.update(self._t_bias, g_bias, cfg.lr, step=self._step)
                    
                    self._last_loss = loss
            
            # ─── Adaptive Feature Amplification ────────────────────────────
            blend = cfg.learn_influence if cfg.use_training else 0.0
            
            max_w = max(abs(self._tw_trend), abs(self._tw_mean), abs(self._tw_momentum))
            safe_max = max(max_w, 0.0001)
            
            learn_trend = self._tw_trend / safe_max
            learn_mean = self._tw_mean / safe_max
            learn_momentum = self._tw_momentum / safe_max
            
            trend_amplifier = 1.0 + learn_trend * blend
            mean_amplifier = 1.0 + learn_mean * blend
            momentum_amplifier = 1.0 + learn_momentum * blend
            
            # AI prediction
            if not pd.isna(trend_feature.iloc[i]):
                ai_pred_raw = (self._tw_trend * trend_feature.iloc[i] +
                              self._tw_mean * mean_feature.iloc[i] +
                              self._tw_momentum * momentum_feature.iloc[i] +
                              self._t_bias)
                ai_pred_raw_full[i] = ai_pred_raw
            
            # ─── Oscillator ────────────────────────────────────────────────
            if not pd.isna(trend_score.iloc[i]):
                trend_pressure = (trend_score.iloc[i] - 50.0) * self.bwm_trend * trend_amplifier
                mean_pressure = (mean_score.iloc[i] - 50.0) * self.bwm_mean * mean_amplifier
                momentum_pressure = (momentum_score.iloc[i] - 50.0) * self.bwm_momentum * momentum_amplifier
                
                raw_osc = 50.0 + trend_pressure + mean_pressure + momentum_pressure
                
                # Smoothing (EMA)
                if i > 0 and not np.isnan(raw_osc_ema[i-1]):
                    alpha_smooth = 2.0 / (cfg.smooth_len + 1)
                    raw_osc_ema[i] = alpha_smooth * raw_osc + (1 - alpha_smooth) * raw_osc_ema[i-1]
                else:
                    raw_osc_ema[i] = raw_osc
                
                # AI line blend
                ai_center = 0.0
                if i >= cfg.ai_center_len:
                    window = ai_pred_raw_full[max(0, i-cfg.ai_center_len):i+1]
                    valid = window[~np.isnan(window)]
                    if len(valid) > 0:
                        ai_center = np.mean(valid)
                
                ai_pred = ai_pred_raw_full[i] - ai_center if not np.isnan(ai_pred_raw_full[i]) else 0
                ai_osc = 50.0 + clamp(ai_pred * 50, -50, 50)
                ai_strength = clamp(abs(ai_pred) * 3.0, 0.0, 1.0)
                
                line_blend = blend * ai_strength * cfg.line_influence
                
                osc_values[i] = clamp(raw_osc_ema[i] * (1.0 - line_blend) + ai_osc * line_blend, 0, 100)
        
        # ─── Signal Line + Histogram ───────────────────────────────────────
        osc_series = pd.Series(osc_values, index=df.index)
        signal_line = ema(osc_series, cfg.signal_len)
        hist_raw = osc_series - signal_line
        histogram = ema(hist_raw, cfg.hist_smooth)
        
        # ─── Price Sweep Confirmation ──────────────────────────────────────
        low_shifted = low.shift(cfg.sweep_lookback)
        high_shifted = high.shift(cfg.sweep_lookback)
        
        bull_sweep_now = (low < low_shifted) & (close > low_shifted)
        bear_sweep_now = (high > high_shifted) & (close < high_shifted)
        
        bull_sweep_bars = barssince(bull_sweep_now)
        bear_sweep_bars = barssince(bear_sweep_now)
        
        bull_sweep = bull_sweep_bars <= cfg.sweep_lookback
        bear_sweep = bear_sweep_bars <= cfg.sweep_lookback
        
        # ─── Crossover / Crossunder ────────────────────────────────────────
        osc_above_signal = osc_series > signal_line
        osc_prev_below = osc_series.shift(1) <= signal_line.shift(1)
        osc_below_signal = osc_series < signal_line
        osc_prev_above = osc_series.shift(1) >= signal_line.shift(1)
        
        crossover_osc_signal = osc_above_signal & osc_prev_below
        crossunder_osc_signal = osc_below_signal & osc_prev_above
        
        # ─── NWO Signals ───────────────────────────────────────────────────
        bull_signal = crossover_osc_signal & (osc_series < 30) & bull_sweep
        bear_signal = crossunder_osc_signal & (osc_series > 70) & bear_sweep
        
        # ─── Stochastic Signals ────────────────────────────────────────────
        stoch_k_cross_d_up = (stoch_k.shift(1) <= stoch_d.shift(1)) & (stoch_k > stoch_d)
        stoch_k_cross_d_down = (stoch_k.shift(1) >= stoch_d.shift(1)) & (stoch_k < stoch_d)
        
        # v3: Strict signals — crossover in extreme zones (original)
        stoch_bull = stoch_k_cross_d_up & (stoch_k < cfg.stoch_oversold)
        stoch_bear = stoch_k_cross_d_down & (stoch_k > cfg.stoch_overbought)
        
        # v3: Relaxed signals — crossover in wider zones (more signals)
        stoch_bull_relaxed = stoch_k_cross_d_up & (stoch_k < 30)
        stoch_bear_relaxed = stoch_k_cross_d_down & (stoch_k > 70)
        
        # v3: Zone-only signals — K enters/exits extreme zones (most signals)
        k_enter_oversold = (stoch_k.shift(1) >= 20) & (stoch_k < 20)
        k_exit_oversold = (stoch_k.shift(1) < 30) & (stoch_k >= 30)
        k_enter_overbought = (stoch_k.shift(1) <= 80) & (stoch_k > 80)
        k_exit_overbought = (stoch_k.shift(1) > 70) & (stoch_k <= 70)
        
        stoch_bull_zone = k_enter_oversold | (k_exit_oversold & stoch_k_cross_d_up)
        stoch_bear_zone = k_enter_overbought | (k_exit_overbought & stoch_k_cross_d_down)
        
        # ─── CVD Signals ───────────────────────────────────────────────────
        cvd_bull = cvd > cfg.cvd_threshold
        cvd_bear = cvd < -cfg.cvd_threshold
        
        # ─── NWO Direction Filter (v2: NWO is FILTER, not trigger) ────────────
        # Histogram direction confirms momentum:
        #   histogram > 0 → NWO momentum is bullish (osc above signal line)
        #   histogram < 0 → NWO momentum is bearish (osc below signal line)
        nwo_bullish_filter = histogram > 0
        nwo_bearish_filter = histogram < 0
        
        # ─── Combined Signal Logic (v2) ──────────────────────────────────────
        # Stoch = PRIMARY TRIGGER (crossover in extreme zones)
        # NWO  = DIRECTION FILTER (momentum confirmation)
        # CVD  = VOLUME CONFIRMATION (threshold ±1.0)
        
        # v3: Multi-level signal hierarchy
        
        # Level 0: STOCH-ONLY (relaxed zone) — lowest confidence, most signals
        stoch_only_bull = stoch_bull_zone
        stoch_only_bear = stoch_bear_zone
        
        # Level 1: STOCH-RELAXED + NWO direction filter
        combined_bull = stoch_bull_relaxed & nwo_bullish_filter
        combined_bear = stoch_bear_relaxed & nwo_bearish_filter
        
        # Level 1b: Original strict Stoch + NWO (high confidence)
        strict_combined_bull = stoch_bull & nwo_bullish_filter
        strict_combined_bear = stoch_bear & nwo_bearish_filter
        
        # Level 2: Full confluence (Stoch + NWO + CVD all align) — highest confidence
        confluence_bull = stoch_bull_relaxed & nwo_bullish_filter & cvd_bull
        confluence_bear = stoch_bear_relaxed & nwo_bearish_filter & cvd_bear
        
        return {
            # NWO core
            "osc": osc_series,
            "signal_line": signal_line,
            "histogram": histogram,
            "hist_plot": 50 + histogram,
            "bull_signal": bull_signal,
            "bear_signal": bear_signal,
            
            # Components
            "trend_score": trend_score,
            "mean_score": mean_score,
            "momentum_score": momentum_score,
            
            # Stochastic
            "stoch_k": stoch_k,
            "stoch_d": stoch_d,
            "stoch_bull": stoch_bull,
            "stoch_bear": stoch_bear,
            "stoch_bull_relaxed": stoch_bull_relaxed,
            "stoch_bear_relaxed": stoch_bear_relaxed,
            "stoch_bull_zone": stoch_bull_zone,
            "stoch_bear_zone": stoch_bear_zone,
            
            # CVD
            "cvd": cvd,
            "cvd_bull": cvd_bull,
            "cvd_bear": cvd_bear,
            
            # Stoch-only (v3: relaxed zone signals)
            "stoch_only_bull": stoch_only_bull,
            "stoch_only_bear": stoch_only_bear,
            
            # Combined (v3: relaxed Stoch + NWO filter)
            "combined_bull": combined_bull,
            "combined_bear": combined_bear,
            
            # Strict combined (v3: original strict Stoch + NWO)
            "strict_combined_bull": strict_combined_bull,
            "strict_combined_bear": strict_combined_bear,
            
            # Confluence (v3: Stoch + NWO filter + CVD)
            "confluence_bull": confluence_bull,
            "confluence_bear": confluence_bear,
            
            # NWO direction filter
            "nwo_bullish_filter": nwo_bullish_filter,
            "nwo_bearish_filter": nwo_bearish_filter,
            
            # BWM weights
            "bwm_trend": self.bwm_trend,
            "bwm_mean": self.bwm_mean,
            "bwm_momentum": self.bwm_momentum,
            "bwm_error": self.bwm_error,
            
            # Training weights
            "tw_trend": self._tw_trend,
            "tw_mean": self._tw_mean,
            "tw_momentum": self._tw_momentum,
            "t_bias": self._t_bias,
            "last_loss": self._last_loss,
        }
    
    def _empty_result(self, n: int) -> Dict:
        """Return empty result when not enough data."""
        nan_series = pd.Series([np.nan] * n)
        return {
            "osc": nan_series.copy(),
            "signal_line": nan_series.copy(),
            "histogram": nan_series.copy(),
            "hist_plot": nan_series.copy(),
            "bull_signal": pd.Series([False] * n),
            "bear_signal": pd.Series([False] * n),
            "trend_score": nan_series.copy(),
            "mean_score": nan_series.copy(),
            "momentum_score": nan_series.copy(),
            "stoch_k": nan_series.copy(),
            "stoch_d": nan_series.copy(),
            "stoch_bull": pd.Series([False] * n),
            "stoch_bear": pd.Series([False] * n),
            "stoch_bull_relaxed": pd.Series([False] * n),
            "stoch_bear_relaxed": pd.Series([False] * n),
            "stoch_bull_zone": pd.Series([False] * n),
            "stoch_bear_zone": pd.Series([False] * n),
            "cvd": nan_series.copy(),
            "cvd_bull": pd.Series([False] * n),
            "cvd_bear": pd.Series([False] * n),
            "stoch_only_bull": pd.Series([False] * n),
            "stoch_only_bear": pd.Series([False] * n),
            "combined_bull": pd.Series([False] * n),
            "combined_bear": pd.Series([False] * n),
            "strict_combined_bull": pd.Series([False] * n),
            "strict_combined_bear": pd.Series([False] * n),
            "confluence_bull": pd.Series([False] * n),
            "confluence_bear": pd.Series([False] * n),
            "nwo_bullish_filter": pd.Series([False] * n),
            "nwo_bearish_filter": pd.Series([False] * n),
            "bwm_trend": self.bwm_trend,
            "bwm_mean": self.bwm_mean,
            "bwm_momentum": self.bwm_momentum,
            "bwm_error": self.bwm_error,
            "tw_trend": self._tw_trend,
            "tw_mean": self._tw_mean,
            "tw_momentum": self._tw_momentum,
            "t_bias": self._t_bias,
            "last_loss": self._last_loss,
        }
    
    def get_current_state(self, df: pd.DataFrame) -> Optional[Dict]:
        """Get current indicator values for status display."""
        result = self.compute(df)
        i = len(df) - 1
        
        osc_val = result["osc"].iloc[i]
        if pd.isna(osc_val):
            return None
        
        return {
            "price": df['close'].iloc[i],
            "osc": round(osc_val, 2),
            "signal_line": round(result["signal_line"].iloc[i], 2) if not pd.isna(result["signal_line"].iloc[i]) else None,
            "histogram": round(result["histogram"].iloc[i], 2) if not pd.isna(result["histogram"].iloc[i]) else None,
            "trend_score": round(result["trend_score"].iloc[i], 2) if not pd.isna(result["trend_score"].iloc[i]) else None,
            "mean_score": round(result["mean_score"].iloc[i], 2) if not pd.isna(result["mean_score"].iloc[i]) else None,
            "momentum_score": round(result["momentum_score"].iloc[i], 2) if not pd.isna(result["momentum_score"].iloc[i]) else None,
            "stoch_k": round(result["stoch_k"].iloc[i], 2) if not pd.isna(result["stoch_k"].iloc[i]) else None,
            "stoch_d": round(result["stoch_d"].iloc[i], 2) if not pd.isna(result["stoch_d"].iloc[i]) else None,
            "cvd": round(result["cvd"].iloc[i], 2) if not pd.isna(result["cvd"].iloc[i]) else None,
            "bwm_trend": round(result["bwm_trend"], 4),
            "bwm_mean": round(result["bwm_mean"], 4),
            "bwm_momentum": round(result["bwm_momentum"], 4),
        }
