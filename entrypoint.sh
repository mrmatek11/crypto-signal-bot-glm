#!/bin/bash
# ═══════════════════════════════════════════════════════════
#  Entrypoint for Docker — przekazuje env vars jako CLI args
#  v2: Sekrety (webhook, API keys) czytane z env vars przez config.py
# ═══════════════════════════════════════════════════════════

set -e

echo "🚀 Crypto Signal Bot — Starting..."
echo "   Mode: 📡 ALERT ONLY (no execution)"
echo "   Symbols: ${SYMBOLS:-BTC/USDT,ETH/USDT,SOL/USDT}"
echo "   Timeframes: ${TIMEFRAMES:-5m,15m,1h}"
echo "   Trend filter: ${TREND_FILTER:-alert}"
echo "   Market: ${MARKET:-crypto}"
echo "   AI Sentiment: ${SENTIMENT:-false}"
echo "   GLM AI Analyst: ${GLM_API_KEY:+✅}${GLM_API_KEY:-❌}"
echo "═════════════════════════════════════════════════════════"

# Build command using bash array (proper quoting)
ARGS=()

# Webhook is now read from env var by config.py — only pass --test if not set
if [ -z "$DISCORD_WEBHOOK" ]; then
    ARGS+=("--test")
    echo "⚠️ No DISCORD_WEBHOOK set — running in TEST mode"
fi

# Symbols
if [ -n "$SYMBOLS" ]; then
    ARGS+=("--symbols" "$SYMBOLS")
fi

# Timeframes
if [ -n "$TIMEFRAMES" ]; then
    ARGS+=("--timeframes" "$TIMEFRAMES")
fi

# Scan interval
if [ -n "$SCAN_INTERVAL" ]; then
    ARGS+=("--interval" "$SCAN_INTERVAL")
fi

# Trend filter
if [ -n "$TREND_FILTER" ]; then
    ARGS+=("--trend-filter" "$TREND_FILTER")
fi

# Market source
if [ -n "$MARKET" ] && [ "$MARKET" != "crypto" ]; then
    ARGS+=("--market" "$MARKET")
fi

# Sentiment
if [ "$SENTIMENT" = "true" ]; then
    ARGS+=("--sentiment")
fi

# API keys are now read from env vars by config.py
# (CRYPTOPANIC_KEY, FINNHUB_KEY — no longer passed via CLI to avoid ps aux leak)
# Only pass if explicitly needed as CLI override:
if [ -n "$CRYPTOPANIC_KEY" ]; then
    ARGS+=("--cryptopanic-key" "$CRYPTOPANIC_KEY")
fi

if [ -n "$FINNHUB_KEY" ]; then
    ARGS+=("--finnhub-key" "$FINNHUB_KEY")
fi

# Position size
if [ -n "$POSITION_SIZE" ]; then
    ARGS+=("--position-size" "$POSITION_SIZE")
fi

# Log level
if [ -n "$LOG_LEVEL" ]; then
    ARGS+=("--log" "$LOG_LEVEL")
fi

# Execute with proper array expansion (handles spaces in values)
exec python3 bot.py "${ARGS[@]}"