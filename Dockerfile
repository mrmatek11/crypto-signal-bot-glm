FROM python:3.11-slim

LABEL maintainer="Crypto Signal Bot"
LABEL description="NWO + Stoch(7,3,2) + CVD Signal Bot — ALERT ONLY"

# System deps + Node.js (for AI sentiment via z-ai-web-dev-sdk)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g z-ai-web-dev-sdk 2>/dev/null || true \
    && apt-get purge -y curl \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Workdir
WORKDIR /app

# Copy bot files
COPY *.py ./
COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh

# Create data dir for SQLite
RUN mkdir -p /data

# Environment defaults (override with .env or docker-compose)
ENV DISCORD_WEBHOOK="" \
    SYMBOLS="BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,XRP/USDT,ADA/USDT,DOGE/USDT,AVAX/USDT,DOT/USDT,LINK/USDT" \
    TIMEFRAMES="5m,15m,1h" \
    SCAN_INTERVAL=60 \
    TREND_FILTER=alert \
    MARKET=crypto \
    SENTIMENT=false \
    CRYPTOPANIC_KEY="" \
    FINNHUB_KEY="" \
    GLM_API_KEY="" \
    GLM_MODEL=glm-4-flash \
    LOG_LEVEL=INFO \
    POSITION_SIZE=100

# Healthcheck
HEALTHCHECK --interval=60s --timeout=5s --retries=3 \
    CMD pgrep -f "bot.py" || exit 1

# Run
ENTRYPOINT ["./entrypoint.sh"]
