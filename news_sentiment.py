"""
News Sentiment Module — AI analiza newsów w czasie rzeczywistym
═════════════════════════════════════════════════════════════════════════

Komponenty:
  1. NewsFetcher — pobiera newsy z darmowych źródeł (CryptoPanic, Finnhub, RSS)
  2. SentimentAnalyzer — analizuje sentiment przez LLM (z-ai-web-dev-sdk)
  3. SentimentFilter — filtruje sygnały na bazie sentimentu

Logika:
  - Newsy pobierane co 5 minut
  - LLM ocenia sentiment na skali -1.0 do +1.0
  - Sentiment < -0.5 → blokuj LONG
  - Sentiment > +0.5 → blokuj SHORT
  - Sentiment jest FILTREM, nie triggerem

Użycie:
  from news_sentiment import SentimentEngine
  engine = SentimentEngine()
  score = engine.get_sentiment("BTC")
  if engine.should_filter_signal("LONG", score):
      # Blokuj sygnał
"""

import os
import time
import json
import requests
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict


# ═══════════════════════════════════════════════════════════════════════════════
# NEWS DATA
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class NewsItem:
    """Pojedynczy news."""
    title: str
    source: str
    url: str
    published_at: datetime
    symbols: List[str] = field(default_factory=list)
    content: str = ""  # Snippet/summary
    
    @property
    def cache_key(self) -> str:
        return hashlib.md5(f"{self.title}:{self.url}".encode()).hexdigest()


# ═══════════════════════════════════════════════════════════════════════════════
# SENTIMENT SCORE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SentimentScore:
    """Wynik analizy sentimentu."""
    score: float           # -1.0 (very bearish) do +1.0 (very bullish)
    confidence: float      # 0.0 do 1.0 — jak pewny jest AI
    summary: str           # Krótkie wyjaśnienie
    news_count: int        # Ile newsów analizowano
    timestamp: datetime    # Kiedy obliczono
    symbol: str            # Dla jakiego symbolu
    
    @property
    def label(self) -> str:
        if self.score > 0.5:
            return "VERY_BULLISH"
        elif self.score > 0.2:
            return "BULLISH"
        elif self.score > -0.2:
            return "NEUTRAL"
        elif self.score > -0.5:
            return "BEARISH"
        else:
            return "VERY_BEARISH"
    
    @property
    def emoji(self) -> str:
        labels = {
            "VERY_BULLISH": "🟢🔥",
            "BULLISH": "🟢",
            "NEUTRAL": "⚪",
            "BEARISH": "🔴",
            "VERY_BEARISH": "🔴🔥",
        }
        return labels.get(self.label, "⚪")


# ═══════════════════════════════════════════════════════════════════════════════
# NEWS FETCHER — pobiera newsy z darmowych źródeł
# ═══════════════════════════════════════════════════════════════════════════════

class NewsFetcher:
    """Pobiera newsy finansowe/krypto z darmowych API i RSS."""
    
    # Crypto-related RSS feeds
    RSS_FEEDS = [
        "https://feeds.coindesk.com/coindesk/Bitcoin",
        "https://cointelegraph.com/rss",
        "https://cryptonews.com/news/feed/",
    ]
    
    def __init__(
        self,
        cryptopanic_api_key: str = "",
        finnhub_api_key: str = "",
        newsapi_key: str = "",
    ):
        self.cryptopanic_key = cryptopanic_api_key
        self.finnhub_key = finnhub_api_key
        self.newsapi_key = newsapi_key
        self._seen_keys = set()
    
    def fetch_cryptopanic(self, symbols: List[str] = None) -> List[NewsItem]:
        """
        CryptoPanic API — newsy krypto (WYMAGA PŁATNEGO KLUCZA).
        Od 2025 free tier został usunięty. Użyj RSS jako darmowej alternatywy.
        """
        if not self.cryptopanic_key:
            return []
        
        news_items = []
        
        try:
            currencies = ",".join([s.replace("/USDT", "").replace("/USD", "") for s in (symbols or ["BTC"])])
            url = f"https://cryptopanic.com/api/v1/posts/?auth_token={self.cryptopanic_key}&currencies={currencies}&kind=news&filter=hot"
            
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                return []
            
            data = resp.json()
            for post in data.get("results", [])[:20]:
                item = NewsItem(
                    title=post.get("title", ""),
                    source=post.get("domain", "cryptopanic"),
                    url=post.get("url", ""),
                    published_at=datetime.fromisoformat(post.get("published_at", "").replace("Z", "+00:00")) if post.get("published_at") else datetime.now(timezone.utc),
                    symbols=[c for c in post.get("currencies", [])],
                )
                
                if item.cache_key not in self._seen_keys:
                    self._seen_keys.add(item.cache_key)
                    news_items.append(item)
            
        except Exception as e:
            print(f"[NewsFetcher] CryptoPanic error: {e}")
        
        # Limit cache
        if len(self._seen_keys) > 1000:
            self._seen_keys = set(list(self._seen_keys)[-500:])
        
        return news_items
    
    def fetch_finnhub(self, symbol: str = "BTC") -> List[NewsItem]:
        """
        Finnhub API — darmowe newsy finansowe.
        Free tier: 60 callów/min.
        """
        if not self.finnhub_key:
            return []
        
        news_items = []
        
        try:
            # Finnhub market news
            url = f"https://finnhub.io/api/v1/news?category=crypto&token={self.finnhub_key}"
            
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                return []
            
            data = resp.json()
            for article in data[:15]:
                item = NewsItem(
                    title=article.get("headline", ""),
                    source=article.get("source", "finnhub"),
                    url=article.get("url", ""),
                    published_at=datetime.fromtimestamp(article.get("datetime", 0), tz=timezone.utc) if article.get("datetime") else datetime.now(timezone.utc),
                    content=article.get("summary", ""),
                )
                
                if item.cache_key not in self._seen_keys:
                    self._seen_keys.add(item.cache_key)
                    news_items.append(item)
            
        except Exception as e:
            print(f"[NewsFetcher] Finnhub error: {e}")
        
        return news_items
    
    def fetch_rss(self) -> List[NewsItem]:
        """Pobierz newsy z RSS feeds (zawsze darmowe, bez API key)."""
        news_items = []
        
        for feed_url in self.RSS_FEEDS:
            try:
                resp = requests.get(feed_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
                if resp.status_code != 200:
                    continue
                
                # Simple XML parsing (no external dependency)
                import xml.etree.ElementTree as ET
                root = ET.fromstring(resp.text)
                
                # RSS 2.0 structure
                for item_elem in root.iter("item"):
                    title = item_elem.findtext("title", "")
                    link = item_elem.findtext("link", "")
                    pub_date = item_elem.findtext("pubDate", "")
                    description = item_elem.findtext("description", "")
                    
                    if not title:
                        continue
                    
                    item = NewsItem(
                        title=title,
                        source=feed_url.split("/")[2],
                        url=link,
                        published_at=datetime.now(timezone.utc),  # Simplified
                        content=description[:300] if description else "",
                    )
                    
                    if item.cache_key not in self._seen_keys:
                        self._seen_keys.add(item.cache_key)
                        news_items.append(item)
                
            except Exception:
                continue
        
        # Limit cache
        if len(self._seen_keys) > 1000:
            self._seen_keys = set(list(self._seen_keys)[-500:])
        
        return news_items
    
    def fetch_all(self, symbol: str = "BTC") -> List[NewsItem]:
        """Pobierz ze wszystkich dostępnych źródeł."""
        all_news = []
        
        # RSS (zawsze dostępne, darmowe — GŁÓWNE ŹRÓDŁO)
        all_news.extend(self.fetch_rss())
        
        # Finnhub (darmowy, 60 req/min — opcjonalny)
        if self.finnhub_key:
            all_news.extend(self.fetch_finnhub(symbol))
        
        # CryptoPanic (PŁATNY — opcjonalny)
        if self.cryptopanic_key:
            all_news.extend(self.fetch_cryptopanic([symbol]))
        
        # Dedup i sortuj
        seen = set()
        unique = []
        for item in all_news:
            if item.cache_key not in seen:
                seen.add(item.cache_key)
                unique.append(item)
        
        return unique[:30]  # Max 30 newsów


# ═══════════════════════════════════════════════════════════════════════════════
# SENTIMENT ANALYZER — LLM analiza przez z-ai-web-dev-sdk
# ═══════════════════════════════════════════════════════════════════════════════

class SentimentAnalyzer:
    """Analizuje sentiment newsów przez LLM."""
    
    def __init__(self):
        self._zai = None
    
    async def _get_zai(self):
        """Lazy init z-ai-web-dev-sdk."""
        if self._zai is None:
            from z_ai_web_dev_sdk import ZAI
            self._zai = await ZAI.create()
        return self._zai
    
    def analyze_sync(self, news_items: List[NewsItem], symbol: str) -> SentimentScore:
        """
        Analizuj sentiment newsów synchronicznie.
        Używa subprocess żeby wywołać Node.js SDK.
        """
        if not news_items:
            return SentimentScore(
                score=0.0,
                confidence=0.0,
                summary="Brak newsów do analizy",
                news_count=0,
                timestamp=datetime.now(timezone.utc),
                symbol=symbol,
            )
        
        # Przygotuj tekst newsów
        news_text = ""
        for i, item in enumerate(news_items[:10]):
            news_text += f"{i+1}. [{item.source}] {item.title}"
            if item.content:
                news_text += f" — {item.content[:150]}"
            news_text += "\n"
        
        # Wywołaj LLM przez Node.js subprocess
        prompt = f"""Analyze the following cryptocurrency news for {symbol} and provide a sentiment score.

NEWS:
{news_text}

Respond ONLY with valid JSON in this exact format:
{{"score": <float between -1.0 and 1.0>, "confidence": <float between 0.0 and 1.0>, "summary": "<one sentence explanation>"}}

Scoring guide:
- score > 0.5: Very bullish (positive news like adoption, ETF approval, institutional buying)
- score 0.2-0.5: Mildly bullish
- score -0.2 to 0.2: Neutral / mixed signals
- score -0.5 to -0.2: Mildly bearish
- score < -0.5: Very bearish (hacks, bans, regulatory crackdowns, major sell-offs)
- confidence: how clear the sentiment is (0.3 = mixed/conflicting, 0.9 = very clear direction)"""
        
        try:
            result = self._call_llm_sync(prompt)
            return self._parse_llm_response(result, symbol, len(news_items))
        except Exception as e:
            print(f"[SentimentAnalyzer] LLM error: {e}")
            return SentimentScore(
                score=0.0,
                confidence=0.0,
                summary=f"LLM analysis failed: {str(e)[:100]}",
                news_count=len(news_items),
                timestamp=datetime.now(timezone.utc),
                symbol=symbol,
            )
    
    def _call_llm_sync(self, prompt: str) -> str:
        """Wywołaj LLM przez Node.js subprocess (z-ai-web-dev-sdk)."""
        import subprocess
        import tempfile
        
        # Stwórz Node.js skrypt
        prompt_escaped = json.dumps(prompt)
        script = f"""
const ZAI = require('z-ai-web-dev-sdk').default;

async function main() {{
    const zai = await ZAI.create();
    const completion = await zai.chat.completions.create({{
        messages: [
            {{ role: "system", content: "You are a financial news sentiment analyzer. Respond ONLY with valid JSON." }},
            {{ role: "user", content: {prompt_escaped} }}
        ],
        temperature: 0.1,
        max_tokens: 200,
    }});
    console.log(completion.choices[0]?.message?.content || "");
}}

main().catch(e => {{ console.error(e.message); process.exit(1); }});
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
            f.write(script)
            script_path = f.name
        
        try:
            # Set NODE_PATH to find globally installed modules
            import subprocess
            env = os.environ.copy()
            npm_root = subprocess.run(['npm', 'root', '-g'], capture_output=True, text=True).stdout.strip()
            if npm_root:
                env['NODE_PATH'] = npm_root
            
            result = subprocess.run(
                ['node', script_path],
                capture_output=True, text=True, timeout=30,
                env=env,
            )
            return result.stdout.strip()
        finally:
            os.unlink(script_path)
    
    def _parse_llm_response(self, response: str, symbol: str, news_count: int) -> SentimentScore:
        """Parse LLM JSON response into SentimentScore."""
        # Extract JSON from response (may have markdown code blocks)
        json_str = response
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0]
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0]
        
        data = json.loads(json_str.strip())
        
        return SentimentScore(
            score=max(-1.0, min(1.0, float(data.get("score", 0)))),
            confidence=max(0.0, min(1.0, float(data.get("confidence", 0)))),
            summary=str(data.get("summary", ""))[:200],
            news_count=news_count,
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SENTIMENT ENGINE — orkiestracja fetch + analyze + cache
# ═══════════════════════════════════════════════════════════════════════════════

class SentimentEngine:
    """
    Główny engine sentimentu.
    
    Pętla:
      1. Pobierz newsy (co 5 min)
      2. Analizuj przez LLM
      3. Cache wynik per symbol
      4. Udostępnij jako filter do strategii
    """
    
    # Filter thresholds
    LONG_BLOCK_THRESHOLD = -0.5    # Blokuj LONG gdy sentiment < -0.5
    SHORT_BLOCK_THRESHOLD = 0.5    # Blokuj SHORT gdy sentiment > +0.5
    WARNING_THRESHOLD = 0.3        # Ostrzeżenie gdy |sentiment| > 0.3
    
    def __init__(
        self,
        cryptopanic_key: str = "",
        finnhub_key: str = "",
        newsapi_key: str = "",
        refresh_interval: int = 300,   # sekundy (5 min)
    ):
        self.fetcher = NewsFetcher(cryptopanic_key, finnhub_key, newsapi_key)
        self.analyzer = SentimentAnalyzer()
        self.refresh_interval = refresh_interval
        
        # Cache: symbol → (SentimentScore, timestamp)
        self._cache: Dict[str, Tuple[SentimentScore, float]] = {}
        self._last_refresh: float = 0
    
    def refresh(self, symbol: str = "BTC") -> SentimentScore:
        """Pobierz i analizuj newsy na nowo."""
        # Map symbol format
        clean_symbol = symbol.replace("/USDT", "").replace("/USD", "")
        
        print(f"[SentimentEngine] Refreshing sentiment for {clean_symbol}...")
        
        # Fetch news
        news = self.fetcher.fetch_all(clean_symbol)
        
        # Analyze
        score = self.analyzer.analyze_sync(news, clean_symbol)
        
        # Cache
        self._cache[clean_symbol] = (score, time.time())
        self._last_refresh = time.time()
        
        print(f"[SentimentEngine] {clean_symbol}: {score.emoji} {score.label} (score={score.score:.2f}, confidence={score.confidence:.2f}) — {score.summary}")
        
        return score
    
    def get_sentiment(self, symbol: str = "BTC") -> SentimentScore:
        """Pobierz sentiment (z cache jeśli świeży, refresh jeśli stary)."""
        clean_symbol = symbol.replace("/USDT", "").replace("/USD", "")
        
        # Check cache
        if clean_symbol in self._cache:
            score, ts = self._cache[clean_symbol]
            age = time.time() - ts
            if age < self.refresh_interval:
                return score
        
        # Refresh
        return self.refresh(symbol)
    
    def should_filter_signal(self, direction: str, score: SentimentScore) -> Tuple[bool, str]:
        """
        Czy zablokować sygnał na bazie sentimentu?
        
        Returns:
            (should_block: bool, reason: str)
        """
        # Nie filtruj gdy niska pewność
        if score.confidence < 0.3:
            return False, "Low confidence sentiment"
        
        if direction == "LONG":
            if score.score < self.LONG_BLOCK_THRESHOLD:
                return True, f"Sentiment {score.label} ({score.score:.2f}) — blocking LONG"
            elif score.score < -self.WARNING_THRESHOLD:
                return False, f"⚠️ Sentiment {score.label} ({score.score:.2f}) — LONG caution"
        
        elif direction == "SHORT":
            if score.score > self.SHORT_BLOCK_THRESHOLD:
                return True, f"Sentiment {score.label} ({score.score:.2f}) — blocking SHORT"
            elif score.score > self.WARNING_THRESHOLD:
                return False, f"⚠️ Sentiment {score.label} ({score.score:.2f}) — SHORT caution"
        
        return False, f"Sentiment {score.label} ({score.score:.2f})"
    
    def get_status(self) -> Dict:
        """Status for display."""
        result = {}
        for symbol, (score, ts) in self._cache.items():
            age = int(time.time() - ts)
            result[symbol] = {
                "score": score.score,
                "label": score.label,
                "emoji": score.emoji,
                "confidence": score.confidence,
                "summary": score.summary,
                "news_count": score.news_count,
                "age_seconds": age,
            }
        return result
