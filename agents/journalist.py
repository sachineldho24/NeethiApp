"""
Journalist Agent - Legal News Updates

Role: Keep users informed about latest legal developments
Goal: Provide relevant, timely legal news and updates
Tools: RSS feeds from legal news sources
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from loguru import logger
import feedparser


@dataclass
class NewsArticle:
    """A legal news article"""
    title: str
    summary: str
    url: str
    source: str
    published_date: datetime
    categories: List[str] = None


class JournalistAgent:
    """
    The Journalist Agent fetches and filters legal news.
    
    Workflow:
    1. Fetch RSS feeds from legal news sources
    2. Parse and structure articles
    3. Filter by relevance and recency
    4. Return top articles matching user's interest
    """
    
    DEFAULT_FEEDS = [
        {"name": "LiveLaw", "url": "https://www.livelaw.in/rss/feed"},
        {"name": "Bar and Bench", "url": "https://www.barandbench.com/feed"},
    ]
    
    # Keywords for filtering relevant legal news
    LEGAL_KEYWORDS = [
        "supreme court", "high court", "bail", "judgment", "verdict",
        "ipc", "bns", "crpc", "bnss", "amendment", "law", "legal",
        "accused", "petitioner", "respondent", "appeal", "review"
    ]
    
    def __init__(
        self,
        feeds: Optional[List[Dict[str, str]]] = None,
        cache_duration_hours: int = 6
    ):
        self.feeds = feeds or self.DEFAULT_FEEDS
        self.cache_duration = timedelta(hours=cache_duration_hours)
        self.cache: Dict[str, List[NewsArticle]] = {}
        self.last_fetch: Optional[datetime] = None
    
    async def fetch_news(
        self,
        topic: Optional[str] = None,
        days: int = 7,
        limit: int = 10
    ) -> List[NewsArticle]:
        """
        Fetch recent legal news.
        
        Args:
            topic: Optional topic filter (e.g., "bail", "supreme court")
            days: Number of days to look back
            limit: Maximum articles to return
            
        Returns:
            List of relevant news articles
        """
        logger.info(f"Journalist fetching news, topic: {topic}")
        
        articles = []
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for feed_config in self.feeds:
            try:
                feed_articles = await self._fetch_feed(feed_config)
                articles.extend(feed_articles)
            except Exception as e:
                logger.error(f"Error fetching {feed_config['name']}: {e}")
        
        # Filter by date
        articles = [a for a in articles if a.published_date >= cutoff_date]
        
        # Filter by topic if specified
        if topic:
            articles = self._filter_by_topic(articles, topic)
        
        # Sort by date (most recent first)
        articles.sort(key=lambda x: x.published_date, reverse=True)
        
        logger.info(f"Found {len(articles)} relevant articles")
        return articles[:limit]
    
    async def _fetch_feed(self, feed_config: Dict[str, str]) -> List[NewsArticle]:
        """Fetch and parse a single RSS feed"""
        feed = feedparser.parse(feed_config["url"])
        articles = []
        
        for entry in feed.entries:
            try:
                published = entry.get("published_parsed") or entry.get("updated_parsed")
                pub_date = datetime(*published[:6]) if published else datetime.now()
                
                article = NewsArticle(
                    title=entry.get("title", ""),
                    summary=entry.get("summary", entry.get("description", ""))[:500],
                    url=entry.get("link", ""),
                    source=feed_config["name"],
                    published_date=pub_date,
                    categories=[t.get("term", "") for t in entry.get("tags", [])]
                )
                articles.append(article)
            except Exception as e:
                logger.debug(f"Error parsing entry: {e}")
        
        return articles
    
    def _filter_by_topic(
        self,
        articles: List[NewsArticle],
        topic: str
    ) -> List[NewsArticle]:
        """Filter articles by topic relevance"""
        topic_lower = topic.lower()
        relevant = []
        
        for article in articles:
            text = f"{article.title} {article.summary}".lower()
            if topic_lower in text:
                relevant.append(article)
        
        return relevant


def create_journalist_agent(**kwargs) -> JournalistAgent:
    """Factory function to create a Journalist agent"""
    return JournalistAgent(**kwargs)
