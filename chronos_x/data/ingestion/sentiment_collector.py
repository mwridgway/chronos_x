"""
Sentiment Data Collection Module

Collects sentiment data from multiple sources:
- Twitter/X (crypto-related tweets)
- Reddit (r/CryptoCurrency, r/Bitcoin, etc.)
- News articles (CoinDesk, CoinTelegraph, etc.)
- On-chain metrics (via Glassnode)

Provides unified sentiment scoring and aggregation.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, AsyncIterator
from dataclasses import dataclass
from enum import Enum
import asyncio

import pandas as pd
import numpy as np
from textblob import TextBlob

logger = logging.getLogger(__name__)


class SentimentSource(Enum):
    """Sentiment data sources"""
    TWITTER = "twitter"
    REDDIT = "reddit"
    NEWS = "news"
    ONCHAIN = "onchain"


@dataclass
class SentimentConfig:
    """Configuration for sentiment collection"""
    # Twitter configuration
    twitter_enabled: bool = True
    twitter_keywords: List[str] = None
    twitter_max_tweets: int = 100

    # Reddit configuration
    reddit_enabled: bool = True
    reddit_subreddits: List[str] = None
    reddit_client_id: Optional[str] = None
    reddit_client_secret: Optional[str] = None
    reddit_user_agent: str = "chronos-x-sentiment"

    # News configuration
    news_enabled: bool = True
    news_sources: List[str] = None

    # On-chain configuration
    onchain_enabled: bool = False
    glassnode_api_key: Optional[str] = None

    # General
    lookback_hours: int = 24
    update_frequency_minutes: int = 60

    def __post_init__(self):
        if self.twitter_keywords is None:
            self.twitter_keywords = ["#BTC", "#Bitcoin", "#ETH", "#Ethereum", "$BTC", "$ETH"]
        if self.reddit_subreddits is None:
            self.reddit_subreddits = ["CryptoCurrency", "Bitcoin", "ethereum", "CryptoMarkets"]
        if self.news_sources is None:
            self.news_sources = ["coindesk", "cointelegraph", "decrypt"]


@dataclass
class SentimentData:
    """Container for sentiment data"""
    timestamp: datetime
    source: SentimentSource
    symbol: str
    text: str
    sentiment_score: float  # -1 to 1
    polarity: float
    subjectivity: float
    engagement: float  # normalized engagement metric
    metadata: Dict


class SentimentAnalyzer:
    """Base sentiment analyzer using TextBlob"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analyze_text(self, text: str) -> tuple[float, float]:
        """
        Analyze sentiment of text using TextBlob

        Args:
            text: Text to analyze

        Returns:
            Tuple of (polarity, subjectivity)
            - Polarity: -1 (negative) to 1 (positive)
            - Subjectivity: 0 (objective) to 1 (subjective)
        """
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity, blob.sentiment.subjectivity
        except Exception as e:
            self.logger.warning(f"Error analyzing text: {e}")
            return 0.0, 0.5

    def compute_sentiment_score(
        self,
        polarity: float,
        engagement: float,
        weight_engagement: float = 0.3
    ) -> float:
        """
        Compute weighted sentiment score

        Args:
            polarity: Sentiment polarity (-1 to 1)
            engagement: Normalized engagement (0 to 1)
            weight_engagement: Weight for engagement factor

        Returns:
            Weighted sentiment score
        """
        return polarity * (1 - weight_engagement) + polarity * engagement * weight_engagement


class TwitterSentimentCollector:
    """Collects sentiment from Twitter/X"""

    def __init__(self, config: SentimentConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.analyzer = SentimentAnalyzer()

    async def collect(
        self,
        keywords: Optional[List[str]] = None,
        since: Optional[datetime] = None,
        max_tweets: int = 100
    ) -> List[SentimentData]:
        """
        Collect tweets and analyze sentiment

        Args:
            keywords: List of keywords to search
            since: Start timestamp for search
            max_tweets: Maximum number of tweets to collect

        Returns:
            List of sentiment data
        """
        if keywords is None:
            keywords = self.config.twitter_keywords

        if since is None:
            since = datetime.utcnow() - timedelta(hours=self.config.lookback_hours)

        self.logger.info(f"Collecting tweets for keywords: {keywords}")

        # Note: This is a placeholder. Actual implementation requires:
        # 1. Twitter API v2 credentials
        # 2. tweepy or snscrape library
        # 3. Rate limiting handling

        try:
            # Placeholder - would use tweepy or snscrape here
            tweets = await self._fetch_tweets_placeholder(keywords, since, max_tweets)

            sentiment_data = []
            for tweet in tweets:
                polarity, subjectivity = self.analyzer.analyze_text(tweet["text"])

                # Normalize engagement
                engagement = self._normalize_engagement(
                    likes=tweet.get("likes", 0),
                    retweets=tweet.get("retweets", 0),
                    replies=tweet.get("replies", 0)
                )

                sentiment_score = self.analyzer.compute_sentiment_score(
                    polarity, engagement
                )

                sentiment_data.append(SentimentData(
                    timestamp=tweet["timestamp"],
                    source=SentimentSource.TWITTER,
                    symbol=self._extract_symbol(tweet["text"]),
                    text=tweet["text"],
                    sentiment_score=sentiment_score,
                    polarity=polarity,
                    subjectivity=subjectivity,
                    engagement=engagement,
                    metadata={
                        "user": tweet.get("user", ""),
                        "likes": tweet.get("likes", 0),
                        "retweets": tweet.get("retweets", 0)
                    }
                ))

            return sentiment_data

        except Exception as e:
            self.logger.error(f"Error collecting Twitter sentiment: {e}")
            return []

    async def _fetch_tweets_placeholder(
        self,
        keywords: List[str],
        since: datetime,
        max_tweets: int
    ) -> List[Dict]:
        """Placeholder for actual Twitter API calls"""
        # This would use tweepy or snscrape
        # Example with snscrape:
        # import snscrape.modules.twitter as sntwitter
        # tweets = []
        # for keyword in keywords:
        #     query = f"{keyword} since:{since.strftime('%Y-%m-%d')}"
        #     for tweet in sntwitter.TwitterSearchScraper(query).get_items():
        #         if len(tweets) >= max_tweets:
        #             break
        #         tweets.append({
        #             "timestamp": tweet.date,
        #             "text": tweet.content,
        #             "user": tweet.user.username,
        #             "likes": tweet.likeCount,
        #             "retweets": tweet.retweetCount,
        #             "replies": tweet.replyCount
        #         })

        self.logger.warning("Twitter API not configured - returning empty data")
        return []

    def _normalize_engagement(
        self,
        likes: int,
        retweets: int,
        replies: int,
        max_likes: int = 1000,
        max_retweets: int = 500
    ) -> float:
        """Normalize engagement metrics to 0-1 scale"""
        engagement = (
            min(likes / max_likes, 1.0) * 0.5 +
            min(retweets / max_retweets, 1.0) * 0.3 +
            min(replies / 100, 1.0) * 0.2
        )
        return engagement

    def _extract_symbol(self, text: str) -> str:
        """Extract crypto symbol from text"""
        # Simple extraction - looks for $BTC, #BTC, etc.
        import re
        symbols = re.findall(r'[\$#]([A-Z]{3,5})', text.upper())
        if symbols:
            return symbols[0]
        # Default
        if 'BTC' in text.upper() or 'BITCOIN' in text.upper():
            return 'BTC'
        if 'ETH' in text.upper() or 'ETHEREUM' in text.upper():
            return 'ETH'
        return 'UNKNOWN'


class RedditSentimentCollector:
    """Collects sentiment from Reddit"""

    def __init__(self, config: SentimentConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.analyzer = SentimentAnalyzer()
        self._reddit = None

    def _get_reddit_client(self):
        """Initialize PRAW Reddit client"""
        if self._reddit is None:
            try:
                import praw

                self._reddit = praw.Reddit(
                    client_id=self.config.reddit_client_id,
                    client_secret=self.config.reddit_client_secret,
                    user_agent=self.config.reddit_user_agent
                )
            except Exception as e:
                self.logger.error(f"Error initializing Reddit client: {e}")
                raise

        return self._reddit

    async def collect(
        self,
        subreddits: Optional[List[str]] = None,
        since: Optional[datetime] = None,
        limit: int = 100
    ) -> List[SentimentData]:
        """
        Collect Reddit posts and comments

        Args:
            subreddits: List of subreddit names
            since: Start timestamp
            limit: Maximum posts per subreddit

        Returns:
            List of sentiment data
        """
        if subreddits is None:
            subreddits = self.config.reddit_subreddits

        if since is None:
            since = datetime.utcnow() - timedelta(hours=self.config.lookback_hours)

        self.logger.info(f"Collecting from subreddits: {subreddits}")

        try:
            reddit = self._get_reddit_client()
            sentiment_data = []

            for subreddit_name in subreddits:
                subreddit = reddit.subreddit(subreddit_name)

                # Get hot posts
                for post in subreddit.hot(limit=limit):
                    post_time = datetime.fromtimestamp(post.created_utc)

                    if post_time < since:
                        continue

                    # Analyze post title + selftext
                    text = f"{post.title} {post.selftext}"
                    polarity, subjectivity = self.analyzer.analyze_text(text)

                    # Engagement based on upvotes and comments
                    engagement = min(post.score / 1000, 1.0) * 0.7 + min(post.num_comments / 100, 1.0) * 0.3

                    sentiment_score = self.analyzer.compute_sentiment_score(polarity, engagement)

                    sentiment_data.append(SentimentData(
                        timestamp=post_time,
                        source=SentimentSource.REDDIT,
                        symbol=self._extract_symbol_from_text(text),
                        text=text[:500],  # Truncate
                        sentiment_score=sentiment_score,
                        polarity=polarity,
                        subjectivity=subjectivity,
                        engagement=engagement,
                        metadata={
                            "subreddit": subreddit_name,
                            "upvotes": post.score,
                            "comments": post.num_comments,
                            "post_id": post.id
                        }
                    ))

            return sentiment_data

        except Exception as e:
            self.logger.error(f"Error collecting Reddit sentiment: {e}")
            return []

    def _extract_symbol_from_text(self, text: str) -> str:
        """Extract crypto symbol from text"""
        text_upper = text.upper()
        if 'BTC' in text_upper or 'BITCOIN' in text_upper:
            return 'BTC'
        if 'ETH' in text_upper or 'ETHEREUM' in text_upper:
            return 'ETH'
        return 'CRYPTO'


class SentimentCollector:
    """Main sentiment collector that aggregates all sources"""

    def __init__(self, config: SentimentConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize collectors
        self.twitter_collector = TwitterSentimentCollector(config) if config.twitter_enabled else None
        self.reddit_collector = RedditSentimentCollector(config) if config.reddit_enabled else None

    async def collect_all(
        self,
        since: Optional[datetime] = None
    ) -> List[SentimentData]:
        """
        Collect sentiment from all enabled sources

        Args:
            since: Start timestamp

        Returns:
            Aggregated sentiment data from all sources
        """
        if since is None:
            since = datetime.utcnow() - timedelta(hours=self.config.lookback_hours)

        tasks = []

        if self.twitter_collector:
            tasks.append(self.twitter_collector.collect(since=since))

        if self.reddit_collector:
            tasks.append(self.reddit_collector.collect(since=since))

        # Collect from all sources concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten results
        all_sentiment = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Collection error: {result}")
            else:
                all_sentiment.extend(result)

        self.logger.info(f"Collected {len(all_sentiment)} sentiment records")

        return all_sentiment

    def aggregate_sentiment(
        self,
        sentiment_data: List[SentimentData],
        symbol: str,
        time_window: timedelta = timedelta(hours=1)
    ) -> Dict[str, float]:
        """
        Aggregate sentiment for a specific symbol and time window

        Args:
            sentiment_data: List of sentiment data points
            symbol: Crypto symbol to aggregate
            time_window: Time window for aggregation

        Returns:
            Dictionary with aggregated metrics
        """
        # Filter by symbol
        filtered = [s for s in sentiment_data if s.symbol == symbol]

        if not filtered:
            return {
                "sentiment_score": 0.0,
                "polarity_mean": 0.0,
                "polarity_std": 0.0,
                "subjectivity_mean": 0.5,
                "volume_factor": 0.0,
                "count": 0
            }

        # Convert to DataFrame for easier aggregation
        df = pd.DataFrame([{
            "timestamp": s.timestamp,
            "sentiment_score": s.sentiment_score,
            "polarity": s.polarity,
            "subjectivity": s.subjectivity,
            "engagement": s.engagement
        } for s in filtered])

        # Weight by engagement
        weighted_sentiment = np.average(
            df["sentiment_score"],
            weights=df["engagement"] + 0.1  # Add small constant to avoid zero weights
        )

        return {
            "sentiment_score": float(weighted_sentiment),
            "polarity_mean": float(df["polarity"].mean()),
            "polarity_std": float(df["polarity"].std()),
            "subjectivity_mean": float(df["subjectivity"].mean()),
            "volume_factor": len(filtered) / 100.0,  # Normalized volume
            "count": len(filtered)
        }


def create_sentiment_collector(
    twitter_enabled: bool = True,
    reddit_enabled: bool = True,
    **kwargs
) -> SentimentCollector:
    """
    Factory function to create sentiment collector

    Args:
        twitter_enabled: Enable Twitter collection
        reddit_enabled: Enable Reddit collection
        **kwargs: Additional configuration options

    Returns:
        Configured SentimentCollector instance
    """
    config = SentimentConfig(
        twitter_enabled=twitter_enabled,
        reddit_enabled=reddit_enabled,
        **kwargs
    )
    return SentimentCollector(config)


# Example usage
if __name__ == "__main__":
    import asyncio

    async def example():
        collector = create_sentiment_collector(
            twitter_enabled=False,  # Requires API credentials
            reddit_enabled=True,
            reddit_client_id="your_client_id",
            reddit_client_secret="your_client_secret"
        )

        # Collect sentiment from last 24 hours
        sentiment_data = await collector.collect_all()

        print(f"Collected {len(sentiment_data)} sentiment records")

        # Aggregate for BTC
        btc_sentiment = collector.aggregate_sentiment(sentiment_data, "BTC")
        print(f"BTC sentiment: {btc_sentiment}")

    asyncio.run(example())
