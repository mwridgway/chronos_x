"""
Advanced Sentiment Analysis for Cryptocurrency Markets

Provides enhanced sentiment analysis using:
- FinBERT for financial sentiment
- VADER for social media sentiment
- Custom crypto-specific lexicons
- Fact vs Subjectivity classification
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re

import numpy as np

logger = logging.getLogger(__name__)


class SentimentModel(Enum):
    """Available sentiment analysis models"""
    TEXTBLOB = "textblob"
    VADER = "vader"
    FINBERT = "finbert"
    CRYPTO_LEXICON = "crypto_lexicon"


@dataclass
class SentimentResult:
    """Container for sentiment analysis results"""
    polarity: float  # -1 to 1
    subjectivity: float  # 0 to 1
    confidence: float  # 0 to 1
    is_factual: bool
    sentiment_label: str  # 'positive', 'negative', 'neutral'
    model_used: str


class CryptoLexicon:
    """Cryptocurrency-specific sentiment lexicon"""

    def __init__(self):
        # Positive crypto terms
        self.positive_terms = {
            'moon': 2.0, 'mooning': 2.0, 'bullish': 1.5, 'bull': 1.5,
            'pump': 1.5, 'green': 1.0, 'profits': 1.5, 'gains': 1.5,
            'rally': 1.5, 'breakout': 1.5, 'adoption': 1.0, 'hodl': 1.0,
            'lambo': 1.5, 'rocket': 2.0, 'ath': 1.5, 'institutional': 1.0,
            'golden cross': 1.5, 'accumulate': 1.0, 'buy the dip': 1.0
        }

        # Negative crypto terms
        self.negative_terms = {
            'crash': -2.0, 'dump': -2.0, 'bearish': -1.5, 'bear': -1.5,
            'red': -1.0, 'losses': -1.5, 'fud': -1.5, 'scam': -2.0,
            'rug pull': -2.5, 'death cross': -1.5, 'capitulation': -1.5,
            'panic': -1.5, 'sell-off': -1.5, 'collapse': -2.0,
            'hack': -2.0, 'rekt': -2.0, 'liquidation': -1.5
        }

        # Neutral/technical terms
        self.neutral_terms = {
            'dyor': 0.0, 'nfa': 0.0, 'ta': 0.0, 'fa': 0.0,
            'resistance': 0.0, 'support': 0.0, 'consolidation': 0.0
        }

    def score_text(self, text: str) -> float:
        """
        Score text using crypto-specific lexicon

        Args:
            text: Text to analyze

        Returns:
            Sentiment score (-2 to 2)
        """
        text_lower = text.lower()
        score = 0.0
        term_count = 0

        for term, weight in self.positive_terms.items():
            if term in text_lower:
                score += weight
                term_count += 1

        for term, weight in self.negative_terms.items():
            if term in text_lower:
                score += weight
                term_count += 1

        # Normalize by number of terms found
        if term_count > 0:
            score = score / term_count

        # Clamp to -1 to 1
        return np.clip(score, -1.0, 1.0)


class AdvancedSentimentAnalyzer:
    """
    Advanced sentiment analyzer with multiple models

    Supports TextBlob, VADER, FinBERT, and crypto-specific lexicons
    """

    def __init__(
        self,
        use_finbert: bool = False,
        use_vader: bool = True,
        use_crypto_lexicon: bool = True
    ):
        self.logger = logging.getLogger(__name__)
        self.use_finbert = use_finbert
        self.use_vader = use_vader
        self.use_crypto_lexicon = use_crypto_lexicon

        # Initialize models
        self._finbert_model = None
        self._finbert_tokenizer = None
        self._vader_analyzer = None
        self.crypto_lexicon = CryptoLexicon() if use_crypto_lexicon else None

        # Lazy loading
        if use_finbert:
            self._load_finbert()
        if use_vader:
            self._load_vader()

    def _load_finbert(self):
        """Load FinBERT model for financial sentiment"""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch

            self.logger.info("Loading FinBERT model...")
            model_name = "ProsusAI/finbert"

            self._finbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self._finbert_model.eval()

            self.logger.info("FinBERT loaded successfully")

        except Exception as e:
            self.logger.error(f"Error loading FinBERT: {e}")
            self.use_finbert = False

    def _load_vader(self):
        """Load VADER sentiment analyzer"""
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            self._vader_analyzer = SentimentIntensityAnalyzer()
            self.logger.info("VADER loaded successfully")

        except Exception as e:
            self.logger.error(f"Error loading VADER: {e}")
            self.use_vader = False

    def analyze_with_textblob(self, text: str) -> Tuple[float, float]:
        """
        Analyze sentiment using TextBlob

        Args:
            text: Text to analyze

        Returns:
            Tuple of (polarity, subjectivity)
        """
        try:
            from textblob import TextBlob

            blob = TextBlob(text)
            return blob.sentiment.polarity, blob.sentiment.subjectivity

        except Exception as e:
            self.logger.warning(f"TextBlob error: {e}")
            return 0.0, 0.5

    def analyze_with_vader(self, text: str) -> float:
        """
        Analyze sentiment using VADER

        Args:
            text: Text to analyze

        Returns:
            Compound sentiment score (-1 to 1)
        """
        if not self._vader_analyzer:
            self._load_vader()

        if self._vader_analyzer:
            scores = self._vader_analyzer.polarity_scores(text)
            return scores['compound']
        else:
            return 0.0

    def analyze_with_finbert(self, text: str) -> Tuple[float, float]:
        """
        Analyze sentiment using FinBERT

        Args:
            text: Text to analyze

        Returns:
            Tuple of (sentiment_score, confidence)
        """
        if not self._finbert_model:
            self._load_finbert()

        if not self._finbert_model:
            return 0.0, 0.0

        try:
            import torch

            # Tokenize
            inputs = self._finbert_tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )

            # Get predictions
            with torch.no_grad():
                outputs = self._finbert_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # FinBERT outputs: [negative, neutral, positive]
            probs = predictions[0].numpy()
            sentiment_score = probs[2] - probs[0]  # positive - negative
            confidence = max(probs)

            return float(sentiment_score), float(confidence)

        except Exception as e:
            self.logger.error(f"FinBERT analysis error: {e}")
            return 0.0, 0.0

    def analyze(self, text: str, model: Optional[SentimentModel] = None) -> SentimentResult:
        """
        Analyze sentiment using specified or ensemble model

        Args:
            text: Text to analyze
            model: Specific model to use (None for ensemble)

        Returns:
            SentimentResult object
        """
        if model == SentimentModel.TEXTBLOB:
            polarity, subjectivity = self.analyze_with_textblob(text)
            label = self._polarity_to_label(polarity)
            return SentimentResult(
                polarity=polarity,
                subjectivity=subjectivity,
                confidence=abs(polarity),
                is_factual=subjectivity < 0.3,
                sentiment_label=label,
                model_used="textblob"
            )

        elif model == SentimentModel.VADER:
            polarity = self.analyze_with_vader(text)
            label = self._polarity_to_label(polarity)
            return SentimentResult(
                polarity=polarity,
                subjectivity=0.5,  # VADER doesn't provide subjectivity
                confidence=abs(polarity),
                is_factual=False,
                sentiment_label=label,
                model_used="vader"
            )

        elif model == SentimentModel.FINBERT:
            polarity, confidence = self.analyze_with_finbert(text)
            label = self._polarity_to_label(polarity)
            return SentimentResult(
                polarity=polarity,
                subjectivity=0.3,  # FinBERT for factual financial text
                confidence=confidence,
                is_factual=True,
                sentiment_label=label,
                model_used="finbert"
            )

        else:
            # Ensemble approach - combine multiple models
            return self._ensemble_analysis(text)

    def _ensemble_analysis(self, text: str) -> SentimentResult:
        """
        Combine multiple sentiment analysis models

        Args:
            text: Text to analyze

        Returns:
            Ensemble sentiment result
        """
        scores = []
        weights = []
        subjectivities = []

        # TextBlob (baseline)
        tb_polarity, tb_subjectivity = self.analyze_with_textblob(text)
        scores.append(tb_polarity)
        weights.append(0.2)
        subjectivities.append(tb_subjectivity)

        # VADER (good for social media)
        if self.use_vader:
            vader_score = self.analyze_with_vader(text)
            scores.append(vader_score)
            weights.append(0.3)

        # FinBERT (best for financial text)
        if self.use_finbert:
            finbert_score, finbert_conf = self.analyze_with_finbert(text)
            scores.append(finbert_score)
            weights.append(0.4 * finbert_conf)  # Weight by confidence

        # Crypto lexicon
        if self.crypto_lexicon:
            crypto_score = self.crypto_lexicon.score_text(text)
            scores.append(crypto_score)
            weights.append(0.3)

        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()

        # Compute weighted average
        polarity = float(np.average(scores, weights=weights))
        subjectivity = float(np.mean(subjectivities)) if subjectivities else 0.5
        confidence = float(np.std(scores) if len(scores) > 1 else abs(polarity))
        confidence = 1.0 - min(confidence, 1.0)  # Lower std = higher confidence

        label = self._polarity_to_label(polarity)
        is_factual = subjectivity < 0.3

        return SentimentResult(
            polarity=polarity,
            subjectivity=subjectivity,
            confidence=confidence,
            is_factual=is_factual,
            sentiment_label=label,
            model_used="ensemble"
        )

    def _polarity_to_label(self, polarity: float) -> str:
        """Convert polarity score to label"""
        if polarity > 0.05:
            return 'positive'
        elif polarity < -0.05:
            return 'negative'
        else:
            return 'neutral'

    def classify_fact_vs_opinion(self, text: str) -> Tuple[bool, float]:
        """
        Classify whether text is factual or opinion/subjective

        Args:
            text: Text to classify

        Returns:
            Tuple of (is_factual, confidence)
        """
        # Use TextBlob subjectivity as baseline
        _, subjectivity = self.analyze_with_textblob(text)

        # Look for factual indicators
        factual_patterns = [
            r'\d+%',  # Percentages
            r'\$\d+',  # Dollar amounts
            r'\d+\s*(billion|million|thousand)',  # Large numbers
            r'(announced|reported|confirmed|stated)',  # Reporting verbs
            r'(according to|based on|data shows)',  # Attribution
        ]

        factual_count = sum(1 for pattern in factual_patterns if re.search(pattern, text, re.IGNORECASE))

        # Opinion indicators
        opinion_patterns = [
            r'(i think|i believe|in my opinion|imho)',
            r'(should|must|need to)',
            r'(bullish|bearish|moon|dump)',  # Crypto opinions
            r'!{2,}',  # Multiple exclamation marks
        ]

        opinion_count = sum(1 for pattern in opinion_patterns if re.search(pattern, text, re.IGNORECASE))

        # Combine signals
        if factual_count > opinion_count and subjectivity < 0.4:
            is_factual = True
            confidence = 0.7 + (factual_count * 0.1)
        elif opinion_count > factual_count or subjectivity > 0.6:
            is_factual = False
            confidence = 0.7 + (opinion_count * 0.1)
        else:
            is_factual = subjectivity < 0.5
            confidence = 0.5

        return is_factual, min(confidence, 1.0)

    def batch_analyze(
        self,
        texts: List[str],
        model: Optional[SentimentModel] = None
    ) -> List[SentimentResult]:
        """
        Analyze multiple texts

        Args:
            texts: List of texts to analyze
            model: Model to use (None for ensemble)

        Returns:
            List of sentiment results
        """
        return [self.analyze(text, model) for text in texts]


def create_sentiment_analyzer(
    use_finbert: bool = False,
    use_vader: bool = True,
    use_crypto_lexicon: bool = True
) -> AdvancedSentimentAnalyzer:
    """
    Factory function to create sentiment analyzer

    Args:
        use_finbert: Enable FinBERT (requires transformers library)
        use_vader: Enable VADER
        use_crypto_lexicon: Enable crypto-specific lexicon

    Returns:
        Configured AdvancedSentimentAnalyzer
    """
    return AdvancedSentimentAnalyzer(
        use_finbert=use_finbert,
        use_vader=use_vader,
        use_crypto_lexicon=use_crypto_lexicon
    )


# Example usage
if __name__ == "__main__":
    analyzer = create_sentiment_analyzer(
        use_finbert=False,  # Set to True if transformers installed
        use_vader=True,
        use_crypto_lexicon=True
    )

    # Test texts
    texts = [
        "Bitcoin is mooning! 🚀 Bullish rally continues!",
        "Major crash incoming. Bearish signals everywhere.",
        "Bitcoin price increased by 5% today following institutional adoption news.",
        "I think ETH will pump soon. DYOR, NFA."
    ]

    for text in texts:
        result = analyzer.analyze(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {result.sentiment_label} (polarity: {result.polarity:.2f})")
        print(f"Subjectivity: {result.subjectivity:.2f}")
        print(f"Is Factual: {result.is_factual}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Model: {result.model_used}")
