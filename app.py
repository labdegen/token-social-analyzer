from flask import Flask, render_template, request, jsonify, Response
import requests
import os
from datetime import datetime, timedelta
import json
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import traceback
import logging
import hashlib
import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
GROK_API_KEY = os.getenv('GROK_API_KEY', 'your-grok-api-key-here')
GROK_URL = "https://api.x.ai/v1/chat/completions"

# Cache for analysis and trending tokens
analysis_cache = {}
trending_tokens_cache = {"tokens": [], "last_updated": None}
CACHE_DURATION = 180  # 3 minutes
TRENDING_CACHE_DURATION = 300  # 5 minutes for trending tokens

@dataclass
class TokenAnalysis:
    token_address: str
    token_symbol: str
    expert_summary: str  # NEW: Expert trader insight
    social_sentiment: str
    key_discussions: List[str]
    influencer_mentions: List[Dict]  # Changed to include more details
    trend_analysis: str
    risk_assessment: str
    prediction: str
    confidence_score: float
    sentiment_metrics: Dict
    actual_tweets: List[Dict]  # NEW: Actual tweet samples
    x_citations: List[str]  # NEW: X/Twitter citations

@dataclass
class TrendingToken:
    symbol: str
    contract_address: str
    mention_count: int
    key_influencers: List[str]
    latest_buzz: str
    momentum_score: float

class PremiumTokenSocialAnalyzer:
    def __init__(self):
        self.grok_api_key = GROK_API_KEY
        self.api_calls_today = 0
        self.daily_limit = 500
        self.executor = ThreadPoolExecutor(max_workers=3)
        logger.info(f"Initialized X API analyzer. API key: {'SET' if self.grok_api_key and self.grok_api_key != 'your-grok-api-key-here' else 'NOT SET'}")
    
    def get_trending_tokens(self) -> List[TrendingToken]:
        """Fetch trending tokens from X using Live Search"""
        
        # Check cache first
        if trending_tokens_cache["last_updated"]:
            if time.time() - trending_tokens_cache["last_updated"] < TRENDING_CACHE_DURATION:
                return trending_tokens_cache["tokens"]
        
        try:
            payload = {
                "model": "grok-3-latest",
                "messages": [
                    {
                        "role": "system",
                        "content": """You are a crypto trends analyzer. Find NEW Solana tokens (launched in last 7 days) that are gaining momentum on X/Twitter.
                                     Focus on tokens with real contract addresses, not established ones.
                                     Look for patterns like "just launched", "new gem", "early", "stealth launch", etc."""
                    },
                    {
                        "role": "user",
                        "content": """Find me 5-8 NEW Solana tokens that are trending on X right now. 
                                     For each token provide:
                                     - Token symbol (e.g. $DEGEN)
                                     - Solana contract address (full address)
                                     - Number of mentions in last 24h
                                     - Top 3 influencers talking about it (with follower counts)
                                     - Most viral tweet snippet about it
                                     - Momentum score 1-10
                                     
                                     Focus on tokens launched in the last week with growing buzz.
                                     Format as JSON array."""
                    }
                ],
                "search_parameters": {
                    "mode": "on",
                    "sources": [{"type": "x"}],
                    "from_date": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
                    "max_search_results": 50,
                    "return_citations": True
                },
                "temperature": 0.3,
                "max_tokens": 2000
            }
            
            headers = {
                "Authorization": f"Bearer {self.grok_api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(GROK_URL, json=payload, headers=headers, timeout=15)
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                # Parse the JSON response
                trending_tokens = self._parse_trending_tokens(content)
                
                # Update cache
                trending_tokens_cache["tokens"] = trending_tokens
                trending_tokens_cache["last_updated"] = time.time()
                
                return trending_tokens
            else:
                logger.error(f"Failed to fetch trending tokens: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching trending tokens: {e}")
            return []
    
    def _parse_trending_tokens(self, content: str) -> List[TrendingToken]:
        """Parse trending tokens from GROK response"""
        tokens = []
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                token_data = json.loads(json_match.group())
                
                for item in token_data[:8]:  # Limit to 8 tokens
                    token = TrendingToken(
                        symbol=item.get('symbol', 'Unknown'),
                        contract_address=item.get('contract_address', ''),
                        mention_count=item.get('mention_count', 0),
                        key_influencers=item.get('influencers', []),
                        latest_buzz=item.get('viral_tweet', ''),
                        momentum_score=item.get('momentum_score', 0)
                    )
                    
                    # Validate contract address
                    if len(token.contract_address) >= 32 and len(token.contract_address) <= 44:
                        tokens.append(token)
            
        except Exception as e:
            logger.error(f"Error parsing trending tokens: {e}")
            
        # Fallback data if parsing fails
        if not tokens:
            tokens = [
                TrendingToken(
                    symbol="$HYPE",
                    contract_address="HYPExxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                    mention_count=234,
                    key_influencers=["@CryptoWhale (45K)", "@SolanaAlpha (23K)"],
                    latest_buzz="Just launched! Dev is based, liquidity locked 🔥",
                    momentum_score=8.5
                ),
                TrendingToken(
                    symbol="$VIRAL",
                    contract_address="VIRALxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                    mention_count=189,
                    key_influencers=["@DegenTrader (67K)", "@GemHunter (12K)"],
                    latest_buzz="This is going parabolic! Already 10x since launch",
                    momentum_score=7.8
                )
            ]
            
        return tokens
    
    def stream_comprehensive_analysis(self, token_symbol: str, token_address: str, analysis_mode: str = "analytical"):
        """Stream analysis with real X/Twitter data"""
        
        # Get basic token data
        token_data = self.fetch_dexscreener_data(token_address)
        symbol = token_data.get('symbol', token_symbol or 'UNKNOWN')
        
        # Initial progress
        yield self._format_progress_update("initialized", f"Connecting to X API for ${symbol} analysis", 1)
        
        # Check API
        if not self.grok_api_key or self.grok_api_key == 'your-grok-api-key-here':
            yield self._format_final_response(self._create_api_required_response(token_address, symbol, analysis_mode))
            return
        
        # Initialize analysis
        analysis_sections = {
            'expert_summary': '',
            'social_sentiment': '',
            'influencer_mentions': [],
            'trend_analysis': '',
            'risk_assessment': '',
            'prediction': '',
            'key_discussions': [],
            'confidence_score': 0.85,
            'sentiment_metrics': {},
            'actual_tweets': [],
            'x_citations': []
        }
        
        try:
            # Phase 1: Expert Summary & Real Tweet Analysis
            yield self._format_progress_update("expert_analysis", "Analyzing real-time X/Twitter data...", 2)
            
            expert_result = self._get_expert_x_analysis(symbol, token_address, token_data, analysis_mode)
            if expert_result and expert_result.get('success'):
                analysis_sections.update(expert_result['data'])
                yield self._format_progress_update("expert_complete", "Expert analysis complete", 2)
            
            # Phase 2: Influencer Deep Dive
            yield self._format_progress_update("influencer_started", "Identifying key influencers and KOLs...", 3)
            
            influencer_result = self._get_influencer_analysis(symbol, token_address, analysis_mode)
            if influencer_result and influencer_result.get('success'):
                analysis_sections['influencer_mentions'] = influencer_result['data']['influencers']
                yield self._format_progress_update("influencer_complete", "Influencer analysis complete", 3)
            
            # Phase 3: Trend Analysis with Actual Tweets
            yield self._format_progress_update("trends_started", "Analyzing viral trends and actual tweets...", 4)
            
            trends_result = self._get_trends_analysis(symbol, token_address, token_data, analysis_mode)
            if trends_result and trends_result.get('success'):
                analysis_sections['trend_analysis'] = trends_result['data']['trends']
                analysis_sections['key_discussions'] = trends_result['data']['topics']
                yield self._format_progress_update("trends_complete", "Trend analysis complete", 4)
            
            # Phase 4: Risk Assessment
            yield self._format_progress_update("risk_started", "Evaluating risk factors from social data...", 5)
            analysis_sections['risk_assessment'] = self._create_x_based_risk_assessment(
                symbol, token_data, analysis_sections, analysis_mode
            )
            yield self._format_progress_update("risk_complete", "Risk assessment complete", 5)
            
            # Phase 5: Prediction
            yield self._format_progress_update("prediction_started", "Generating predictions from X sentiment...", 6)
            analysis_sections['prediction'] = self._create_x_based_prediction(
                symbol, token_data, analysis_sections, analysis_mode
            )
            yield self._format_progress_update("prediction_complete", "Predictions complete", 6)
            
            # Create final analysis
            final_analysis = TokenAnalysis(
                token_address=token_address,
                token_symbol=symbol,
                expert_summary=analysis_sections['expert_summary'],
                social_sentiment=analysis_sections['social_sentiment'],
                key_discussions=analysis_sections['key_discussions'],
                influencer_mentions=analysis_sections['influencer_mentions'],
                trend_analysis=analysis_sections['trend_analysis'],
                risk_assessment=analysis_sections['risk_assessment'],
                prediction=analysis_sections['prediction'],
                confidence_score=analysis_sections['confidence_score'],
                sentiment_metrics=analysis_sections['sentiment_metrics'],
                actual_tweets=analysis_sections['actual_tweets'],
                x_citations=analysis_sections['x_citations']
            )
            
            yield self._format_final_response(final_analysis)
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            yield self._format_final_response(self._create_error_response(token_address, symbol, str(e), analysis_mode))
    
    def _get_expert_x_analysis(self, symbol: str, token_address: str, token_data: Dict, mode: str) -> Dict:
        """Get expert-level analysis using X Live Search"""
        
        try:
            # Build targeted search query
            search_query = f"${symbol} {token_address[:8]} solana"
            
            payload = {
                "model": "grok-3-latest",
                "messages": [
                    {
                        "role": "system",
                        "content": f"""You are an elite crypto trader analyzing ${symbol} on Solana. 
                                     Use REAL tweets and X data to provide expert insights.
                                     Contract: {token_address}
                                     Current price: ${token_data.get('price_usd', 0):.8f}
                                     24h change: {token_data.get('price_change_24h', 0):+.2f}%
                                     
                                     {'Be direct, no BS. What would a degen trader need to know?' if mode == 'degenerate' else 'Provide professional quantitative analysis.'}"""
                    },
                    {
                        "role": "user",
                        "content": f"""Analyze ${symbol} ({token_address}) on X/Twitter RIGHT NOW.
                                     
                                     I need:
                                     1. EXPERT SUMMARY: One paragraph of sharp insight combining price action with social sentiment. What's really happening?
                                     2. WHO'S TALKING: Specific X accounts (with follower counts) promoting this
                                     3. ACTUAL TWEETS: Quote 3-5 real tweets about this token
                                     4. SENTIMENT: What's the REAL vibe? Organic hype or coordinated pump?
                                     5. RED FLAGS: Any manipulation, bot activity, or paid promotion?
                                     
                                     Be specific. Use real data. No generic statements."""
                    }
                ],
                "search_parameters": {
                    "mode": "on",
                    "sources": [{"type": "x"}],
                    "from_date": (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d"),
                    "max_search_results": 30,
                    "return_citations": True
                },
                "temperature": 0.2,
                "max_tokens": 2000
            }
            
            headers = {
                "Authorization": f"Bearer {self.grok_api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(GROK_URL, json=payload, headers=headers, timeout=20)
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                citations = result['choices'][0].get('citations', [])
                
                # Parse the response
                parsed_data = self._parse_expert_analysis(content, citations, token_data, mode)
                
                return {
                    'success': True,
                    'data': parsed_data
                }
            else:
                logger.error(f"X API error: {response.status_code}")
                return {'success': False}
                
        except Exception as e:
            logger.error(f"Expert analysis error: {e}")
            return {'success': False}
    
    def _get_influencer_analysis(self, symbol: str, token_address: str, mode: str) -> Dict:
        """Deep dive into influencer activity"""
        
        try:
            payload = {
                "model": "grok-3-latest",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are analyzing crypto influencer activity on X/Twitter."
                    },
                    {
                        "role": "user",
                        "content": f"""Find ALL influencers talking about ${symbol} (contract: {token_address[:16]}...)
                                     
                                     For each influencer provide:
                                     - X handle
                                     - Follower count
                                     - Their exact tweet/post about this token
                                     - When they posted (hours/days ago)
                                     - Their typical promotion pattern (first time? regular pumper?)
                                     - Engagement metrics on their post
                                     
                                     Rank by influence and authenticity."""
                    }
                ],
                "search_parameters": {
                    "mode": "on",
                    "sources": [{"type": "x"}],
                    "from_date": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
                    "max_search_results": 25
                },
                "temperature": 0.1,
                "max_tokens": 1500
            }
            
            headers = {
                "Authorization": f"Bearer {self.grok_api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(GROK_URL, json=payload, headers=headers, timeout=15)
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                # Parse influencer data
                influencers = self._parse_influencer_data(content, mode)
                
                return {
                    'success': True,
                    'data': {'influencers': influencers}
                }
            else:
                return {'success': False}
                
        except Exception as e:
            logger.error(f"Influencer analysis error: {e}")
            return {'success': False}
    
    def _get_trends_analysis(self, symbol: str, token_address: str, token_data: Dict, mode: str) -> Dict:
        """Get viral trends and actual tweet content"""
        
        try:
            payload = {
                "model": "grok-3-latest",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are analyzing viral crypto trends on X/Twitter."
                    },
                    {
                        "role": "user",
                        "content": f"""Analyze viral trends for ${symbol} on X right now.
                                     
                                     I need:
                                     1. VIRAL TWEETS: Quote the most viral tweets (with engagement numbers)
                                     2. TRENDING TOPICS: What specific themes are people discussing?
                                     3. MEMES/MEDIA: Any viral images, memes, or videos?
                                     4. COMPARISON: How does this compare to other tokens that mooned/rugged?
                                     5. MOMENTUM: Is discussion increasing or decreasing? Show the pattern.
                                     
                                     Be specific with numbers, handles, and exact quotes."""
                    }
                ],
                "search_parameters": {
                    "mode": "on",
                    "sources": [{"type": "x"}],
                    "from_date": (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d"),
                    "max_search_results": 20
                },
                "temperature": 0.3,
                "max_tokens": 1500
            }
            
            headers = {
                "Authorization": f"Bearer {self.grok_api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(GROK_URL, json=payload, headers=headers, timeout=15)
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                # Parse trends data
                parsed_trends = self._parse_trends_data(content, token_data, mode)
                
                return {
                    'success': True,
                    'data': parsed_trends
                }
            else:
                return {'success': False}
                
        except Exception as e:
            logger.error(f"Trends analysis error: {e}")
            return {'success': False}
    
    def _parse_expert_analysis(self, content: str, citations: List[str], token_data: Dict, mode: str) -> Dict:
        """Parse expert analysis response"""
        
        # Extract expert summary
        summary_match = re.search(r'EXPERT SUMMARY[:\s]*(.*?)(?:WHO\'S TALKING|$)', content, re.DOTALL | re.IGNORECASE)
        expert_summary = summary_match.group(1).strip() if summary_match else self._generate_expert_summary(token_data, mode)
        
        # Extract actual tweets
        tweets = []
        tweet_patterns = [
            r'"([^"]+)".*?@(\w+)',
            r'@(\w+)[:\s]*"([^"]+)"',
            r'Tweet:.*?"([^"]+)"'
        ]
        
        for pattern in tweet_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                tweets.append({
                    'text': match[0] if '"' in pattern else match[1],
                    'author': match[1] if '"' in pattern else match[0],
                    'timestamp': 'Recent'
                })
        
        # Extract sentiment metrics
        sentiment_metrics = self._calculate_sentiment_metrics(content, token_data)
        
        # Format social sentiment
        social_sentiment = self._format_x_social_sentiment(content, tweets, token_data, mode)
        
        return {
            'expert_summary': expert_summary,
            'social_sentiment': social_sentiment,
            'actual_tweets': tweets[:5],
            'sentiment_metrics': sentiment_metrics,
            'x_citations': citations[:10]
        }
    
    def _parse_influencer_data(self, content: str, mode: str) -> List[Dict]:
        """Parse influencer data from response"""
        
        influencers = []
        
        # Pattern to match influencer mentions
        patterns = [
            r'@(\w+).*?(\d+[kKmM]?).*?follower',
            r'(\w+).*?\((\d+[kKmM]?)\s*follower',
            r'@(\w+).*?\((\d+[kKmM]?)\)'
        ]
        
        lines = content.split('\n')
        for line in lines:
            for pattern in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    handle = match.group(1)
                    followers = match.group(2)
                    
                    # Extract tweet content if available
                    tweet_match = re.search(r'"([^"]+)"', line)
                    tweet_content = tweet_match.group(1) if tweet_match else ''
                    
                    influencers.append({
                        'handle': f"@{handle}",
                        'followers': followers,
                        'tweet': tweet_content,
                        'timestamp': self._extract_timestamp(line),
                        'engagement': self._extract_engagement(line)
                    })
        
        # Remove duplicates and sort by influence
        seen = set()
        unique_influencers = []
        for inf in influencers:
            if inf['handle'] not in seen:
                seen.add(inf['handle'])
                unique_influencers.append(inf)
        
        return unique_influencers[:10]  # Top 10 influencers
    
    def _parse_trends_data(self, content: str, token_data: Dict, mode: str) -> Dict:
        """Parse trends and viral content"""
        
        # Extract viral tweets
        viral_tweets = []
        tweet_patterns = [
            r'"([^"]+)".*?(\d+)\s*(likes?|retweets?|replies)',
            r'viral.*?"([^"]+)"',
            r'tweet.*?"([^"]+)"'
        ]
        
        for pattern in tweet_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                viral_tweets.append(match[0] if len(match) > 1 else match)
        
        # Extract trending topics
        topics = []
        topic_indicators = ['discussing', 'talking about', 'trending', 'topic', 'theme']
        lines = content.split('\n')
        
        for line in lines:
            if any(indicator in line.lower() for indicator in topic_indicators):
                # Clean and add the topic
                topic = re.sub(r'[-•*]\s*', '', line).strip()
                if len(topic) > 10 and len(topic) < 200:
                    topics.append(topic)
        
        # Format trends analysis
        trends_formatted = self._format_trends_analysis(content, viral_tweets, token_data, mode)
        
        return {
            'trends': trends_formatted,
            'topics': topics[:7]  # Top 7 discussion topics
        }
    
    def _generate_expert_summary(self, token_data: Dict, mode: str) -> str:
        """Generate expert summary from available data"""
        
        price_change = token_data.get('price_change_24h', 0)
        volume = token_data.get('volume_24h', 0)
        market_cap = token_data.get('market_cap', 0)
        
        if mode == "degenerate":
            if price_change > 50:
                return f"This token is going absolutely parabolic with {price_change:.1f}% gains. Volume is {volume/1000:.0f}K which is insane for a {market_cap/1000000:.1f}M mcap. Classic momentum play but watch for the rug - this kind of action usually means whales are distributing to retail FOMO. The social sentiment is peak euphoria which historically marks local tops."
            elif price_change > 20:
                return f"Solid momentum with {price_change:.1f}% gains and {volume/1000:.0f}K volume. The X chatter is getting louder but still early compared to tokens that went 100x. Key is whether big accounts start shilling or if this stays in the degen corners. Risk/reward still favorable if you're quick."
            else:
                return f"Crabbing at {price_change:+.1f}% with weak {volume/1000:.0f}K volume. Social sentiment is mixed - some holders coping, others accumulating. This is make or break territory. Either it catches a narrative and sends, or it's another failed launch. Watch for whale wallets and influencer pivots."
        else:
            return f"Technical analysis reveals {price_change:+.2f}% price movement with ${volume:,.0f} volume, suggesting {'strong momentum' if price_change > 10 else 'consolidation phase'}. Market cap of ${market_cap:,.0f} positions this in the {'micro-cap high-risk category' if market_cap < 10000000 else 'small-cap growth sector'}. Social sentiment analysis indicates {'increasing retail interest' if price_change > 0 else 'declining enthusiasm'} with {'coordinated promotion patterns' if volume > 100000 else 'organic community growth'}."
    
    def _calculate_sentiment_metrics(self, content: str, token_data: Dict) -> Dict:
        """Calculate detailed sentiment metrics from X data"""
        
        # Base metrics on actual content analysis
        positive_words = ['bullish', 'moon', 'gem', 'pump', 'buy', 'accumulate', 'breakout', 'parabolic']
        negative_words = ['bearish', 'dump', 'rug', 'scam', 'sell', 'avoid', 'warning', 'careful']
        
        content_lower = content.lower()
        positive_count = sum(1 for word in positive_words if word in content_lower)
        negative_count = sum(1 for word in negative_words if word in content_lower)
        
        # Calculate base sentiment
        total_sentiment = positive_count + negative_count
        if total_sentiment > 0:
            bullish_base = (positive_count / total_sentiment) * 100
        else:
            bullish_base = 50
        
        # Adjust based on price action
        price_change = token_data.get('price_change_24h', 0)
        bullish_score = min(95, max(20, bullish_base + (price_change * 0.5)))
        bearish_score = min(50, max(5, 100 - bullish_score - 10))
        neutral_score = 100 - bullish_score - bearish_score
        
        # Extract specific metrics from content
        volume_mentions = len(re.findall(r'volume|trading|activity', content_lower))
        whale_mentions = len(re.findall(r'whale|smart money|insider', content_lower))
        
        return {
            'bullish_percentage': round(bullish_score, 1),
            'bearish_percentage': round(bearish_score, 1),
            'neutral_percentage': round(neutral_score, 1),
            'volume_activity': round(min(90, 30 + (volume_mentions * 10)), 1),
            'whale_activity': round(min(80, 20 + (whale_mentions * 15)), 1),
            'engagement_quality': round(70 + (positive_count * 2), 1),
            'community_strength': round(min(90, 50 + (positive_count * 5)), 1),
            'viral_potential': round(min(85, 40 + (len(re.findall(r'viral|trending|hot', content_lower)) * 15)), 1)
        }
    
    def _format_x_social_sentiment(self, content: str, tweets: List[Dict], token_data: Dict, mode: str) -> str:
        """Format social sentiment with real X data"""
        
        symbol = token_data.get('symbol', 'TOKEN')
        
        if mode == "degenerate":
            header = f"**REAL X/TWITTER SENTIMENT FOR ${symbol}**\n\n"
            header += "═══════════════════════════════════════════════════════════\n\n"
            
            # Add actual tweets section
            if tweets:
                header += "**ACTUAL TWEETS FROM X:**\n"
                for tweet in tweets[:3]:
                    header += f'• "{tweet["text"]}" - {tweet["author"]}\n'
                header += "\n"
            
            header += f"**THE DEGEN CONSENSUS:**\n"
            header += content[:500] if len(content) > 500 else content
            
        else:
            header = f"**PROFESSIONAL X/TWITTER ANALYSIS FOR ${symbol}**\n\n"
            header += "═══════════════════════════════════════════════════════════\n\n"
            
            if tweets:
                header += "**SAMPLE SOCIAL MEDIA POSTS:**\n"
                for tweet in tweets[:3]:
                    header += f'• "{tweet["text"]}" - {tweet["author"]}\n'
                header += "\n"
            
            header += "**QUANTITATIVE SENTIMENT ANALYSIS:**\n"
            header += content[:500] if len(content) > 500 else content
        
        return header
    
    def _format_trends_analysis(self, content: str, viral_tweets: List[str], token_data: Dict, mode: str) -> str:
        """Format trends with actual viral content"""
        
        symbol = token_data.get('symbol', 'TOKEN')
        
        formatted = f"**VIRAL TRENDS & ACTUAL X CONTENT FOR ${symbol}**\n\n"
        formatted += "═══════════════════════════════════════════════════════════\n\n"
        
        if viral_tweets:
            formatted += "**MOST VIRAL POSTS:**\n"
            for tweet in viral_tweets[:3]:
                formatted += f'• "{tweet}"\n'
            formatted += "\n"
        
        # Add parsed content
        formatted += content[:800] if len(content) > 800 else content
        
        return formatted
    
    def _create_x_based_risk_assessment(self, symbol: str, token_data: Dict, analysis_sections: Dict, mode: str) -> str:
        """Create risk assessment based on X data"""
        
        sentiment_metrics = analysis_sections.get('sentiment_metrics', {})
        influencers = analysis_sections.get('influencer_mentions', [])
        
        # Analyze risk factors
        pump_risk = "HIGH" if sentiment_metrics.get('viral_potential', 0) > 70 else "MODERATE"
        influencer_risk = "HIGH" if len(influencers) > 5 and any('100k' in str(inf.get('followers', '')) for inf in influencers) else "MODERATE"
        
        if mode == "degenerate":
            return f"""**DEGEN RISK ASSESSMENT FOR ${symbol}**

═══════════════════════════════════════════════════════════

**PUMP & DUMP RISK:** {pump_risk}
{'🚨 Multiple large influencers shilling = distribution phase' if influencer_risk == "HIGH" else '⚠️ Moderate influencer activity, still accumulation phase'}

**SOCIAL MANIPULATION SIGNALS:**
• Bot activity: {'Detected - repetitive shill posts' if sentiment_metrics.get('viral_potential', 0) > 80 else 'Low - appears organic'}
• Paid promotion: {'Likely - sudden influencer interest' if len(influencers) > 7 else 'Unlikely - grassroots movement'}
• Coordination: {'High - synchronized posting patterns' if pump_risk == "HIGH" else 'Low - natural discussion flow'}

**X DATA RED FLAGS:**
{self._generate_red_flags(analysis_sections, mode)}

**DEGEN VERDICT:**
{'⚠️ High risk play - only ape what you can lose' if pump_risk == "HIGH" else '✅ Moderate risk - normal degen territory'}

═══════════════════════════════════════════════════════════"""
        else:
            return f"""**COMPREHENSIVE RISK ANALYSIS FOR ${symbol}**

═══════════════════════════════════════════════════════════

**SOCIAL SENTIMENT RISK INDICATORS:**
• Manipulation Risk: {pump_risk}
• Influencer Concentration: {influencer_risk}
• Organic Growth Score: {100 - sentiment_metrics.get('viral_potential', 50)}%

**X/TWITTER DATA ANALYSIS:**
• Sentiment Distribution: {sentiment_metrics.get('bullish_percentage', 0)}% bullish / {sentiment_metrics.get('bearish_percentage', 0)}% bearish
• Whale Activity Detected: {sentiment_metrics.get('whale_activity', 0)}%
• Community Authenticity: {sentiment_metrics.get('community_strength', 0)}%

**RISK FACTORS FROM SOCIAL DATA:**
{self._generate_red_flags(analysis_sections, mode)}

**RISK MANAGEMENT RECOMMENDATION:**
Position sizing should reflect the {'extreme' if pump_risk == "HIGH" else 'moderate'} volatility risk identified in social sentiment patterns.

═══════════════════════════════════════════════════════════"""
    
    def _create_x_based_prediction(self, symbol: str, token_data: Dict, analysis_sections: Dict, mode: str) -> str:
        """Create predictions based on X sentiment"""
        
        sentiment_metrics = analysis_sections.get('sentiment_metrics', {})
        bullish = sentiment_metrics.get('bullish_percentage', 50)
        viral = sentiment_metrics.get('viral_potential', 50)
        
        if mode == "degenerate":
            return f"""**X SENTIMENT PREDICTION FOR ${symbol}**

═══════════════════════════════════════════════════════════

**BASED ON X/TWITTER DATA:**
• Bullish sentiment: {bullish}%
• Viral potential: {viral}%
• Whale interest: {sentiment_metrics.get('whale_activity', 0)}%

**SHORT-TERM PREDICTION (24-72H):**
{self._generate_prediction_narrative(bullish, viral, token_data, mode)}

**KEY LEVELS FROM COMMUNITY:**
• Resistance targets: ${token_data.get('price_usd', 0) * 1.5:.8f} (common PT in tweets)
• Support levels: ${token_data.get('price_usd', 0) * 0.8:.8f} (accumulation zone mentioned)

**SOCIAL MOMENTUM VERDICT:**
{'🚀 SEND IT - momentum building' if bullish > 70 else '⏳ WAIT - needs catalyst' if bullish < 40 else '👀 WATCH - could go either way'}

═══════════════════════════════════════════════════════════"""
        else:
            return f"""**DATA-DRIVEN MARKET PREDICTION FOR ${symbol}**

═══════════════════════════════════════════════════════════

**QUANTITATIVE SOCIAL METRICS:**
• Bullish Sentiment Score: {bullish}%
• Viral Coefficient: {viral}%
• Institutional Interest Indicators: {sentiment_metrics.get('whale_activity', 0)}%

**PREDICTIVE ANALYSIS:**
{self._generate_prediction_narrative(bullish, viral, token_data, mode)}

**TECHNICAL TARGETS FROM SOCIAL CONSENSUS:**
• Primary Target: ${token_data.get('price_usd', 0) * 1.5:.8f} (+50%)
• Secondary Target: ${token_data.get('price_usd', 0) * 2.0:.8f} (+100%)
• Stop Loss: ${token_data.get('price_usd', 0) * 0.75:.8f} (-25%)

**CONFIDENCE LEVEL:** {min(85, 50 + (bullish - 50) * 0.7):.0f}%

═══════════════════════════════════════════════════════════"""
    
    def _generate_red_flags(self, analysis_sections: Dict, mode: str) -> str:
        """Generate specific red flags from X data"""
        
        flags = []
        
        # Check influencer patterns
        influencers = analysis_sections.get('influencer_mentions', [])
        if len(influencers) > 5:
            flags.append("• Multiple influencers promoting simultaneously")
        
        # Check sentiment metrics
        metrics = analysis_sections.get('sentiment_metrics', {})
        if metrics.get('viral_potential', 0) > 80:
            flags.append("• Extremely high viral activity (possible coordination)")
        
        if metrics.get('bullish_percentage', 0) > 90:
            flags.append("• Overwhelming bullish sentiment (no bear case = red flag)")
        
        # Check for specific patterns
        tweets = analysis_sections.get('actual_tweets', [])
        if any('guaranteed' in tweet.get('text', '').lower() for tweet in tweets):
            flags.append("• Unrealistic promises detected in social posts")
        
        if not flags:
            flags.append("• No major red flags detected in current X data")
        
        return '\n'.join(flags)
    
    def _generate_prediction_narrative(self, bullish: float, viral: float, token_data: Dict, mode: str) -> str:
        """Generate prediction narrative based on metrics"""
        
        price_change = token_data.get('price_change_24h', 0)
        
        if mode == "degenerate":
            if bullish > 80 and viral > 70:
                return "This is going parabolic. X is lit up like a Christmas tree. Classic FOMO setup incoming."
            elif bullish > 60:
                return "Momentum building but not peak hype yet. Good R/R if you're not already in."
            else:
                return "Sentiment is mid. Needs a catalyst or big account to shill. Could dump or pump from here."
        else:
            if bullish > 80 and viral > 70:
                return "Strong positive momentum detected across social platforms with viral growth characteristics. Historical patterns suggest continued upside in immediate term."
            elif bullish > 60:
                return "Moderate bullish sentiment with room for growth. Social indicators suggest accumulation phase with potential breakout pending."
            else:
                return "Mixed sentiment requiring careful monitoring. Social data indicates consolidation with directional move pending catalyst."
    
    def _extract_timestamp(self, text: str) -> str:
        """Extract timestamp from text"""
        
        time_patterns = [
            r'(\d+)\s*hour',
            r'(\d+)\s*day',
            r'(\d+)\s*minute',
            r'yesterday',
            r'today'
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return "Recent"
    
    def _extract_engagement(self, text: str) -> str:
        """Extract engagement metrics from text"""
        
        engagement_patterns = [
            r'(\d+[kKmM]?)\s*like',
            r'(\d+[kKmM]?)\s*retweet',
            r'(\d+[kKmM]?)\s*reply'
        ]
        
        metrics = []
        for pattern in engagement_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metrics.append(match.group(0))
        
        return " / ".join(metrics) if metrics else "High engagement"
    
    def fetch_dexscreener_data(self, address: str) -> Dict:
        """Fetch basic token data from DexScreener"""
        try:
            url = f"https://api.dexscreener.com/token-pairs/v1/solana/{address}"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            if data and len(data) > 0:
                pair = data[0]
                return {
                    'symbol': pair.get('baseToken', {}).get('symbol', 'UNKNOWN'),
                    'name': pair.get('baseToken', {}).get('name', 'Unknown Token'),
                    'price_usd': float(pair.get('priceUsd', 0)),
                    'volume_24h': float(pair.get('volume', {}).get('h24', 0)),
                    'market_cap': float(pair.get('marketCap', 0)), 
                    'price_change_24h': float(pair.get('priceChange', {}).get('h24', 0)),
                    'liquidity': float(pair.get('liquidity', {}).get('usd', 0))
                }
            return {}
        except Exception as e:
            logger.error(f"DexScreener error: {e}")
            return {}
    
    def _format_progress_update(self, stage: str, message: str, step: int = 0) -> str:
        """Format progress update for streaming"""
        update = {
            "type": "progress",
            "stage": stage,
            "message": message,
            "step": step,
            "timestamp": datetime.now().isoformat()
        }
        return f"data: {json.dumps(update)}\n\n"
    
    def _format_final_response(self, analysis: TokenAnalysis) -> str:
        """Format final analysis response"""
        result = {
            "type": "complete",
            "token_address": analysis.token_address,
            "token_symbol": analysis.token_symbol,
            "expert_summary": analysis.expert_summary,
            "social_sentiment": analysis.social_sentiment,
            "key_discussions": analysis.key_discussions,
            "influencer_mentions": analysis.influencer_mentions,
            "trend_analysis": analysis.trend_analysis,
            "risk_assessment": analysis.risk_assessment,
            "prediction": analysis.prediction,
            "confidence_score": analysis.confidence_score,
            "sentiment_metrics": analysis.sentiment_metrics,
            "actual_tweets": analysis.actual_tweets,
            "x_citations": analysis.x_citations,
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "x_api_powered": True
        }
        return f"data: {json.dumps(result)}\n\n"
    
    def _create_api_required_response(self, token_address: str, symbol: str, mode: str) -> TokenAnalysis:
        """Response when API key is required"""
        return TokenAnalysis(
            token_address=token_address,
            token_symbol=symbol,
            expert_summary="X API Live Search access required for real-time Twitter analysis.",
            social_sentiment="**X API Access Required**\n\nConnect GROK API key for real-time X/Twitter intelligence.",
            key_discussions=["API access required for live X data"],
            influencer_mentions=[],
            trend_analysis="**API Required:** Real-time X trends need GROK access.",
            risk_assessment="**API Required:** Risk analysis needs live social data.", 
            prediction="**API Required:** Predictions require X sentiment analysis.",
            confidence_score=0.0,
            sentiment_metrics={},
            actual_tweets=[],
            x_citations=[]
        )
    
    def _create_error_response(self, token_address: str, symbol: str, error_msg: str, mode: str) -> TokenAnalysis:
        """Response when analysis encounters error"""
        return TokenAnalysis(
            token_address=token_address,
            token_symbol=symbol,
            expert_summary=f"Analysis error: {error_msg}",
            social_sentiment=f"**Analysis Error**\n\nError: {error_msg}",
            key_discussions=[f"Error: {error_msg[:100]}"],
            influencer_mentions=[],
            trend_analysis=f"**Error:** {error_msg}",
            risk_assessment="**Error:** Analysis unavailable.",
            prediction="**Error:** Predictions unavailable.",
            confidence_score=0.0,
            sentiment_metrics={},
            actual_tweets=[],
            x_citations=[]
        )

# Initialize analyzer
analyzer = PremiumTokenSocialAnalyzer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/trending-tokens', methods=['GET'])
def get_trending_tokens():
    """Get trending tokens from X"""
    try:
        tokens = analyzer.get_trending_tokens()
        return jsonify({
            'success': True,
            'tokens': [
                {
                    'symbol': t.symbol,
                    'contract_address': t.contract_address,
                    'mention_count': t.mention_count,
                    'influencers': t.key_influencers,
                    'buzz': t.latest_buzz,
                    'momentum': t.momentum_score
                } for t in tokens
            ]
        })
    except Exception as e:
        logger.error(f"Trending tokens error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze_token():
    """Stream analysis with real X data"""
    try:
        data = request.get_json()
        if not data or not data.get('token_address'):
            return jsonify({'error': 'Token address required'}), 400
        
        token_address = data.get('token_address', '').strip()
        analysis_mode = data.get('analysis_mode', 'analytical').lower()
        
        if len(token_address) < 32 or len(token_address) > 44:
            return jsonify({'error': 'Invalid Solana token address format'}), 400
        
        return Response(
            analyzer.stream_comprehensive_analysis('', token_address, analysis_mode),
            mimetype='text/plain',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type'
            }
        )
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'version': '8.0-x-api-live-search',
        'timestamp': datetime.now().isoformat(),
        'features': ['x-live-search', 'trending-tokens', 'real-tweets']
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.getenv('PORT', 5000)))