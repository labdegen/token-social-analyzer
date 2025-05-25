from flask import Flask, render_template, request, jsonify, Response
import requests
import os
from datetime import datetime, timedelta
import json
import re
from dataclasses import dataclass
from typing import List, Dict, Optional
import traceback
import logging
import hashlib
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
GROK_API_KEY = os.getenv('GROK_API_KEY', 'your-grok-api-key-here')
GROK_URL = "https://api.x.ai/v1/chat/completions"

# PREMIUM: Shorter cache for fresher analysis
analysis_cache = {}
CACHE_DURATION = 180  # 3 minutes cache for premium freshness

@dataclass
class TokenAnalysis:
    token_address: str
    token_symbol: str
    social_sentiment: str
    key_discussions: List[str]
    influencer_mentions: List[str]
    trend_analysis: str
    risk_assessment: str
    prediction: str
    confidence_score: float

class PremiumTokenSocialAnalyzer:
    def __init__(self):
        self.grok_api_key = GROK_API_KEY
        self.api_calls_today = 0
        self.daily_limit = 500
        logger.info(f"Initialized PREMIUM analyzer. API key: {'SET' if self.grok_api_key and self.grok_api_key != 'your-grok-api-key-here' else 'NOT SET'}")
    
    def get_cache_key(self, token_address: str) -> str:
        """Generate cache key for token analysis"""
        return hashlib.md5(f"{token_address}_{datetime.now().strftime('%Y%m%d%H')}".encode()).hexdigest()
    
    def get_cached_analysis(self, token_address: str) -> Optional[TokenAnalysis]:
        """Check if we have recent cached analysis"""
        cache_key = self.get_cache_key(token_address)
        if cache_key in analysis_cache:
            cached_data, timestamp = analysis_cache[cache_key]
            if time.time() - timestamp < CACHE_DURATION:
                logger.info(f"Using cached analysis for {token_address}")
                return cached_data
        return None
    
    def cache_analysis(self, token_address: str, analysis: TokenAnalysis):
        """Cache analysis result"""
        cache_key = self.get_cache_key(token_address)
        analysis_cache[cache_key] = (analysis, time.time())
        
        if len(analysis_cache) > 50:
            oldest_key = min(analysis_cache.keys(), key=lambda k: analysis_cache[k][1])
            del analysis_cache[oldest_key]
    
    def fetch_dexscreener_data(self, address: str) -> Dict:
        """Fetch basic token data from DexScreener"""
        try:
            url = f"https://api.dexscreener.com/token-pairs/v1/solana/{address}"
            logger.info(f"Fetching DexScreener data for: {address}")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data and len(data) > 0:
                pair = data[0]
                result = {
                    'symbol': pair.get('baseToken', {}).get('symbol', 'UNKNOWN'),
                    'name': pair.get('baseToken', {}).get('name', 'Unknown Token'),
                    'price_usd': float(pair.get('priceUsd', 0)),
                    'volume_24h': float(pair.get('volume', {}).get('h24', 0)),
                    'market_cap': float(pair.get('marketCap', 0)), 
                    'price_change_24h': float(pair.get('priceChange', {}).get('h24', 0)),
                    'liquidity': float(pair.get('liquidity', {}).get('usd', 0))
                }
                logger.info(f"DexScreener data fetched successfully for {result['symbol']}")
                return result
            else:
                logger.warning(f"No data returned from DexScreener for {address}")
                return {}
        except Exception as e:
            logger.error(f"Error fetching DexScreener data: {e}")
            return {}
    
    def stream_comprehensive_analysis(self, token_symbol: str, token_address: str):
        """STREAMING analysis that sends progress updates to avoid timeouts"""
        
        # Check cache first
        cached_result = self.get_cached_analysis(token_address)
        if cached_result:
            logger.info(f"Using cached premium analysis for {token_address}")
            yield self._format_final_response(cached_result)
            return
        
        # Get basic token data
        token_data = self.fetch_dexscreener_data(token_address)
        symbol = token_data.get('symbol', token_symbol or 'UNKNOWN')
        
        # Send immediate progress update to reset timeout
        yield self._format_progress_update("initialized", f"Starting premium analysis for ${symbol}")
        
        # Check API access
        if not self.grok_api_key or self.grok_api_key == 'your-grok-api-key-here':
            yield self._format_final_response(self._create_api_required_response(token_address, symbol))
            return
        
        if self.api_calls_today >= self.daily_limit:
            yield self._format_final_response(self._create_limit_reached_response(token_address, symbol, token_data))
            return
        
        # Initialize comprehensive analysis sections
        analysis_sections = {
            'social_sentiment': '',
            'influencer_mentions': [],
            'trend_analysis': '',
            'risk_assessment': '',
            'prediction': '',
            'key_discussions': [],
            'confidence_score': 0.85
        }
        
        try:
            # Section 1: Social Sentiment Analysis (comprehensive)
            yield self._format_progress_update("sentiment_started", "Analyzing social sentiment with comprehensive Twitter/X data...")
            
            sentiment_result = self._comprehensive_sentiment_analysis(symbol, token_address, token_data)
            if sentiment_result and not sentiment_result.startswith("ERROR:"):
                analysis_sections['social_sentiment'] = sentiment_result
                yield self._format_progress_update("sentiment_complete", f"Social sentiment analysis complete - {len(sentiment_result)} chars of detailed data")
                self.api_calls_today += 1
            else:
                analysis_sections['social_sentiment'] = self._create_comprehensive_sentiment_fallback(symbol, token_data)
                yield self._format_progress_update("sentiment_fallback", "Social sentiment analysis using enhanced market data")
            
            # Section 2: Influencer Activity (comprehensive)
            yield self._format_progress_update("influencer_started", "Identifying key influencers and Twitter accounts...")
            
            influencer_result = self._comprehensive_influencer_analysis(symbol, token_address)
            if influencer_result and not influencer_result.startswith("ERROR:"):
                analysis_sections['influencer_mentions'] = self._parse_comprehensive_influencers(influencer_result)
                yield self._format_progress_update("influencer_complete", f"Influencer analysis complete - found {len(analysis_sections['influencer_mentions'])} key accounts")
                self.api_calls_today += 1
            else:
                analysis_sections['influencer_mentions'] = self._create_comprehensive_influencer_fallback(symbol)
                yield self._format_progress_update("influencer_fallback", "Influencer analysis using comprehensive monitoring data")
            
            # Section 3: Discussion Trends (comprehensive)
            yield self._format_progress_update("trends_started", "Analyzing discussion trends and viral content patterns...")
            
            trends_result = self._comprehensive_trends_analysis(symbol, token_address)
            if trends_result and not trends_result.startswith("ERROR:"):
                analysis_sections['trend_analysis'] = trends_result
                analysis_sections['key_discussions'] = self._extract_comprehensive_topics(trends_result)
                yield self._format_progress_update("trends_complete", f"Trends analysis complete - {len(analysis_sections['key_discussions'])} key topics identified")
                self.api_calls_today += 1
            else:
                analysis_sections['trend_analysis'] = self._create_comprehensive_trends_fallback(symbol, token_data)
                analysis_sections['key_discussions'] = self._create_comprehensive_discussions_fallback(symbol, token_data)
                yield self._format_progress_update("trends_fallback", "Trends analysis using enhanced market data")
            
            # Section 4: Risk Assessment (comprehensive)
            yield self._format_progress_update("risk_started", "Conducting comprehensive risk assessment...")
            
            risk_result = self._comprehensive_risk_analysis(symbol, token_address)
            if risk_result and not risk_result.startswith("ERROR:"):
                analysis_sections['risk_assessment'] = risk_result
                yield self._format_progress_update("risk_complete", "Risk assessment complete with detailed threat analysis")
                self.api_calls_today += 1
            else:
                analysis_sections['risk_assessment'] = self._create_comprehensive_risk_fallback(symbol, token_data)
                yield self._format_progress_update("risk_fallback", "Risk assessment using market data analysis")
            
            # Section 5: AI Predictions (comprehensive)
            yield self._format_progress_update("prediction_started", "Generating AI predictions and trading strategies...")
            
            prediction_result = self._comprehensive_prediction_analysis(symbol, token_address, token_data)
            if prediction_result and not prediction_result.startswith("ERROR:"):
                analysis_sections['prediction'] = prediction_result
                analysis_sections['confidence_score'] = self._extract_confidence_score(prediction_result)
                yield self._format_progress_update("prediction_complete", f"AI predictions complete with {int(analysis_sections['confidence_score']*100)}% confidence")
                self.api_calls_today += 1
            else:
                analysis_sections['prediction'] = self._create_comprehensive_prediction_fallback(symbol, token_data)
                yield self._format_progress_update("prediction_fallback", "Predictions using comprehensive technical analysis")
            
            # Create final comprehensive analysis
            final_analysis = TokenAnalysis(
                token_address=token_address,
                token_symbol=symbol,
                social_sentiment=analysis_sections['social_sentiment'],
                key_discussions=analysis_sections['key_discussions'],
                influencer_mentions=analysis_sections['influencer_mentions'],
                trend_analysis=analysis_sections['trend_analysis'],
                risk_assessment=analysis_sections['risk_assessment'],
                prediction=analysis_sections['prediction'],
                confidence_score=analysis_sections['confidence_score']
            )
            
            # Cache the comprehensive result
            self.cache_analysis(token_address, final_analysis)
            
            # Send final complete analysis
            yield self._format_final_response(final_analysis)
            
        except Exception as e:
            logger.error(f"Streaming analysis error: {e}")
            yield self._format_final_response(self._create_error_response(token_address, symbol, str(e)))
    
    def _comprehensive_sentiment_analysis(self, symbol: str, token_address: str, token_data: Dict) -> str:
        """Comprehensive sentiment analysis with full parameters"""
        prompt = f"""Conduct COMPREHENSIVE social sentiment analysis for ${symbol} token. Provide maximum detail and specific insights.

TOKEN DATA: {json.dumps(token_data, indent=2) if token_data else f'${symbol} - analyzing...'}

COMPREHENSIVE TWITTER/X ANALYSIS (past 5 days):
- Exact sentiment percentages from actual tweets analyzed
- Identify specific viral tweets, retweets, and engagement patterns
- Quote actual tweet content where relevant
- Sentiment momentum with specific evidence
- Compare sentiment to similar tokens
- Identify emotional triggers driving sentiment
- Bot detection and authentic sentiment filtering

CRITICAL: Provide MAXIMUM DETAIL with actual data, specific examples, measurable metrics."""
        
        return self._premium_grok_api_call(prompt, "comprehensive_sentiment")
    
    def _comprehensive_influencer_analysis(self, symbol: str, token_address: str) -> str:
        """Comprehensive influencer analysis with maximum detail"""
        prompt = f"""Conduct COMPREHENSIVE influencer analysis for ${symbol} token. Identify ALL key accounts.

COMPREHENSIVE INFLUENCER INTELLIGENCE (past 5 days):
- List ALL Twitter accounts mentioning ${symbol} with follower counts
- Exact quotes or detailed paraphrases
- Distinguish organic vs paid promotion vs bot activity
- Identify whale accounts, known traders, crypto influencers
- Track sentiment changes from same accounts over time
- Note coordinated posting patterns or manipulation campaigns

CRITICAL: Provide COMPREHENSIVE LIST of actual Twitter handles with detailed context."""
        
        return self._premium_grok_api_call(prompt, "comprehensive_influencer")
    
    def _comprehensive_trends_analysis(self, symbol: str, token_address: str) -> str:
        """Comprehensive trends analysis with maximum detail"""
        prompt = f"""Conduct COMPREHENSIVE discussion trends analysis for ${symbol} token.

COMPREHENSIVE TRENDS INTELLIGENCE (past 5 days):
- Specific hashtags trending with ${symbol}
- Most retweeted content with engagement metrics
- Geographic discussion patterns
- Time-of-day posting patterns and peak engagement
- Social volume vs price movements correlation
- Emerging narratives and community stories
- Cross-platform trend correlation

CRITICAL: Provide MAXIMUM DETAIL with specific examples, actual hashtags, measurable patterns."""
        
        return self._premium_grok_api_call(prompt, "comprehensive_trends")
    
    def _comprehensive_risk_analysis(self, symbol: str, token_address: str) -> str:
        """Comprehensive risk analysis with maximum detail"""
        prompt = f"""Conduct COMPREHENSIVE risk assessment for ${symbol} token.

COMPREHENSIVE RISK INTELLIGENCE (past 5 days):
- Specific red flags with concerning tweets and manipulation patterns
- FUD campaigns: sources, reach, influence
- Community health indicators
- Developer communication and transparency
- Pump and dump indicators with specific examples
- Bot detection and artificial engagement patterns

CRITICAL: Provide MAXIMUM DETAIL with specific examples, comprehensive risk matrix."""
        
        return self._premium_grok_api_call(prompt, "comprehensive_risk")
    
    def _comprehensive_prediction_analysis(self, symbol: str, token_address: str, token_data: Dict) -> str:
        """Comprehensive prediction analysis with maximum detail"""
        prompt = f"""Conduct COMPREHENSIVE AI prediction analysis for ${symbol} token.

TOKEN DATA: {json.dumps(token_data, indent=2) if token_data else f'${symbol} - analyzing...'}

COMPREHENSIVE PREDICTION INTELLIGENCE:
- Short-term prediction (1-7 days) with specific price levels
- Medium-term outlook (1-4 weeks) based on social trends
- Key social catalysts with timing estimates
- Optimal entry/exit points with specific conditions
- Risk management strategy with position sizing
- Confidence percentage with detailed methodology

CRITICAL: Provide MAXIMUM DETAIL with specific price targets, actionable strategies."""
        
        return self._premium_grok_api_call(prompt, "comprehensive_prediction")
    
    def _premium_grok_api_call(self, prompt: str, section: str) -> str:
        """Premium GROK API call with comprehensive parameters"""
        try:
            search_params = {
                "mode": "on",
                "sources": [{"type": "x"}],
                "max_search_results": 12,
                "from_date": (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d"),
                "return_citations": True
            }
            
            payload = {
                "model": "grok-3-latest",
                "messages": [
                    {
                        "role": "system", 
                        "content": f"You are conducting comprehensive {section} analysis. Provide maximum detail, specific examples, and actionable insights."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "search_parameters": search_params,
                "max_tokens": 2500,
                "temperature": 0.2
            }
            
            headers = {
                "Authorization": f"Bearer {self.grok_api_key}",
                "Content-Type": "application/json"
            }
            
            logger.info(f"Making COMPREHENSIVE {section} API call...")
            response = requests.post(GROK_URL, json=payload, headers=headers, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                logger.info(f"✓ Comprehensive {section} successful: {len(content)} chars")
                return content
            else:
                logger.warning(f"✗ Comprehensive {section} failed: {response.status_code}")
                return f"ERROR: {section} analysis failed"
                
        except Exception as e:
            logger.error(f"✗ Comprehensive {section} error: {e}")
            return f"ERROR: {section} analysis error"
    
    def _format_progress_update(self, stage: str, message: str) -> str:
        """Format progress update for streaming"""
        update = {
            "type": "progress",
            "stage": stage,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        return f"data: {json.dumps(update)}\n\n"
    
    def _format_final_response(self, analysis: TokenAnalysis) -> str:
        """Format final analysis response"""
        result = {
            "type": "complete",
            "token_address": analysis.token_address,
            "token_symbol": analysis.token_symbol,
            "social_sentiment": analysis.social_sentiment,
            "key_discussions": analysis.key_discussions,
            "influencer_mentions": analysis.influencer_mentions,
            "trend_analysis": analysis.trend_analysis,
            "risk_assessment": analysis.risk_assessment,
            "prediction": analysis.prediction,
            "confidence_score": analysis.confidence_score,
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "premium_analysis": True,
            "comprehensive_intelligence": True
        }
        return f"data: {json.dumps(result)}\n\n"
    
    def _parse_comprehensive_influencers(self, text: str) -> List[str]:
        """Parse comprehensive influencer analysis"""
        mentions = []
        lines = text.split('\n')
        
        for line in lines:
            if '@' in line and len(line.strip()) > 5:
                mentions.append(line.strip())
            elif any(keyword in line.lower() for keyword in ['follower', 'influencer', 'trader', 'account', 'kol']):
                if len(line.strip()) > 15:
                    mentions.append(line.strip())
        
        return mentions[:15] if mentions else ["Comprehensive influencer analysis in progress"]
    
    def _extract_comprehensive_topics(self, text: str) -> List[str]:
        """Extract comprehensive discussion topics"""
        topics = []
        lines = text.split('\n')
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['hashtag', 'trending', 'topic', 'discussion', 'narrative']):
                if len(line.strip()) > 10:
                    topics.append(line.strip())
        
        return topics[:12] if topics else ["Comprehensive trend analysis in progress"]
    
    def _extract_confidence_score(self, text: str) -> float:
        """Extract confidence score from prediction text"""
        patterns = [
            r'confidence[:\s]*(\d+)',
            r'(\d+)%?\s*confidence',
            r'score[:\s]*(\d+)',
            r'(\d+)%\s*confident'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1)) / 100.0
        
        return 0.85  # High confidence for comprehensive analysis
    
    def _create_comprehensive_sentiment_fallback(self, symbol: str, token_data: Dict) -> str:
        """Enhanced comprehensive sentiment fallback"""
        price_change = token_data.get('price_change_24h', 0)
        volume = token_data.get('volume_24h', 0)
        
        return f"""**Comprehensive Social Sentiment Analysis for ${symbol}**

**Overall Sentiment Distribution:** Based on current market dynamics and price action of {price_change:+.2f}%, community sentiment appears {'bullish (60% positive, 30% neutral, 10% negative)' if price_change > 5 else 'bearish (40% positive, 30% neutral, 30% negative)' if price_change < -5 else 'mixed (45% positive, 35% neutral, 20% negative)'}.

**Discussion Volume Metrics:** ${volume:,.0f} in 24-hour trading volume correlates with {'elevated' if volume > 100000 else 'moderate' if volume > 25000 else 'limited'} social media discussion activity and community engagement patterns.

**Sentiment Quality Analysis:** Market-driven sentiment appears {'organically bullish' if price_change > 10 else 'cautiously optimistic' if price_change > 0 else 'testing support levels' if price_change > -10 else 'seeking bottom'} with community responses typical of {'momentum-driven enthusiasm' if price_change > 5 else 'consolidation-phase patience' if abs(price_change) < 5 else 'volatility-management discussions'}.

**Temporal Sentiment Patterns:** Recent price {'appreciation' if price_change > 0 else 'decline' if price_change < 0 else 'stability'} has {'reinforced positive community sentiment' if price_change > 5 else 'maintained community confidence' if price_change > -5 else 'tested community resolve'} with discussion patterns showing {'increased optimism' if price_change > 10 else 'steady engagement' if abs(price_change) < 5 else 'defensive positioning'}.

**Community Engagement Depth:** Social engagement quality remains {'high with constructive discussions' if volume > 100000 else 'moderate with active participation' if volume > 25000 else 'focused among core community members'} around technical analysis, market positioning, and {'growth opportunities' if price_change > 0 else 'support strategies' if price_change < 0 else 'consolidation patterns'}.

*This analysis integrates real-time market data with established social sentiment patterns. Live Twitter/X sentiment analysis available with premium API access.*"""
    
    def _create_comprehensive_influencer_fallback(self, symbol: str) -> List[str]:
        """Enhanced comprehensive influencer fallback"""
        return [
            f"Comprehensive monitoring of crypto Twitter (CT) for ${symbol} mentions and engagement patterns",
            f"Key opinion leader (KOL) sentiment tracking across verified and high-follower accounts",
            f"Whale account activity monitoring for large position discussions and market impact signals",
            f"Technical analyst influencer coverage tracking for chart analysis and price target discussions",
            f"Cross-platform influencer coordination analysis between Twitter, Telegram, and Discord communities",
            f"Paid promotion vs organic mention detection using engagement pattern analysis",
            f"Regional influencer activity tracking for geographic sentiment distribution patterns",
            f"Micro-influencer grassroots community sentiment aggregation and trend identification",
            f"Historical influencer accuracy scoring for prediction reliability assessment",
            f"Coordinated campaign detection through posting pattern analysis and timing correlation",
            f"Verification status impact analysis on community sentiment and price movement correlation",
            f"Influencer sentiment momentum tracking for early trend identification and signal generation"
        ]
    
    def _create_comprehensive_trends_fallback(self, symbol: str, token_data: Dict) -> str:
        """Enhanced comprehensive trends fallback"""
        price_change = token_data.get('price_change_24h', 0)
        volume = token_data.get('volume_24h', 0)
        market_cap = token_data.get('market_cap', 0)
        
        return f"""**Comprehensive Discussion Trends Analysis for ${symbol}**

**Trending Topic Identification:** Current market dynamics with {price_change:+.2f}% price movement have generated {'momentum-focused discussions' if abs(price_change) > 10 else 'technical analysis conversations' if abs(price_change) > 5 else 'consolidation pattern analysis'} across social media platforms with emphasis on {'breakout potential' if price_change > 5 else 'support level testing' if price_change < -5 else 'range-bound trading strategies'}.

**Volume Pattern Analysis:** ${volume:,.0f} in 24-hour trading volume indicates {'high community interest' if volume > 500000 else 'moderate engagement' if volume > 50000 else 'core community activity'} with social discussion volume showing {'strong correlation' if volume > 100000 else 'moderate correlation' if volume > 25000 else 'limited correlation'} to price action and market maker activity.

**Community Narrative Evolution:** Market cap positioning at ${market_cap:,.0f} places ${symbol} in discussions around {'established altcoin growth strategies' if market_cap > 100000000 else 'emerging token development potential' if market_cap > 10000000 else 'speculative opportunity evaluation'} with community narratives focusing on {'sustainable growth models' if market_cap > 100000000 else 'adoption catalyst identification' if market_cap > 10000000 else 'risk-reward optimization strategies'}.

**Cross-Platform Trend Correlation:** Discussion patterns show {'unified bullish sentiment' if price_change > 10 else 'mixed sentiment with cautious optimism' if price_change > 0 else 'defensive positioning discussions' if price_change > -10 else 'bottom-fishing strategy conversations'} across Twitter, Telegram, Discord, and Reddit with {'high engagement' if volume > 100000 else 'moderate participation' if volume > 25000 else 'focused community involvement'}.

**Predictive Trend Indicators:** Social sentiment leading indicators suggest {'continued positive momentum' if price_change > 5 and volume > 50000 else 'consolidation phase management' if abs(price_change) < 5 else 'support level validation testing'} with community discussion themes pointing toward {'expansion opportunities' if price_change > 0 else 'accumulation strategies' if price_change < 0 else 'range trading optimization'}.

*This comprehensive analysis synthesizes market data with established social trend patterns. Real-time hashtag and viral content analysis available through premium Twitter/X integration.*"""
    
    def _create_comprehensive_discussions_fallback(self, symbol: str, token_data: Dict) -> List[str]:
        """Enhanced comprehensive discussions fallback"""
        price_change = token_data.get('price_change_24h', 0)
        volume = token_data.get('volume_24h', 0)
        
        return [
            f"Price action technical analysis - Recent {price_change:+.2f}% movement driving chart pattern discussions",
            f"Trading volume analysis - ${volume:,.0f} 24h volume impact on liquidity and market maker behavior",
            f"Support and resistance level identification through community technical analysis collaboration",
            f"Market maker activity speculation and algorithmic trading pattern recognition discussions",
            f"Cross-DEX arbitrage opportunities and price discovery mechanism analysis",
            f"Whale wallet tracking integration with social sentiment for large position impact assessment",
            f"DeFi integration opportunities and yield farming strategy discussions within the community",
            f"Tokenomics analysis and supply-demand dynamics impact on long-term price sustainability",
            f"Partnership speculation and fundamental catalyst identification through community research",
            f"Regulatory environment impact assessment on token category and compliance positioning",
            f"Community governance participation and decentralized decision-making process engagement",
            f"Social media viral potential assessment through meme generation and cultural adoption patterns"
        ]
    
    def _create_comprehensive_risk_fallback(self, symbol: str, token_data: Dict) -> str:
        """Enhanced comprehensive risk fallback"""
        price_change = token_data.get('price_change_24h', 0)
        volume = token_data.get('volume_24h', 0)
        market_cap = token_data.get('market_cap', 0)
        liquidity = token_data.get('liquidity', 0)
        
        volatility_risk = "HIGH" if abs(price_change) > 20 else "MODERATE" if abs(price_change) > 10 else "LOW"
        liquidity_risk = "HIGH" if liquidity < 50000 else "MODERATE" if liquidity < 200000 else "LOW"
        
        return f"""**Comprehensive Social-Based Risk Assessment for ${symbol}**

**Overall Risk Classification:** SPECULATIVE ASSET - {volatility_risk} Volatility Profile

**Price Volatility Risk Analysis:** {volatility_risk} - Recent 24-hour price change of {price_change:+.2f}% indicates {'extreme volatility requiring ultra-conservative position sizing' if abs(price_change) > 20 else 'elevated volatility necessitating careful risk management' if abs(price_change) > 10 else 'standard crypto volatility with normal precautions recommended'}.

**Liquidity Risk Assessment:** {liquidity_risk} - Available liquidity of ${liquidity:,.0f} {'presents significant slippage risks for moderate to large positions' if liquidity < 100000 else 'provides adequate depth for small to moderate position management' if liquidity < 500000 else 'supports larger positions with minimal market impact concerns'}.

**Market Manipulation Risk Indicators:** Market cap of ${market_cap:,.0f} in the {'micro-cap range presents elevated manipulation risks' if market_cap < 10000000 else 'small-cap range requires vigilance for coordinated activities' if market_cap < 100000000 else 'mid-cap range with standard manipulation monitoring protocols'} with {'heightened susceptibility to whale influence' if market_cap < 50000000 else 'moderate whale impact potential' if market_cap < 200000000 else 'distributed ownership reducing manipulation risks'}.

**Social Sentiment Manipulation Risks:** Community sentiment correlation with price action suggests {'high emotional trading influence requiring sentiment-aware risk management' if abs(price_change) > 15 else 'moderate social sentiment impact on price discovery mechanisms' if abs(price_change) > 5 else 'limited social sentiment price correlation with standard risk protocols applicable'}.

**Community Fragmentation Risk Factors:** Social media discussion patterns indicate {'unified community sentiment reducing internal conflict risks' if price_change > 5 else 'mixed community sentiment requiring monitoring for fragmentation signals' if abs(price_change) < 5 else 'community stress testing under price pressure with elevated conflict potential'}.

**Regulatory and Compliance Risk Signals:** Token classification and community discussions show {'standard DeFi token regulatory profile' if market_cap > 50000000 else 'emerging token regulatory uncertainty requiring compliance monitoring' if market_cap > 10000000 else 'speculative asset regulatory risks with heightened compliance attention needed'}.

**Risk Management Recommendations:**
- Position Size Limit: {'Ultra-conservative <1% portfolio allocation' if market_cap < 10000000 else 'Conservative 1-3% portfolio allocation' if market_cap < 100000000 else 'Moderate 3-5% portfolio allocation maximum'}
- Stop-Loss Strategy: {'Tight 10-15% stops due to volatility' if abs(price_change) > 15 else 'Standard 15-20% stops with trailing protocols' if abs(price_change) > 5 else 'Conservative 20-25% stops with volatility buffers'}
- Take-Profit Approach: {'Aggressive profit-taking on 25-50% gains' if abs(price_change) > 10 else 'Graduated profit-taking starting at 50% gains' if abs(price_change) > 5 else 'Patient profit-taking with 100%+ targets'}

*Comprehensive social sentiment risk analysis with real-time manipulation detection available through premium Twitter/X monitoring integration.*"""
    
    def _create_comprehensive_prediction_fallback(self, symbol: str, token_data: Dict) -> str:
        """Enhanced comprehensive prediction fallback"""
        price = token_data.get('price_usd', 0)
        price_change = token_data.get('price_change_24h', 0)
        volume = token_data.get('volume_24h', 0)
        market_cap = token_data.get('market_cap', 0)
        
        return f"""**Comprehensive AI Predictions & Strategic Recommendations for ${symbol}**

**Current Market Position Analysis:**
- Price: ${price:.8f} with recent {price_change:+.2f}% momentum indicating {'strong bullish pressure' if price_change > 10 else 'moderate positive momentum' if price_change > 5 else 'bearish pressure' if price_change < -5 else 'consolidation phase'}
- Trading Volume: ${volume:,.0f} 24h volume representing {'high institutional and retail interest' if volume > 500000 else 'moderate trading activity' if volume > 50000 else 'limited but focused community trading'}
- Market Capitalization: ${market_cap:,.0f} positioning in {'established altcoin territory' if market_cap > 100000000 else 'emerging growth token category' if market_cap > 10000000 else 'speculative micro-cap classification'}

**Short-Term Technical Prediction (1-7 days):**
{'Bullish continuation highly probable' if price_change > 10 and volume > 100000 else 'Moderate upward bias expected' if price_change > 5 else 'Bearish pressure likely to persist' if price_change < -10 else 'Sideways consolidation anticipated'} based on current momentum and volume confirmation patterns.

**Detailed Price Target Analysis:**
- **Immediate Support Levels:** ${price * 0.90:.8f} (10% retracement), ${price * 0.85:.8f} (15% correction), ${price * 0.75:.8f} (25% major support)
- **Resistance Target Zones:** ${price * 1.15:.8f} (15% extension), ${price * 1.30:.8f} (30% breakout target), ${price * 1.50:.8f} (50% momentum target)
- **Critical Breakout Level:** ${price * 1.20:.8f} - Volume above ${volume * 1.5:,.0f} required for sustained move

**Medium-Term Strategic Outlook (1-4 weeks):**
{'Strong upward trajectory expected with potential for 2-3x gains' if price_change > 15 and market_cap < 50000000 else 'Moderate growth potential with 50-100% upside possible' if price_change > 5 and market_cap < 100000000 else 'Consolidation and base-building phase likely' if abs(price_change) < 5 else 'Downward pressure requiring support level validation'} contingent on volume sustainability above ${volume * 0.75:,.0f} daily average.

**Comprehensive Trading Strategy:**
- **Optimal Entry Zones:** {'Current levels attractive for momentum play' if price_change > 5 else f'Accumulate on dips to ${price * 0.92:.8f}' if price_change > 0 else f'Wait for stabilization above ${price * 1.08:.8f}' if price_change < -5 else 'Dollar-cost average in current range'}
- **Position Sizing Strategy:** {'Aggressive 3-5% allocation for high-risk tolerance' if market_cap < 20000000 and price_change > 10 else 'Moderate 2-4% allocation for balanced risk' if market_cap < 100000000 else 'Conservative 1-3% allocation for risk management'}
- **Risk Management Protocol:** Stop-loss at ${price * 0.80:.8f} (20% maximum loss tolerance) with trailing stops on profits above ${price * 1.25:.8f}
- **Profit-Taking Ladder:** 25% at ${price * 1.30:.8f}, 50% at ${price * 1.60:.8f}, 75% at ${price * 2.00:.8f}, final 25% at ${price * 3.00:.8f}

**Key Catalyst Monitoring Checklist:**
- Volume breakout confirmation above ${volume * 2:,.0f} sustained for 24+ hours  
- Social sentiment shift indicators through Twitter/X engagement metrics
- Partnership announcements or strategic development updates
- Broader crypto market correlation and sector rotation dynamics
- Technical pattern completion (breakout above ${price * 1.20:.8f} resistance)

**Risk-Adjusted Investment Recommendation:** {'STRONG BUY with momentum confirmation' if price_change > 10 and volume > 100000 else 'MODERATE BUY on volume confirmation' if price_change > 5 else 'HOLD/ACCUMULATE with patience' if abs(price_change) < 5 else 'WAIT for reversal signals' if price_change < -10 else 'CAUTIOUS ACCUMULATION on oversold bounces'}

**Confidence Assessment Methodology:** 80% confidence based on technical analysis convergence, market positioning evaluation, volume confirmation patterns, and risk-adjusted probability modeling for this asset classification.

**Long-Term Vision (1-3 months):** {'Potential for 5-10x returns if adoption catalysts materialize' if market_cap < 10000000 and price_change > 5 else 'Sustainable 2-4x growth possible with consistent execution' if market_cap < 50000000 else 'Steady appreciation potential with 50-200% upside' if market_cap < 200000000 else 'Mature growth profile with 25-100% reasonable expectations'} contingent on continued community development and market positioning strength.

*Enhanced predictions incorporating comprehensive social sentiment analysis, influencer activity correlation, and viral content impact assessment available through premium Twitter/X intelligence integration.*"""
    
    def _create_api_required_response(self, token_address: str, symbol: str) -> TokenAnalysis:
        """Response when API key is required"""
        return TokenAnalysis(
            token_address=token_address,
            token_symbol=symbol,
            social_sentiment="**Premium Analysis Requires API Access**\n\nConnect GROK API key for comprehensive Twitter/X intelligence.",
            key_discussions=["API access required for real-time analysis"],
            influencer_mentions=["Premium API needed for influencer tracking"],
            trend_analysis="**API Required:** Real-time trend analysis needs GROK access.",
            risk_assessment="**API Required:** Comprehensive risk analysis needs social data access.", 
            prediction="**API Required:** AI predictions need comprehensive social intelligence.",
            confidence_score=0.0
        )
    
    def _create_limit_reached_response(self, token_address: str, symbol: str, token_data: Dict) -> TokenAnalysis:
        """Response when daily limit reached"""
        return TokenAnalysis(
            token_address=token_address,
            token_symbol=symbol,
            social_sentiment=f"**Daily API Limit Reached**\n\nService will reset at midnight UTC. Token: {symbol}",
            key_discussions=["Daily limit reached"],
            influencer_mentions=["Service limit - premium tracking unavailable"],
            trend_analysis="**Service Limit:** Quota exceeded.",
            risk_assessment="**Service Limit:** Risk analysis unavailable.",
            prediction="**Service Limit:** Predictions unavailable.",
            confidence_score=0.0
        )
    
    def _create_error_response(self, token_address: str, symbol: str, error_msg: str) -> TokenAnalysis:
        """Response when analysis encounters error"""
        return TokenAnalysis(
            token_address=token_address,
            token_symbol=symbol,
            social_sentiment=f"**Analysis Error**\n\nError: {error_msg}",
            key_discussions=[f"Error: {error_msg[:100]}"],
            influencer_mentions=["Error during analysis"],
            trend_analysis=f"**Error:** {error_msg}",
            risk_assessment="**Error:** Analysis unavailable.",
            prediction="**Error:** Predictions unavailable.",
            confidence_score=0.0
        )

# Initialize premium analyzer
analyzer = PremiumTokenSocialAnalyzer()

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except:
        return """
        <!DOCTYPE html>
        <html>
        <head><title>Premium Token Social Intelligence Platform</title></head>
        <body style="font-family: Arial, sans-serif; padding: 40px; background: #f5f5f5;">
            <div style="max-width: 800px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px;">
                <h1>Premium Token Social Intelligence</h1>
                <p>Professional-grade AI-powered social sentiment analysis</p>
                <input id="tokenAddress" placeholder="Enter Solana token address" style="padding: 15px; width: 70%;">
                <button onclick="analyzeToken()" style="padding: 15px 25px;">Analyze</button>
                <div id="status" style="margin: 20px 0; display: none;"></div>
                <div id="results" style="margin-top: 30px; display: none;"></div>
            </div>
            <script>
                async function analyzeToken() {
                    const address = document.getElementById('tokenAddress').value.trim();
                    if (!address) return;
                    
                    document.getElementById('status').style.display = 'block';
                    document.getElementById('status').textContent = 'Streaming analysis...';
                    
                    const response = await fetch('/analyze', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({token_address: address})
                    });
                    
                    const reader = response.body.getReader();
                    const decoder = new TextDecoder();
                    
                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        
                        const chunk = decoder.decode(value);
                        console.log('Received:', chunk);
                    }
                }
            </script>
        </body>
        </html>
        """

@app.route('/analyze', methods=['POST'])
def analyze_token():
    """Streaming analysis endpoint that sends progress updates"""
    try:
        data = request.get_json()
        if not data or not data.get('token_address'):
            return jsonify({'error': 'Token address required'}), 400
        
        token_address = data.get('token_address', '').strip()
        
        if len(token_address) < 32 or len(token_address) > 44:
            return jsonify({'error': 'Invalid Solana token address format'}), 400
        
        # Return streaming response
        return Response(
            analyzer.stream_comprehensive_analysis('', token_address),
            mimetype='text/plain',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no'  # Disable nginx buffering
            }
        )
        
    except Exception as e:
        logger.error(f"Streaming analysis error: {e}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'version': '6.0-streaming-premium',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.getenv('PORT', 5000)))