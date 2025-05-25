from flask import Flask, render_template, request, jsonify
import requests
import os
from datetime import datetime, timedelta
import json
import re
from dataclasses import dataclass
from typing import List, Dict, Optional
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
GROK_API_KEY = os.getenv('GROK_API_KEY', 'your-grok-api-key-here')
GROK_URL = "https://api.x.ai/v1/chat/completions"

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

class TokenSocialAnalyzer:
    def __init__(self):
        self.grok_api_key = GROK_API_KEY
        logger.info(f"Initialized with GROK API key: {'SET' if self.grok_api_key and self.grok_api_key != 'your-grok-api-key-here' else 'NOT SET'}")
    
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
    
    def analyze_token_social_sentiment(self, token_symbol: str, token_address: str) -> TokenAnalysis:
        """Optimized analysis with single API call instead of 5 separate calls"""
        try:
            # Get basic token data first
            token_data = self.fetch_dexscreener_data(token_address)
            symbol = token_data.get('symbol', token_symbol or 'UNKNOWN')
            
            logger.info(f"Starting optimized social analysis for {symbol} ({token_address})")
            
            # Check if GROK API key is available
            if not self.grok_api_key or self.grok_api_key == 'your-grok-api-key-here':
                logger.warning("GROK API key not set, using mock analysis")
                return self._create_mock_analysis(token_address, symbol, token_data)
            
            # OPTIMIZED: Single comprehensive analysis instead of 5 separate calls
            logger.info("Performing comprehensive social analysis...")
            comprehensive_analysis = self._comprehensive_social_analysis(symbol, token_address, token_data)
            
            logger.info(f"Analysis completed successfully for {symbol}")
            
            return comprehensive_analysis
            
        except Exception as e:
            logger.error(f"Error in analyze_token_social_sentiment: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Return mock analysis on error
            return self._create_mock_analysis(token_address, token_symbol or 'UNKNOWN', {})
    
    def _comprehensive_social_analysis(self, symbol: str, token_address: str, token_data: Dict) -> TokenAnalysis:
        """Single comprehensive GROK API call instead of multiple calls"""
        
        comprehensive_prompt = f"""
        Perform a comprehensive social media analysis for ${symbol} token (contract: {token_address}).
        
        TOKEN DATA: {json.dumps(token_data, indent=2)}
        
        Analyze the past 3 days of Twitter/X discussions and provide:
        
        1. SOCIAL SENTIMENT ANALYSIS:
           - Overall sentiment (bullish/bearish/neutral) with percentage breakdown
           - Discussion volume trends (high/medium/low activity)
           - Common themes and emotional tone
           - Community engagement levels
        
        2. INFLUENCER & KEY ACCOUNT ACTIVITY:
           - Notable accounts discussing this token (KOLs, crypto influencers)
           - What they're saying (positive/negative/neutral opinions)
           - Any endorsements, warnings, or coordinated campaigns
           - Impact potential of these mentions
        
        3. DISCUSSION TRENDS & TOPICS:
           - Trending topics (partnerships, listings, developments, news)
           - Discussion volume patterns (increasing/decreasing/stable)
           - Recurring themes and narratives
           - Community sentiment shifts
        
        4. RISK ASSESSMENT:
           - Red flags: dump campaigns, FUD, technical concerns
           - Developer/team related discussions
           - Regulatory or compliance mentions
           - Community fragmentation or disputes
           - Overall risk level (LOW/MODERATE/HIGH)
        
        5. PRICE PREDICTION & RECOMMENDATIONS:
           - Short-term prediction (1-7 days) with reasoning
           - Medium-term outlook (1-4 weeks)
           - Key catalysts to watch for
           - Recommended action (BUY/SELL/HOLD/AVOID)
           - Confidence score (0-100) for this analysis
        
        Format your response with clear sections using **bold headers** for each analysis area.
        Be specific, actionable, and include concrete examples where possible.
        """
        
        try:
            logger.info("Making single comprehensive GROK API call...")
            result = self._grok_live_search_query(comprehensive_prompt)
            
            # Parse the comprehensive result into components
            return self._parse_comprehensive_analysis(result, token_address, symbol)
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            return self._create_mock_analysis(token_address, symbol, token_data)
    
    def _parse_comprehensive_analysis(self, analysis_text: str, token_address: str, symbol: str) -> TokenAnalysis:
        """Parse the comprehensive analysis into structured components"""
        
        try:
            # Split analysis into sections
            sections = self._split_analysis_sections(analysis_text)
            
            # Extract key information
            key_discussions = self._extract_key_topics(analysis_text)
            influencer_mentions = self._extract_key_mentions(analysis_text)
            confidence_score = self._extract_confidence_score(analysis_text)
            
            return TokenAnalysis(
                token_address=token_address,
                token_symbol=symbol,
                social_sentiment=sections.get('sentiment', analysis_text[:800] + "..."),
                key_discussions=key_discussions,
                influencer_mentions=influencer_mentions,
                trend_analysis=sections.get('trends', analysis_text[800:1600] + "..."),
                risk_assessment=sections.get('risks', analysis_text[1600:2400] + "..."),
                prediction=sections.get('prediction', analysis_text[-800:]),
                confidence_score=confidence_score
            )
            
        except Exception as e:
            logger.error(f"Error parsing comprehensive analysis: {e}")
            # Fallback: use the full analysis text split into sections
            text_length = len(analysis_text)
            section_size = text_length // 5
            
            return TokenAnalysis(
                token_address=token_address,
                token_symbol=symbol,
                social_sentiment=analysis_text[:section_size],
                key_discussions=self._extract_key_topics(analysis_text),
                influencer_mentions=self._extract_key_mentions(analysis_text),
                trend_analysis=analysis_text[section_size:section_size*2],
                risk_assessment=analysis_text[section_size*2:section_size*3],
                prediction=analysis_text[section_size*3:],
                confidence_score=self._extract_confidence_score(analysis_text)
            )
    
    def _split_analysis_sections(self, text: str) -> Dict[str, str]:
        """Split comprehensive analysis into sections"""
        sections = {}
        
        # Look for section headers
        sentiment_match = re.search(r'\*\*.*?SENTIMENT.*?\*\*(.*?)(?=\*\*.*?(?:INFLUENCER|TREND|RISK|PREDICTION)|$)', text, re.DOTALL | re.IGNORECASE)
        influencer_match = re.search(r'\*\*.*?INFLUENCER.*?\*\*(.*?)(?=\*\*.*?(?:TREND|RISK|PREDICTION)|$)', text, re.DOTALL | re.IGNORECASE)
        trends_match = re.search(r'\*\*.*?(?:TREND|DISCUSSION).*?\*\*(.*?)(?=\*\*.*?(?:RISK|PREDICTION)|$)', text, re.DOTALL | re.IGNORECASE)
        risk_match = re.search(r'\*\*.*?RISK.*?\*\*(.*?)(?=\*\*.*?PREDICTION|$)', text, re.DOTALL | re.IGNORECASE)
        prediction_match = re.search(r'\*\*.*?(?:PREDICTION|RECOMMENDATION).*?\*\*(.*?)$', text, re.DOTALL | re.IGNORECASE)
        
        if sentiment_match:
            sections['sentiment'] = sentiment_match.group(1).strip()
        if influencer_match:
            sections['influencer'] = influencer_match.group(1).strip()
        if trends_match:
            sections['trends'] = trends_match.group(1).strip()
        if risk_match:
            sections['risks'] = risk_match.group(1).strip()
        if prediction_match:
            sections['prediction'] = prediction_match.group(1).strip()
            
        return sections
    
    def _create_mock_analysis(self, token_address: str, symbol: str, token_data: Dict) -> TokenAnalysis:
        """Create a mock analysis for testing/fallback"""
        logger.info(f"Creating mock analysis for {symbol}")
        
        return TokenAnalysis(
            token_address=token_address,
            token_symbol=symbol,
            social_sentiment=f"**Mock Analysis for ${symbol}**\n\nThis is a demonstration of the social sentiment analysis platform. In production, this would show:\n\n• Real-time Twitter/X sentiment analysis\n• Community discussion volume and trends\n• Overall market sentiment (bullish/bearish/neutral)\n• Emotional tone of recent discussions\n\n*Note: Connect your GROK API key for live analysis.*",
            key_discussions=[
                "Technical development updates",
                "Community engagement metrics", 
                "Partnership announcements",
                "Trading volume discussions",
                "Price movement analysis"
            ],
            influencer_mentions=[
                "Demo: @CryptoInfluencer mentioned positive outlook",
                "Demo: @BlockchainExpert shared technical analysis",
                "Demo: @DefiTrader discussed volume trends"
            ],
            trend_analysis=f"**Trend Analysis for ${symbol}**\n\n• Discussion volume: Moderate activity\n• Trending topics: Technical updates, community growth\n• Engagement patterns: Steady community interaction\n• Narrative themes: Innovation and development focus\n\n*This is mock data - live analysis requires GROK API access*",
            risk_assessment=f"**Risk Assessment for ${symbol}**\n\n**Low Risk Indicators:**\n• Consistent community engagement\n• No coordinated negative campaigns detected\n• Technical discussions remain positive\n\n**Moderate Risk Factors:**\n• Market volatility affects sentiment\n• Limited mainstream coverage\n\n**Overall Risk Level: LOW-MODERATE**\n\n*Live risk assessment available with GROK API*",
            prediction=f"**AI Prediction for ${symbol}**\n\n**Short-term (1-7 days):**\n• Expected consolidation around current levels\n• Community sentiment remains stable\n• Technical indicators suggest neutral momentum\n\n**Medium-term (1-4 weeks):**\n• Potential for positive movement with developments\n• Social engagement trends upward\n• Market conditions favor community-driven tokens\n\n**Key Catalysts to Watch:**\n• Partnership announcements\n• Technical milestones\n• Community growth metrics\n\n**Recommended Action:** HOLD/ACCUMULATE\n\n*This is a demonstration - live predictions require GROK API*",
            confidence_score=0.65
        )
    
    def _grok_live_search_query(self, prompt: str, search_params: Dict = None) -> str:
        """Optimized GROK API call with better parameters"""
        try:
            if not self.grok_api_key or self.grok_api_key == 'your-grok-api-key-here':
                return "GROK API key not configured - using mock response"
            
            # Optimized search parameters
            default_search_params = {
                "mode": "on",
                "sources": [{"type": "x"}],
                "max_search_results": 8,  # Reduced for faster responses
                "from_date": (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d"),  # Reduced from 7 days
                "return_citations": False  # Disable citations for faster response
            }
            
            if search_params:
                default_search_params.update(search_params)
            
            payload = {
                "model": "grok-3-latest", 
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert crypto analyst. Provide concise, actionable analysis based on social media discussions. Focus on key insights that affect price movements. Use clear section headers with **bold text**."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "search_parameters": default_search_params,
                "max_tokens": 1200,  # Increased for comprehensive response
                "temperature": 0.6   # Slightly more focused
            }
            
            headers = {
                "Authorization": f"Bearer {self.grok_api_key}",
                "Content-Type": "application/json"
            }
            
            logger.info(f"Making optimized GROK API call with {len(prompt)} char prompt...")
            response = requests.post(GROK_URL, json=payload, headers=headers, timeout=120)  # 2-minute timeout
            
            logger.info(f"GROK API response status: {response.status_code}")
            
            if response.status_code == 401:
                logger.error("GROK API: Unauthorized - check API key")
                return "Error: Invalid GROK API key"
            elif response.status_code == 429:
                logger.error("GROK API: Rate limit exceeded")
                return "Error: GROK API rate limit exceeded - please try again later"
            
            response.raise_for_status()
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            logger.info(f"GROK API call successful, response length: {len(content)}")
            return content
            
        except requests.exceptions.Timeout:
            logger.error("GROK API call timed out")
            return "Analysis timed out - the social media data is being processed. Please try again."
        except requests.exceptions.RequestException as e:
            logger.error(f"GROK API request error: {e}")
            return f"API request failed: {str(e)}"
        except Exception as e:
            logger.error(f"GROK API Error: {e}")
            return f"Analysis error: {str(e)}"
    
    def _extract_key_mentions(self, text: str) -> List[str]:
        """Extract key account mentions from analysis"""
        mentions = []
        lines = text.split('\n')
        for line in lines:
            if '@' in line and ('influencer' in line.lower() or 'kol' in line.lower() or 'account' in line.lower() or 'mentioned' in line.lower()):
                mentions.append(line.strip())
        return mentions[:5]  # Top 5
    
    def _extract_key_topics(self, text: str) -> List[str]:
        """Extract key discussion topics"""
        topics = []
        lines = text.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['partnership', 'listing', 'development', 'news', 'update', 'trending', 'discussion']):
                topics.append(line.strip())
        return topics[:5]  # Top 5
    
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
        
        return 0.7  # Default good confidence for live analysis

# Initialize analyzer
analyzer = TokenSocialAnalyzer()

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except:
        return """
        <!DOCTYPE html>
        <html>
        <head><title>Token Social Intelligence Platform</title></head>
        <body>
            <h1>Token Social Intelligence Platform</h1>
            <p>Templates not found - using fallback interface.</p>
            <div>
                <input id="tokenAddress" placeholder="Enter Solana token address" style="padding: 10px; width: 400px;">
                <button onclick="analyzeToken()" style="padding: 10px 20px;">Analyze</button>
            </div>
            <div id="results"></div>
            <script>
                async function analyzeToken() {
                    const address = document.getElementById('tokenAddress').value;
                    if (!address) return;
                    
                    document.getElementById('results').innerHTML = '<p>Analyzing...</p>';
                    
                    try {
                        const response = await fetch('/analyze', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({token_address: address})
                        });
                        
                        const data = await response.json();
                        document.getElementById('results').innerHTML = '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
                    } catch (error) {
                        document.getElementById('results').innerHTML = '<p>Error: ' + error.message + '</p>';
                    }
                }
            </script>
        </body>
        </html>
        """

@app.route('/analyze', methods=['POST'])
def analyze_token():
    try:
        logger.info("Analysis request received")
        data = request.get_json()
        
        if not data:
            logger.error("No JSON data received")
            return jsonify({'error': 'No data provided'}), 400
        
        token_address = data.get('token_address', '').strip()
        
        if not token_address:
            logger.error("No token address provided")
            return jsonify({'error': 'Token address is required'}), 400
        
        # Validate Solana address format (basic check)
        if len(token_address) < 32 or len(token_address) > 44:
            logger.error(f"Invalid token address format: {token_address}")
            return jsonify({'error': 'Invalid Solana token address format'}), 400
        
        logger.info(f"Starting optimized analysis for token: {token_address}")
        
        # Perform analysis
        analysis = analyzer.analyze_token_social_sentiment('', token_address)
        
        # Convert to dict for JSON response
        result = {
            'token_address': analysis.token_address,
            'token_symbol': analysis.token_symbol,
            'social_sentiment': analysis.social_sentiment,
            'key_discussions': analysis.key_discussions,
            'influencer_mentions': analysis.influencer_mentions,
            'trend_analysis': analysis.trend_analysis,
            'risk_assessment': analysis.risk_assessment,
            'prediction': analysis.prediction,
            'confidence_score': analysis.confidence_score,
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }
        
        logger.info(f"Analysis completed successfully for {analysis.token_symbol}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'error': f'Analysis failed: {str(e)}',
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/health')
def health():
    try:
        grok_status = 'configured' if GROK_API_KEY and GROK_API_KEY != 'your-grok-api-key-here' else 'not_configured'
        return jsonify({
            'status': 'healthy', 
            'timestamp': datetime.now().isoformat(),
            'grok_api': grok_status,
            'version': '2.0'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/debug')
def debug():
    """Debug endpoint to check configuration"""
    return jsonify({
        'grok_api_key_set': bool(GROK_API_KEY and GROK_API_KEY != 'your-grok-api-key-here'),
        'grok_api_key_preview': f"{GROK_API_KEY[:10]}..." if GROK_API_KEY else "Not set",
        'environment_vars': list(os.environ.keys()),
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.getenv('PORT', 5000)))