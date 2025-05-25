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
        
        **1. SOCIAL SENTIMENT ANALYSIS:**
        - Overall sentiment (bullish/bearish/neutral) with percentage breakdown
        - Discussion volume trends (high/medium/low activity)
        - Common themes and emotional tone
        - Community engagement levels
        
        **2. INFLUENCER & KEY ACCOUNT ACTIVITY:**
        - Notable accounts discussing this token (KOLs, crypto influencers)
        - What they're saying (positive/negative/neutral opinions)
        - Any endorsements, warnings, or coordinated campaigns
        - Impact potential of these mentions
        
        **3. DISCUSSION TRENDS & TOPICS:**
        - Trending topics (partnerships, listings, developments, news)
        - Discussion volume patterns (increasing/decreasing/stable)
        - Recurring themes and narratives
        - Community sentiment shifts
        
        **4. RISK ASSESSMENT:**
        - Red flags: dump campaigns, FUD, technical concerns
        - Developer/team related discussions
        - Regulatory or compliance mentions
        - Community fragmentation or disputes
        - Overall risk level (LOW/MODERATE/HIGH)
        
        **5. PRICE PREDICTION & RECOMMENDATIONS:**
        - Short-term prediction (1-7 days) with reasoning
        - Medium-term outlook (1-4 weeks)
        - Key catalysts to watch for
        - Recommended action (BUY/SELL/HOLD/AVOID)
        - Confidence score (0-100) for this analysis
        
        Format your response with clear sections using the **bold headers** above.
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
        """Enhanced parsing with better section extraction"""
        
        try:
            logger.info(f"Parsing comprehensive analysis ({len(analysis_text)} chars)")
            
            # Enhanced section extraction
            sections = self._enhanced_split_analysis_sections(analysis_text)
            
            # Extract key information with fallbacks
            key_discussions = self._extract_key_topics(analysis_text)
            influencer_mentions = self._extract_key_mentions(analysis_text)
            confidence_score = self._extract_confidence_score(analysis_text)
            
            # Use extracted sections with intelligent fallbacks
            social_sentiment = sections.get('sentiment') or self._extract_sentiment_fallback(analysis_text)
            trend_analysis = sections.get('trends') or self._extract_trends_fallback(analysis_text)
            risk_assessment = sections.get('risks') or self._extract_risks_fallback(analysis_text)
            prediction = sections.get('prediction') or self._extract_prediction_fallback(analysis_text)
            
            logger.info(f"Parsed sections: sentiment={len(social_sentiment)}, trends={len(trend_analysis)}, risks={len(risk_assessment)}, prediction={len(prediction)}")
            
            return TokenAnalysis(
                token_address=token_address,
                token_symbol=symbol,
                social_sentiment=social_sentiment,
                key_discussions=key_discussions,
                influencer_mentions=influencer_mentions,
                trend_analysis=trend_analysis,
                risk_assessment=risk_assessment,
                prediction=prediction,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            logger.error(f"Error in enhanced parsing: {e}")
            # More intelligent fallback parsing
            return self._intelligent_fallback_parsing(analysis_text, token_address, symbol)
    
    def _enhanced_split_analysis_sections(self, text: str) -> Dict[str, str]:
        """Enhanced section splitting with multiple patterns"""
        sections = {}
        
        # Multiple patterns to catch different formatting styles
        section_patterns = [
            # Pattern 1: **1. SECTION NAME**
            (r'\*\*\s*1\.\s*SOCIAL SENTIMENT.*?\*\*(.*?)(?=\*\*\s*2\.|$)', 'sentiment'),
            (r'\*\*\s*2\.\s*INFLUENCER.*?\*\*(.*?)(?=\*\*\s*3\.|$)', 'influencer'),
            (r'\*\*\s*3\.\s*DISCUSSION.*?\*\*(.*?)(?=\*\*\s*4\.|$)', 'trends'),
            (r'\*\*\s*4\.\s*RISK.*?\*\*(.*?)(?=\*\*\s*5\.|$)', 'risks'),
            (r'\*\*\s*5\.\s*PRICE.*?\*\*(.*?)$', 'prediction'),
            
            # Pattern 2: **SECTION NAME**
            (r'\*\*\s*SOCIAL SENTIMENT.*?\*\*(.*?)(?=\*\*.*?(?:INFLUENCER|DISCUSSION|RISK|PRICE)|$)', 'sentiment'),
            (r'\*\*\s*INFLUENCER.*?\*\*(.*?)(?=\*\*.*?(?:DISCUSSION|RISK|PRICE)|$)', 'influencer'),
            (r'\*\*\s*DISCUSSION.*?\*\*(.*?)(?=\*\*.*?(?:RISK|PRICE)|$)', 'trends'),
            (r'\*\*\s*RISK.*?\*\*(.*?)(?=\*\*.*?PRICE|$)', 'risks'),
            (r'\*\*\s*PRICE.*?\*\*(.*?)$', 'prediction'),
        ]
        
        for pattern, section_key in section_patterns:
            if section_key not in sections:  # Don't overwrite if already found
                match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
                if match:
                    content = match.group(1).strip()
                    if len(content) > 50:  # Only use if substantial content
                        sections[section_key] = content
                        logger.info(f"Found {section_key} section: {len(content)} chars")
        
        return sections
    
    def _extract_sentiment_fallback(self, text: str) -> str:
        """Extract sentiment analysis from text if section parsing fails"""
        # Look for sentiment-related content
        sentiment_keywords = ['sentiment', 'bullish', 'bearish', 'neutral', 'optimistic', 'pessimistic', 'mood', 'feeling']
        
        lines = text.split('\n')
        sentiment_lines = []
        
        for i, line in enumerate(lines):
            if any(keyword in line.lower() for keyword in sentiment_keywords):
                # Include this line and next few lines for context
                sentiment_lines.extend(lines[i:i+3])
        
        if sentiment_lines:
            return '\n'.join(sentiment_lines[:10])  # First 10 relevant lines
        
        # Fallback: first portion of analysis
        return text[:400] + "..." if len(text) > 400 else text
    
    def _extract_trends_fallback(self, text: str) -> str:
        """Extract trend analysis from text"""
        trend_keywords = ['trend', 'discussion', 'volume', 'engagement', 'pattern', 'topic', 'narrative']
        
        lines = text.split('\n')
        trend_lines = []
        
        for i, line in enumerate(lines):
            if any(keyword in line.lower() for keyword in trend_keywords):
                trend_lines.extend(lines[i:i+2])
        
        if trend_lines:
            return '\n'.join(trend_lines[:8])
        
        # Look for middle portion of analysis
        mid_point = len(text) // 3
        return text[mid_point:mid_point+400] + "..." if len(text) > mid_point+400 else text[mid_point:]
    
    def _extract_risks_fallback(self, text: str) -> str:
        """Extract risk assessment from text"""
        risk_keywords = ['risk', 'red flag', 'concern', 'warning', 'danger', 'volatility', 'dump', 'fud']
        
        lines = text.split('\n')
        risk_lines = []
        
        for i, line in enumerate(lines):
            if any(keyword in line.lower() for keyword in risk_keywords):
                risk_lines.extend(lines[i:i+2])
        
        if risk_lines:
            return '\n'.join(risk_lines[:8])
        
        # Look for risk-related content in latter part
        risk_point = len(text) * 2 // 3
        return text[risk_point:risk_point+400] + "..." if len(text) > risk_point+400 else text[risk_point:]
    
    def _extract_prediction_fallback(self, text: str) -> str:
        """Extract prediction from text"""
        pred_keywords = ['prediction', 'forecast', 'expect', 'outlook', 'short-term', 'medium-term', 'recommend', 'buy', 'sell', 'hold']
        
        lines = text.split('\n')
        pred_lines = []
        
        for i, line in enumerate(lines):
            if any(keyword in line.lower() for keyword in pred_keywords):
                pred_lines.extend(lines[i:i+3])
        
        if pred_lines:
            return '\n'.join(pred_lines[:12])
        
        # Use last portion of analysis
        return text[-500:] if len(text) > 500 else text
    
    def _intelligent_fallback_parsing(self, analysis_text: str, token_address: str, symbol: str) -> TokenAnalysis:
        """Intelligent fallback when section parsing completely fails"""
        
        # Split text into roughly equal sections
        text_length = len(analysis_text)
        section_size = text_length // 5
        
        sections = [
            analysis_text[i*section_size:(i+1)*section_size] 
            for i in range(5)
        ]
        
        return TokenAnalysis(
            token_address=token_address,
            token_symbol=symbol,
            social_sentiment=sections[0] if sections[0] else f"Social sentiment analysis for ${symbol} - comprehensive data available",
            key_discussions=self._extract_key_topics(analysis_text),
            influencer_mentions=self._extract_key_mentions(analysis_text),
            trend_analysis=sections[1] if len(sections) > 1 and sections[1] else "Discussion trends and community engagement patterns",
            risk_assessment=sections[2] if len(sections) > 2 and sections[2] else "Risk assessment based on social signals and market data",
            prediction=sections[3] if len(sections) > 3 and sections[3] else "AI-powered prediction based on comprehensive analysis",
            confidence_score=self._extract_confidence_score(analysis_text)
        )
    
    def _create_mock_analysis(self, token_address: str, symbol: str, token_data: Dict) -> TokenAnalysis:
        """Create a mock analysis for testing/fallback"""
        logger.info(f"Creating mock analysis for {symbol}")
        
        return TokenAnalysis(
            token_address=token_address,
            token_symbol=symbol,
            social_sentiment=f"**Social Sentiment Analysis for ${symbol}**\n\n**Overall Sentiment:** Demonstrating mixed to cautiously optimistic sentiment based on recent market activity.\n\n**Discussion Volume:** Moderate activity with consistent engagement from the crypto community.\n\n**Key Themes:** Price action analysis, technical discussions, and community building efforts.\n\n**Community Engagement:** Active participation in discussions with focus on long-term potential.\n\n*Note: This is demonstration data. Connect your GROK API key for live Twitter/X analysis.*",
            key_discussions=[
                "Technical development updates and roadmap discussions",
                "Community engagement metrics and growth patterns", 
                "Partnership announcements and collaboration talks",
                "Trading volume analysis and market dynamics",
                "Price movement discussions and technical analysis"
            ],
            influencer_mentions=[
                "Demo: @CryptoAnalyst mentioned positive technical outlook",
                "Demo: @BlockchainExpert shared comprehensive analysis",
                "Demo: @DefiTrader discussed volume trends and momentum"
            ],
            trend_analysis=f"**Discussion Trend Analysis for ${symbol}**\n\n**Trending Topics:** Technical updates, community growth initiatives, and market positioning strategies.\n\n**Volume Patterns:** Steady discussion volume with periodic spikes during significant announcements.\n\n**Engagement Trends:** Consistent community interaction with growing interest from new participants.\n\n**Narrative Themes:** Focus on innovation, development progress, and long-term value creation.\n\n*Live trend analysis requires GROK API access for real-time Twitter/X data.*",
            risk_assessment=f"**Risk Assessment for ${symbol}**\n\n**Low Risk Indicators:**\n- Consistent community engagement and development activity\n- No coordinated negative campaigns detected\n- Technical discussions remain constructive\n- Transparent communication from project stakeholders\n\n**Moderate Risk Factors:**\n- General market volatility affecting sentiment\n- Limited mainstream adoption currently\n- Competitive landscape considerations\n\n**Overall Risk Level: LOW-MODERATE**\n\nThe combination of active development, community support, and transparent operations suggests a lower risk profile.\n\n*Live risk assessment available with GROK API for real-time social signal analysis.*",
            prediction=f"**AI Prediction & Recommendations for ${symbol}**\n\n**Short-term Outlook (1-7 days):**\nExpected consolidation around current levels with potential for modest upward movement based on community sentiment and recent activity patterns.\n\n**Medium-term Outlook (1-4 weeks):**\nPositive momentum expected with continued development progress and community growth. Social engagement trends suggest sustained interest.\n\n**Key Catalysts to Monitor:**\n- Partnership announcements and strategic collaborations\n- Technical milestones and development updates\n- Community growth metrics and engagement levels\n- Market sentiment shifts and trading volume changes\n\n**Recommended Action:** HOLD/ACCUMULATE\n\nBased on current social sentiment analysis, the token shows promise for gradual appreciation with strong community backing.\n\n**Confidence Score:** 65%\n\n*This is demonstration analysis. Live predictions require GROK API for real-time social intelligence.*",
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
                "max_tokens": 1400,  # Increased for comprehensive response
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
        <body style="font-family: Arial, sans-serif; padding: 40px; background: #f5f5f5;">
            <div style="max-width: 800px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
                <h1 style="color: #333; margin-bottom: 20px;">🚀 Token Social Intelligence Platform</h1>
                <p style="color: #666; margin-bottom: 30px;">AI-powered social sentiment analysis for Solana tokens</p>
                <div style="margin-bottom: 20px;">
                    <input id="tokenAddress" placeholder="Enter Solana token address" 
                           style="padding: 15px; width: 70%; border: 2px solid #ddd; border-radius: 8px; font-size: 16px;">
                    <button onclick="analyzeToken()" 
                            style="padding: 15px 25px; background: #667eea; color: white; border: none; border-radius: 8px; font-size: 16px; cursor: pointer; margin-left: 10px;">
                        Analyze
                    </button>
                </div>
                <div id="status" style="margin: 20px 0; padding: 15px; background: #f0f8ff; border-radius: 8px; display: none;"></div>
                <div id="results" style="margin-top: 30px; padding: 20px; background: #f9f9f9; border-radius: 8px; white-space: pre-wrap; font-family: monospace; display: none;"></div>
            </div>
            <script>
                async function analyzeToken() {
                    const address = document.getElementById('tokenAddress').value.trim();
                    if (!address) {
                        alert('Please enter a token address');
                        return;
                    }
                    
                    const statusEl = document.getElementById('status');
                    const resultsEl = document.getElementById('results');
                    
                    statusEl.style.display = 'block';
                    statusEl.textContent = 'Analyzing token social sentiment...';
                    resultsEl.style.display = 'none';
                    
                    try {
                        const response = await fetch('/analyze', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({token_address: address})
                        });
                        
                        const data = await response.json();
                        statusEl.style.display = 'none';
                        resultsEl.style.display = 'block';
                        resultsEl.textContent = JSON.stringify(data, null, 2);
                    } catch (error) {
                        statusEl.textContent = 'Error: ' + error.message;
                        statusEl.style.background = '#ffe6e6';
                        statusEl.style.color = '#cc0000';
                    }
                }
                
                document.getElementById('tokenAddress').addEventListener('keypress', function(e) {
                    if (e.key === 'Enter') analyzeToken();
                });
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
        
        logger.info(f"Starting premium analysis for token: {token_address}")
        
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
        
        logger.info(f"Premium analysis completed successfully for {analysis.token_symbol}")
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
            'version': '3.0-premium'
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