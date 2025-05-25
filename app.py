from flask import Flask, render_template, request, jsonify
import requests
import os
from datetime import datetime, timedelta
import json
import re
from dataclasses import dataclass
from typing import List, Dict, Optional

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
    
    def fetch_dexscreener_data(self, address: str) -> Dict:
        """Fetch basic token data from DexScreener"""
        try:
            url = f"https://api.dexscreener.com/token-pairs/v1/solana/{address}"
            response = requests.get(url, timeout=10)
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
        except Exception as e:
            print(f"Error fetching DexScreener data: {e}")
            return {}
    
    def analyze_token_social_sentiment(self, token_symbol: str, token_address: str) -> TokenAnalysis:
        """Main analysis function using GROK Live Search"""
        
        # Get basic token data first
        token_data = self.fetch_dexscreener_data(token_address)
        symbol = token_data.get('symbol', token_symbol)
        
        # Phase 1: General sentiment analysis
        sentiment_analysis = self._analyze_general_sentiment(symbol, token_address)
        
        # Phase 2: Influencer and key account mentions
        influencer_analysis = self._analyze_influencer_mentions(symbol)
        
        # Phase 3: Trend and volume analysis
        trend_analysis = self._analyze_discussion_trends(symbol, token_address)
        
        # Phase 4: Risk assessment based on social signals
        risk_analysis = self._assess_social_risks(symbol, token_address)
        
        # Phase 5: Prediction and confidence scoring
        prediction = self._generate_prediction(
            token_data, sentiment_analysis, influencer_analysis, 
            trend_analysis, risk_analysis
        )
        
        return TokenAnalysis(
            token_address=token_address,
            token_symbol=symbol,
            social_sentiment=sentiment_analysis,
            key_discussions=trend_analysis.get('key_topics', []),
            influencer_mentions=influencer_analysis.get('mentions', []),
            trend_analysis=trend_analysis.get('summary', ''),
            risk_assessment=risk_analysis,
            prediction=prediction.get('prediction', ''),
            confidence_score=prediction.get('confidence', 0.0)
        )
    
    def _grok_live_search_query(self, prompt: str, search_params: Dict = None) -> str:
        """Make a GROK API call with live search"""
        try:
            default_search_params = {
                "mode": "on",
                "sources": [{"type": "x"}],
                "max_search_results": 20,
                "from_date": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
                "return_citations": True
            }
            
            if search_params:
                default_search_params.update(search_params)
            
            payload = {
                "model": "grok-3-latest",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert crypto social media analyst. Analyze Twitter/X discussions about cryptocurrency tokens to identify sentiment, trends, and key insights that could affect price movements."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "search_parameters": default_search_params,
                "max_tokens": 800,
                "temperature": 0.7
            }
            
            headers = {
                "Authorization": f"Bearer {self.grok_api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(GROK_URL, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
            
        except Exception as e:
            print(f"GROK API Error: {e}")
            return "Analysis unavailable due to API error."
    
    def _analyze_general_sentiment(self, symbol: str, address: str) -> str:
        """Analyze general sentiment around the token"""
        prompt = f"""
        Analyze the recent Twitter/X discussions about ${symbol} token (contract: {address}).
        
        Focus on:
        1. Overall sentiment (bullish/bearish/neutral)
        2. Volume of discussions (high/medium/low activity)
        3. Common themes in conversations
        4. Emotional tone of discussions
        
        Provide a concise summary with sentiment score and key observations.
        """
        
        return self._grok_live_search_query(prompt)
    
    def _analyze_influencer_mentions(self, symbol: str) -> Dict:
        """Analyze mentions by crypto influencers and key accounts"""
        prompt = f"""
        Find and analyze mentions of ${symbol} by crypto influencers, KOLs (Key Opinion Leaders), 
        and accounts with significant followings in the past 7 days.
        
        Look for:
        1. Who are the notable accounts discussing this token?
        2. What are they saying (positive/negative/neutral)?
        3. Are there any coordinated discussions or campaigns?
        4. Any endorsements or warnings from respected accounts?
        
        List the most important mentions and their impact potential.
        """
        
        search_params = {
            "sources": [{"type": "x"}],
            "max_search_results": 15
        }
        
        result = self._grok_live_search_query(prompt, search_params)
        
        return {
            'mentions': self._extract_key_mentions(result),
            'summary': result
        }
    
    def _analyze_discussion_trends(self, symbol: str, address: str) -> Dict:
        """Analyze trending topics and discussion patterns"""
        prompt = f"""
        Analyze trending discussion patterns for ${symbol} token over the past 7 days.
        
        Identify:
        1. What specific topics are trending (partnerships, listings, developments)?
        2. Discussion volume patterns (increasing/decreasing/stable)
        3. Recurring themes or narratives
        4. Community engagement levels
        5. Any coordinated activities or campaigns
        
        Highlight the most significant trends that could impact token performance.
        """
        
        result = self._grok_live_search_query(prompt)
        
        return {
            'key_topics': self._extract_key_topics(result),
            'summary': result
        }
    
    def _assess_social_risks(self, symbol: str, address: str) -> str:
        """Assess potential risks based on social signals"""
        prompt = f"""
        Assess potential risks for ${symbol} token based on social media signals and discussions.
        
        Look for red flags such as:
        1. Coordinated dump campaigns or negative sentiment
        2. Developer or team-related concerns
        3. Regulatory or compliance issues being discussed
        4. Technical problems or exploits mentioned
        5. Unusual trading pattern discussions
        6. Community fragmentation or disputes
        
        Provide a risk assessment with severity levels and explanations.
        """
        
        return self._grok_live_search_query(prompt)
    
    def _generate_prediction(self, token_data: Dict, sentiment: str, 
                           influencer: Dict, trends: Dict, risks: str) -> Dict:
        """Generate prediction based on all social intelligence"""
        prompt = f"""
        Based on the following comprehensive social media analysis for a token, provide a prediction:
        
        TOKEN DATA: {json.dumps(token_data, indent=2)}
        
        SENTIMENT ANALYSIS: {sentiment}
        
        INFLUENCER MENTIONS: {influencer.get('summary', '')}
        
        TREND ANALYSIS: {trends.get('summary', '')}
        
        RISK ASSESSMENT: {risks}
        
        Provide:
        1. Short-term prediction (1-7 days)
        2. Medium-term outlook (1-4 weeks)  
        3. Key catalysts to watch
        4. Confidence score (0-100)
        5. Recommended action (buy/sell/hold/avoid)
        
        Be specific and actionable.
        """
        
        result = self._grok_live_search_query(prompt)
        
        # Extract confidence score from result
        confidence = self._extract_confidence_score(result)
        
        return {
            'prediction': result,
            'confidence': confidence
        }
    
    def _extract_key_mentions(self, text: str) -> List[str]:
        """Extract key account mentions from analysis"""
        # Simple extraction - could be enhanced with NLP
        mentions = []
        lines = text.split('\n')
        for line in lines:
            if '@' in line and ('influencer' in line.lower() or 'kol' in line.lower() or 'account' in line.lower()):
                mentions.append(line.strip())
        return mentions[:5]  # Top 5
    
    def _extract_key_topics(self, text: str) -> List[str]:
        """Extract key discussion topics"""
        topics = []
        lines = text.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['partnership', 'listing', 'development', 'news', 'update']):
                topics.append(line.strip())
        return topics[:5]  # Top 5
    
    def _extract_confidence_score(self, text: str) -> float:
        """Extract confidence score from prediction text"""
        # Look for patterns like "confidence: 75" or "75% confidence"
        import re
        patterns = [
            r'confidence[:\s]*(\d+)',
            r'(\d+)%?\s*confidence',
            r'score[:\s]*(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1)) / 100.0
        
        return 0.5  # Default moderate confidence

# Initialize analyzer
analyzer = TokenSocialAnalyzer()

@app.route('/')
def index():
    # If no templates directory, serve HTML directly
    try:
        return render_template('index.html')
    except:
        # Fallback: serve the HTML directly
        return """
        <!DOCTYPE html>
        <html>
        <head><title>Token Social Intelligence Platform</title></head>
        <body>
            <h1>Token Social Intelligence Platform</h1>
            <p>Please ensure templates/index.html exists or check the deployment.</p>
            <div>
                <input id="tokenAddress" placeholder="Enter Solana token address" style="padding: 10px; width: 400px;">
                <button onclick="analyzeToken()" style="padding: 10px 20px;">Analyze</button>
            </div>
            <div id="results"></div>
            <script>
                async function analyzeToken() {
                    const address = document.getElementById('tokenAddress').value;
                    if (!address) return;
                    
                    const response = await fetch('/analyze', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({token_address: address})
                    });
                    
                    const data = await response.json();
                    document.getElementById('results').innerHTML = '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
                }
            </script>
        </body>
        </html>
        """

@app.route('/analyze', methods=['POST'])
def analyze_token():
    try:
        data = request.get_json()
        token_address = data.get('token_address', '').strip()
        
        if not token_address:
            return jsonify({'error': 'Token address is required'}), 400
        
        # Validate Solana address format (basic check)
        if len(token_address) < 32 or len(token_address) > 44:
            return jsonify({'error': 'Invalid Solana token address format'}), 400
        
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
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Analysis error: {e}")
        return jsonify({'error': 'Analysis failed. Please try again.'}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)