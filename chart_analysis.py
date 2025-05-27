"""
Safe Chart Analysis Module for ASK DEGEN - Render Compatible
Handles GPT-4o Vision API integration with DexScreener data integration
Uses lazy initialization to prevent deployment issues
"""

import os
import base64
import requests
from flask import request, jsonify
from werkzeug.utils import secure_filename
import logging
import json

logger = logging.getLogger(__name__)

class SafeChartAnalyzer:
    def __init__(self):
        """Initialize with lazy loading to prevent deployment issues"""
        self._client = None
        self._client_initialized = False
        
        # Allowed file extensions
        self.allowed_extensions = {'png', 'jpg', 'jpeg', 'webp'}
        # Max file size (10MB)
        self.max_file_size = 10 * 1024 * 1024
        
        logger.info("SafeChartAnalyzer initialized with lazy loading")

    @property
    def client(self):
        """Lazy load OpenAI client only when needed"""
        if not self._client_initialized:
            self._initialize_openai_client()
        return self._client

    def _initialize_openai_client(self):
        """Initialize OpenAI client with comprehensive error handling"""
        self._client_initialized = True
        
        try:
            openai_api_key = os.getenv('OPENAI_API_KEY')
            if not openai_api_key or openai_api_key in ['your-openai-api-key-here', '']:
                logger.warning("OpenAI API key not found or invalid")
                self._client = None
                return
            
            # Import OpenAI only when needed to avoid module-level issues
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=openai_api_key)
                logger.info("OpenAI client initialized successfully")
            except ImportError as e:
                logger.error(f"OpenAI package not available: {e}")
                self._client = None
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                # Try alternative initialization for compatibility
                try:
                    from openai import OpenAI
                    # Initialize with minimal parameters to avoid httpx conflicts
                    self._client = OpenAI(
                        api_key=openai_api_key,
                        timeout=30.0,
                        max_retries=2
                    )
                    logger.info("OpenAI client initialized with fallback method")
                except Exception as e2:
                    logger.error(f"Fallback OpenAI initialization also failed: {e2}")
                    self._client = None
                    
        except Exception as e:
            logger.error(f"Unexpected error during OpenAI initialization: {e}")
            self._client = None

    def is_allowed_file(self, filename):
        """Check if uploaded file has allowed extension"""
        return ('.' in filename and 
                filename.rsplit('.', 1)[1].lower() in self.allowed_extensions)

    def validate_file(self, file):
        """Validate uploaded file"""
        if not file or file.filename == '':
            return False, 'No file selected'
        
        if not self.is_allowed_file(file.filename):
            return False, 'Invalid file type. Please upload PNG, JPG, or WebP'
        
        # Check file size (read file to get size)
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        if file_size > self.max_file_size:
            return False, 'File size too large. Please upload an image under 10MB'
        
        return True, 'File is valid'

    def fetch_token_data(self, contract_address):
        """Fetch token data from DexScreener API"""
        try:
            if not contract_address or len(contract_address) < 32:
                return None
            
            url = f"https://api.dexscreener.com/latest/dex/tokens/{contract_address}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('pairs') and len(data['pairs']) > 0:
                    # Get the highest volume pair for most accurate data
                    pairs = sorted(data['pairs'], key=lambda x: float(x.get('volume', {}).get('h24', 0)), reverse=True)
                    pair = pairs[0]
                    base_token = pair.get('baseToken', {})
                    
                    return {
                        'symbol': base_token.get('symbol', 'UNKNOWN'),
                        'name': base_token.get('name', 'Unknown Token'),
                        'price_usd': float(pair.get('priceUsd', 0)),
                        'price_change_24h': float(pair.get('priceChange', {}).get('h24', 0)),
                        'price_change_6h': float(pair.get('priceChange', {}).get('h6', 0)),
                        'price_change_1h': float(pair.get('priceChange', {}).get('h1', 0)),
                        'volume_24h': float(pair.get('volume', {}).get('h24', 0)),
                        'volume_6h': float(pair.get('volume', {}).get('h6', 0)),
                        'market_cap': float(pair.get('marketCap', 0)),
                        'liquidity': float(pair.get('liquidity', {}).get('usd', 0)),
                        'fdv': float(pair.get('fdv', 0)),
                        'dex_name': pair.get('dexId', 'Unknown DEX'),
                        'chain': pair.get('chainId', 'Unknown Chain'),
                        'buys_24h': pair.get('txns', {}).get('h24', {}).get('buys', 0),
                        'sells_24h': pair.get('txns', {}).get('h24', {}).get('sells', 0),
                        'dex_url': pair.get('url', ''),
                        'created_at': pair.get('pairCreatedAt', None)
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching token data: {e}")
            return None

    def analyze_chart_image(self, image_file, contract_address=None):
        """
        Analyze chart image using GPT-4o Vision with optional token context
        
        Args:
            image_file: Flask uploaded file object
            contract_address: Optional Solana contract address for additional context
            
        Returns:
            dict: Analysis result with success status and content
        """
        try:
            # Check if OpenAI client is available
            if not self.client:
                return {
                    'success': False, 
                    'error': 'Chart analysis requires OpenAI API key. Please set OPENAI_API_KEY environment variable with a valid GPT-4o API key.'
                }

            # Validate file
            is_valid, message = self.validate_file(image_file)
            if not is_valid:
                return {'success': False, 'error': message}

            # Fetch token data if contract address provided
            token_data = None
            if contract_address:
                token_data = self.fetch_token_data(contract_address)

            # Read and encode image
            image_data = image_file.read()
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # Prepare enhanced prompt with token context
            prompt = self._get_enhanced_analysis_prompt(token_data)

            # Call OpenAI GPT-4o Vision API
            try:
                response = self.client.chat.completions.create(
                    model="chatgpt-4o-latest",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}",
                                        "detail": "high"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=3000,
                    temperature=0.7
                )
                
                analysis = response.choices[0].message.content
                
            except Exception as api_error:
                logger.error(f"OpenAI API call failed: {api_error}")
                return {
                    'success': False,
                    'error': f'OpenAI API call failed: {str(api_error)}. Please check your API key and try again.'
                }
            
            # Format the analysis with enhanced HTML
            formatted_analysis = self._format_enhanced_analysis_html(analysis, token_data)
            
            return {
                'success': True,
                'analysis': formatted_analysis,
                'token_data': token_data
            }

        except Exception as e:
            logger.error(f"Chart analysis error: {e}")
            return {'success': False, 'error': f'Analysis failed: {str(e)}'}

    def _get_enhanced_analysis_prompt(self, token_data=None):
        """Get the enhanced analysis prompt for GPT-4o with token context"""
        
        token_context = ""
        if token_data:
            token_context = f"""
ADDITIONAL TOKEN CONTEXT:
- Token: {token_data['symbol']} ({token_data['name']})
- Current Price: ${token_data['price_usd']:.8f}
- 24h Change: {token_data['price_change_24h']:+.2f}%
- 6h Change: {token_data['price_change_6h']:+.2f}%
- 1h Change: {token_data['price_change_1h']:+.2f}%
- 24h Volume: ${token_data['volume_24h']:,.0f}
- Market Cap: ${token_data['market_cap']:,.0f}
- Liquidity: ${token_data['liquidity']:,.0f}
- Buy/Sell Ratio (24h): {token_data['buys_24h']}/{token_data['sells_24h']}
- DEX: {token_data['dex_name']} on {token_data['chain']}

Please incorporate this real-time data into your analysis and compare it with what you see in the chart.
"""
        
        return f"""You are an expert cryptocurrency and financial chart analyst. Analyze this trading chart image and provide a comprehensive analysis.

{token_context}

IMPORTANT: Format your response as clean HTML with proper structure. Use the following guidelines:

1. **CHART PATTERN IDENTIFICATION** - FIRST identify any specific chart patterns you see:
   - Head and Shoulders, Inverse Head and Shoulders
   - Bull/Bear Flags and Pennants
   - Triangles (Ascending, Descending, Symmetrical)
   - Double/Triple Tops and Bottoms
   - Cup and Handle
   - Wedges (Rising/Falling)
   - Rectangles/Trading Ranges
   - Channels (Upward/Downward)

2. **Technical Analysis:**
   - Key support and resistance levels
   - Trend analysis (uptrend, downtrend, sideways)
   - Volume analysis if visible
   - Breakout/breakdown scenarios

3. **Technical Indicators:**
   - Moving averages positioning and crossovers
   - RSI levels and interpretation (overbought/oversold)
   - MACD signals if visible
   - Any other indicators present

4. **Price Action Analysis:**
   - Recent price movement analysis
   - Key levels to watch
   - Market structure analysis

5. **Trading Strategy:**
   - Potential entry points with rationale
   - Stop loss recommendations
   - Take profit targets
   - Risk assessment and position sizing

6. **Overall Assessment:**
   - Bullish, bearish, or neutral outlook
   - Confidence level (High/Medium/Low)
   - Time horizon for the analysis
   - Key risks and opportunities

FORMAT REQUIREMENTS:
- Use HTML tables for data comparisons
- Use proper headings (<h2>, <h3>)
- Use bullet points (<ul><li>) for lists
- Use <strong> for emphasis
- Use colored text for bullish (green) and bearish (red) signals
- Include relevant emojis for visual appeal
- Use <div class="highlight-box"> for important alerts
- Create summary tables where appropriate

If you identify specific chart patterns, clearly state them at the beginning with high confidence.
Be specific about price levels when visible on the chart.
Make the analysis actionable for traders.
"""

    def _format_enhanced_analysis_html(self, analysis_text, token_data=None):
        """Convert analysis to enhanced HTML format with better styling"""
        
        # Start with token data header if available
        html_content = []
        
        if token_data:
            html_content.append(f"""
            <div class="token-info-header">
                <h2>📊 {token_data['symbol']} Chart Analysis</h2>
                <div class="token-metrics-grid">
                    <div class="metric">
                        <span class="metric-label">Price</span>
                        <span class="metric-value">${token_data['price_usd']:.8f}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">24h Change</span>
                        <span class="metric-value {'positive' if token_data['price_change_24h'] > 0 else 'negative'}">
                            {token_data['price_change_24h']:+.2f}%
                        </span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Volume 24h</span>
                        <span class="metric-value">${token_data['volume_24h']:,.0f}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Market Cap</span>
                        <span class="metric-value">${token_data['market_cap']:,.0f}</span>
                    </div>
                </div>
            </div>
            """)
        
        # Process the analysis text - it should already be HTML formatted by GPT
        # Just add some additional styling classes
        processed_analysis = analysis_text
        
        # Add styling classes to common elements
        processed_analysis = processed_analysis.replace('<h2>', '<h2 class="analysis-heading">')
        processed_analysis = processed_analysis.replace('<h3>', '<h3 class="analysis-subheading">')
        processed_analysis = processed_analysis.replace('<table>', '<table class="analysis-table">')
        processed_analysis = processed_analysis.replace('<ul>', '<ul class="analysis-list">')
        
        # Add highlight boxes for important information
        if 'BULLISH' in processed_analysis.upper():
            processed_analysis = processed_analysis.replace('BULLISH', '<span class="bullish-signal">🟢 BULLISH</span>')
        if 'BEARISH' in processed_analysis.upper():
            processed_analysis = processed_analysis.replace('BEARISH', '<span class="bearish-signal">🔴 BEARISH</span>')
        if 'NEUTRAL' in processed_analysis.upper():
            processed_analysis = processed_analysis.replace('NEUTRAL', '<span class="neutral-signal">🟡 NEUTRAL</span>')
        
        html_content.append(processed_analysis)
        
        # Add analysis footer
        html_content.append("""
        <div class="analysis-footer">
            <p><strong>⚠️ Risk Disclaimer:</strong> This analysis is for educational purposes only. 
            Always conduct your own research and implement proper risk management strategies.</p>
        </div>
        """)
        
        return ''.join(html_content)


# Use lazy initialization to prevent module-level errors
def get_chart_analyzer():
    """Get chart analyzer instance with lazy initialization"""
    global _chart_analyzer_instance
    if '_chart_analyzer_instance' not in globals():
        _chart_analyzer_instance = SafeChartAnalyzer()
    return _chart_analyzer_instance

def handle_chart_analysis():
    """
    Safe Flask route handler for chart analysis with token context
    """
    try:
        # Check for chart image
        if 'chart' not in request.files:
            return jsonify({'success': False, 'error': 'No chart image provided'})
        
        file = request.files['chart']
        
        # Get optional contract address
        contract_address = request.form.get('contract_address', '').strip()
        if contract_address and len(contract_address) < 32:
            contract_address = None  # Invalid address, ignore
        
        # Get analyzer instance safely
        analyzer = get_chart_analyzer()
        
        # Analyze with enhanced features
        result = analyzer.analyze_chart_image(file, contract_address)
        
        return jsonify(result)

    except Exception as e:
        logger.error(f"Chart analysis handler error: {e}")
        return jsonify({'success': False, 'error': f'Analysis failed: {str(e)}'})

# For backward compatibility
def handle_enhanced_chart_analysis():
    """Backward compatibility wrapper"""
    return handle_chart_analysis()