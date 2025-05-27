"""
Chart Analysis Module for ASK DEGEN
Handles GPT-4o Vision API integration for chart analysis
"""

import os
import base64
from openai import OpenAI
from flask import request, jsonify
from werkzeug.utils import secure_filename

class ChartAnalyzer:
    def __init__(self):
        """Initialize the chart analyzer with OpenAI client"""
        self.client = None
        if os.getenv('OPENAI_API_KEY'):
            self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Allowed file extensions
        self.allowed_extensions = {'png', 'jpg', 'jpeg', 'webp'}
        # Max file size (10MB)
        self.max_file_size = 10 * 1024 * 1024

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

    def analyze_chart_image(self, image_file):
        """
        Analyze chart image using GPT-4o Vision
        
        Args:
            image_file: Flask uploaded file object
            
        Returns:
            dict: Analysis result with success status and content
        """
        try:
            if not self.client:
                return {
                    'success': False, 
                    'error': 'OpenAI API not configured. Please set OPENAI_API_KEY environment variable.'
                }

            # Validate file
            is_valid, message = self.validate_file(image_file)
            if not is_valid:
                return {'success': False, 'error': message}

            # Read and encode image
            image_data = image_file.read()
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # Prepare prompt for chart analysis
            prompt = self._get_analysis_prompt()

            # Call OpenAI GPT-4o Vision API
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
                max_tokens=2000,
                temperature=0.7
            )

            analysis = response.choices[0].message.content
            
            # Format the analysis with HTML
            formatted_analysis = self._format_analysis_html(analysis)
            
            return {
                'success': True,
                'analysis': formatted_analysis
            }

        except Exception as e:
            return {'success': False, 'error': f'Analysis failed: {str(e)}'}

    def _get_analysis_prompt(self):
        """Get the analysis prompt for GPT-4o"""
        return """You are an expert cryptocurrency and financial chart analyst. Analyze this trading chart image and provide a comprehensive analysis including:

1. **Technical Analysis:**
   - Key support and resistance levels
   - Chart patterns identified
   - Trend analysis (uptrend, downtrend, sideways)
   - Volume analysis if visible

2. **Technical Indicators:**
   - Moving averages positioning
   - RSI levels and interpretation
   - MACD signals if visible
   - Any other indicators present

3. **Price Action:**
   - Recent price movement analysis
   - Key levels to watch
   - Breakout/breakdown scenarios

4. **Trading Signals:**
   - Potential entry points
   - Stop loss recommendations
   - Take profit targets
   - Risk assessment

5. **Overall Assessment:**
   - Bullish, bearish, or neutral outlook
   - Confidence level in analysis
   - Key risks and opportunities

Format your response with clear headings and bullet points. Focus on actionable insights for traders. Be specific about price levels when visible on the chart."""

    def _format_analysis_html(self, analysis_text):
        """Convert plain text analysis to HTML format"""
        lines = analysis_text.split('\n')
        html_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Handle headings (lines starting with ##, **, or numbered lists)
            if line.startswith('##') or (line.startswith('**') and line.endswith('**')):
                heading_text = line.replace('##', '').replace('**', '').strip()
                html_content.append(f'<div class="content-heading">{heading_text}</div>')
            elif line.startswith('- ') or line.startswith('• '):
                # Handle bullet points
                bullet_text = line[2:].strip()
                html_content.append(f'<div class="bullet-point">{bullet_text}</div>')
            elif line[0].isdigit() and line[1:3] == '. ':
                # Handle numbered lists as bullet points
                bullet_text = line[3:].strip()
                html_content.append(f'<div class="bullet-point">{bullet_text}</div>')
            else:
                # Regular paragraph
                html_content.append(f'<p>{line}</p>')
        
        return ''.join(html_content)

# Create global instance
chart_analyzer = ChartAnalyzer()

def handle_chart_analysis():
    """
    Flask route handler for chart analysis
    Call this from your main Flask app
    """
    try:
        if 'chart' not in request.files:
            return jsonify({'success': False, 'error': 'No chart image provided'})
        
        file = request.files['chart']
        result = chart_analyzer.analyze_chart_image(file)
        
        return jsonify(result)

    except Exception as e:
        return jsonify({'success': False, 'error': f'Analysis failed: {str(e)}'})