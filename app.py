# app.py
# Simplified: Direct Grok chat completions with live search, streaming SSE to client

from flask import Flask, render_template, request, Response, jsonify
import requests
import os
import json
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)

# Config
GROK_API_KEY = os.getenv('GROK_API_KEY', 'your-grok-api-key-here')
GROK_URL = "https://api.x.ai/v1/chat/completions"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/trending-tokens')
def trending():
    # Example: live search for trending newly-launched Solana tokens
    payload = {
        "model": "grok-3-latest",
        "messages": [
            {"role": "system", "content": "You are a crypto trends analyzer."},
            {"role": "user", "content": "List 5 recently launched Solana tokens gaining traction on X/Twitter in the last 7 days, with symbol, contract address, mention count, top 3 influencers and a viral tweet snippet."}
        ],
        "search_parameters": {
            "mode": "on",
            "sources": [{"type": "x"}],
            "from_date": (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d"),
            "max_search_results": 50,
            "return_citations": True
        },
        "temperature": 0.3,
        "stream": False
    }
    headers = {"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"}
    resp = requests.post(GROK_URL, json=payload, headers=headers, timeout=30)
    if resp.status_code != 200:
        return jsonify({'success': False, 'error': resp.text}), 500
    data = resp.json()
    return jsonify({'success': True, 'result': data})

@app.route('/analyze', methods=['POST'])
def analyze_token():
    data = request.get_json() or {}
    address = data.get('token_address', '').strip()
    mode = data.get('analysis_mode', 'analytical')
    if len(address) < 32 or len(address) > 44:
        return jsonify({'error': 'Invalid Solana token address'}), 400

    # Build prompt
    system_msg = (
        f"You are a Solana token social media analyst. Provide comprehensive analysis for token {address}."
    )
    user_msg = f"""Analyze token at {address} on X/Twitter right now.
1. Expert summary combining price action and social sentiment.
2. Top influencer tweets with handles and follower counts.
3. Viral trends and key discussion topics.
4. Risk assessment from social data.
5. Short-term price prediction."""

    payload = {
        "model": "grok-3-latest",
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        "search_parameters": {
            "mode": "on",
            "sources": [{"type": "x"}],
            "from_date": (datetime.utcnow() - timedelta(days=3)).strftime("%Y-%m-%d"),
            "return_citations": True,
            "max_search_results": 30
        },
        "temperature": 0.2,
        "stream": True
    }
    headers = {"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"}

    # Stream response chunks as SSE
    def generate():
        try:
            r = requests.post(GROK_URL, json=payload, headers=headers, stream=True, timeout=60)
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if line:
                    yield f"data: {line}\n\n"
        except Exception as e:
            err = {"type": "error", "message": str(e)}
            yield f"data: {json.dumps(err)}\n\n"

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'timestamp': datetime.utcnow().isoformat()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=True)
```
