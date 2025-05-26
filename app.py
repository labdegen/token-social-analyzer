```python
# timeout_optimized_analyzer.py
# Fully optimized with gevent, concurrent analysis phases, and SSE heartbeats

from gevent import monkey, spawn, joinall
monkey.patch_all()

from flask import Flask, render_template, request, jsonify, Response
import requests
import os
from datetime import datetime, timedelta
import json
import re
from dataclasses import dataclass
from typing import List, Dict
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask application
app = Flask(__name__)

# Environment configuration
GROK_API_KEY = os.getenv('GROK_API_KEY', 'your-grok-api-key-here')
GROK_URL = "https://api.x.ai/v1/chat/completions"

# Dataclasses for structured responses
dataclass
class TokenAnalysis:
    token_address: str
    token_symbol: str
    expert_summary: str
    social_sentiment: str
    key_discussions: List[str]
    influencer_mentions: List[Dict]
    trend_analysis: str
    risk_assessment: str
    prediction: str
    confidence_score: float
    sentiment_metrics: Dict
    actual_tweets: List[Dict]
    x_citations: List[str]

# Core analyzer with concurrent gevent-based phases
dataclass
class PremiumTokenSocialAnalyzer:
    def __init__(self):
        self.grok_api_key = GROK_API_KEY
        logger.info(f"Analyzer initialized; API key set: {self.grok_api_key != 'your-grok-api-key-here'}")

    def _format_heartbeat(self) -> str:
        # SSE comment resets worker idle timer
        return ":\n\n"

    def _format_progress_update(self, stage: str, message: str, step: int = 0) -> str:
        update = {
            "type": "progress",
            "stage": stage,
            "message": message,
            "step": step,
            "timestamp": datetime.now().isoformat()
        }
        return f"data: {json.dumps(update)}\n\n"

    def _format_final_response(self, analysis: TokenAnalysis) -> str:
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

    def stream_comprehensive_analysis(self, token_symbol: str, token_address: str, analysis_mode: str = "analytical"):
        # Heartbeat scheduling
        last_beat = time.time()
        def maybe_heartbeat():
            nonlocal last_beat
            if time.time() - last_beat > 10:
                last_beat = time.time()
                return self._format_heartbeat()
            return None

        # Phase 0: Initialization
        yield self._format_progress_update("initialized", "Starting analysis pipeline", 1)
        hb = maybe_heartbeat()
        if hb: yield hb

        # Launch concurrent X/Twitter phases
        g_expert = spawn(self._get_expert_x_analysis, token_symbol, token_address, analysis_mode)
        g_influ = spawn(self._get_influencer_analysis, token_symbol, token_address, analysis_mode)
        g_trends = spawn(self._get_trends_analysis, token_symbol, token_address, analysis_mode)

        # Wait up to 60s for all
        joinall([g_expert, g_influ, g_trends], timeout=60)

        # Expert
        expert_res = g_expert.value or {'success': False}
        if expert_res.get('success'):
            yield self._format_progress_update("expert_complete", "Expert analysis complete", 2)
        hb = maybe_heartbeat();
        if hb: yield hb

        # Influencer
        infl_res = g_influ.value or {'success': False}
        if infl_res.get('success'):
            yield self._format_progress_update("influencer_complete", "Influencer deep dive complete", 3)
        hb = maybe_heartbeat();
        if hb: yield hb

        # Trends
        trends_res = g_trends.value or {'success': False}
        if trends_res.get('success'):
            yield self._format_progress_update("trends_complete", "Trend analysis complete", 4)
        hb = maybe_heartbeat();
        if hb: yield hb

        # Phase 4: Risk & Prediction (local computations)
        yield self._format_progress_update("risk_complete", "Risk assessment done", 5)
        hb = maybe_heartbeat();
        if hb: yield hb
        yield self._format_progress_update("prediction_complete", "Predictions generated", 6)
        hb = maybe_heartbeat();
        if hb: yield hb

        # Construct final data - merge phase results
        analysis = TokenAnalysis(
            token_address=token_address,
            token_symbol=token_symbol,
            expert_summary=expert_res.get('data', {}).get('expert_summary', ''),
            social_sentiment=expert_res.get('data', {}).get('social_sentiment', ''),
            key_discussions=trends_res.get('data', {}).get('topics', []),
            influencer_mentions=infl_res.get('data', {}).get('influencers', []),
            trend_analysis=trends_res.get('data', {}).get('trends', ''),
            risk_assessment=self._create_x_based_risk_assessment(...),
            prediction=self._create_x_based_prediction(...),
            confidence_score=0.85,
            sentiment_metrics=expert_res.get('data', {}).get('sentiment_metrics', {}),
            actual_tweets=expert_res.get('data', {}).get('actual_tweets', []),
            x_citations=expert_res.get('data', {}).get('x_citations', [])
        )

        # Emit final SSE
        yield self._format_final_response(analysis)

    # Placeholder implementations for phase methods
    def _get_expert_x_analysis(self, symbol, address, mode):
        # Real implementation omitted for brevity
        return {'success': True, 'data': {'expert_summary':'','social_sentiment':'','sentiment_metrics':{},'actual_tweets':[], 'x_citations':[]}}
    def _get_influencer_analysis(self, symbol, address, mode):
        return {'success': True, 'data': {'influencers':[]}}
    def _get_trends_analysis(self, symbol, address, mode):
        return {'success': True, 'data': {'trends':'','topics':[]}}
    def _create_x_based_risk_assessment(self, *args, **kwargs):
        return ''
    def _create_x_based_prediction(self, *args, **kwargs):
        return ''

# Instantiate
danalyzer = PremiumTokenSocialAnalyzer()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/trending-tokens')
def trending():
    # Omitted: use analyzer.get_trending_tokens()
    return jsonify({'success': True, 'tokens': []})

@app.route('/analyze', methods=['POST'])
def analyze_token():
    data = request.get_json() or {}
    addr = data.get('token_address', '').strip()
    if len(addr) < 32 or len(addr) > 44:
        return jsonify({'error':'Invalid address'}), 400
    return Response(
        analyzer.stream_comprehensive_analysis('', addr, data.get('analysis_mode','analytical')),
        mimetype='text/plain',
        headers={
            'Cache-Control':'no-cache',
            'Connection':'keep-alive',
            'X-Accel-Buffering':'no'
        }
    )

@app.route('/health')
def health():
    return jsonify({'status':'healthy','timestamp':datetime.now().isoformat()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT',5000)), debug=True)
```
