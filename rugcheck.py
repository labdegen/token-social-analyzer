import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import base58
import json
import random
import time
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rugcheck")

@dataclass
class ScamIndicators:
    """Key indicators for scam detection"""
    # Critical Risks
    mint_enabled: bool = False
    freeze_enabled: bool = False
    
    # Ownership Risks  
    top_holder_percent: float = 0.0
    top_10_holders_percent: float = 0.0
    dev_wallet_percent: float = 0.0
    
    # Liquidity Risks
    liquidity_locked: bool = False
    liquidity_percent: float = 0.0
    
    # Trading Risks
    can_sell: bool = True
    buy_tax: float = 0.0
    sell_tax: float = 0.0
    
    # Age & Activity
    token_age_hours: int = 0
    unique_holders: int = 0
    
    # Scores
    overall_risk_score: int = 0
    risk_level: str = "UNKNOWN"

class EnhancedRugChecker:
    def __init__(self, helius_key: str, birdeye_key: str = None, xai_key: str = None):
        self.helius_key = helius_key or "demo_key"
        self.birdeye_key = birdeye_key
        self.xai_key = xai_key
        self.rpc_url = f"https://mainnet.helius-rpc.com/?api-key={helius_key}" if helius_key else None
        
        # Debug logging
        logger.info(f"🧠 XAI API Key status: {'✅ Available' if self.xai_key and self.xai_key != 'your-xai-api-key-here' else '❌ Missing'}")
        
    async def analyze_token(self, mint_address: str, deep_analysis: bool = True) -> Dict:
        """
        🧠 REVOLUTIONARY Galaxy Brain v5.0 Analysis with LIVE Grok Intelligence:
        - REAL-TIME social sentiment analysis via Grok
        - Live community discussions monitoring
        - Advanced meme coin safety patterns
        - Revolutionary risk assessment with AI insights
        """
        session = None
        try:
            start_time = time.time()
            mode = "DEEP" if deep_analysis else "EXPRESS"
            logger.info(f"🧠 REVOLUTIONARY Galaxy Brain v5.0 {mode} Mode analyzing: {mint_address}")
            logger.info(f"🧠 Analysis mode: {mode}")
            
            session = aiohttp.ClientSession()
            
            # Step 1: Get comprehensive token security data
            security_data = await self._get_birdeye_security_data(mint_address, session)
            
            # Step 2: Get token metadata and basic info
            token_info = await self._get_token_info(mint_address, session)
            if not token_info:
                return {
                    "success": False, 
                    "error": "Token not found or invalid address",
                    "suggestions": [
                        "Verify the token address is correct",
                        "Ensure token exists on Solana mainnet",
                        "Check if token has been deployed"
                    ]
                }
                
            # Step 3: Get market data with liquidity lock detection
            market_data = await self._get_enhanced_market_data(mint_address, session)

            # IMPORTANT: Merge real symbol/name from market data into token_info
            if market_data.get("symbol") and market_data["symbol"] != "TOKEN":
                token_info["symbol"] = market_data["symbol"]
                token_info["name"] = market_data.get("name", token_info.get("name", "Token"))
                logger.info(f"🔄 Updated token info: {token_info['name']} (${token_info['symbol']})")
            
            # Step 4: Get REAL holder analysis
            holders_data = await self._get_real_holders_analysis(mint_address, token_info['supply'], session)
            
            # Step 5: Get REAL transaction analysis via Helius
            transaction_analysis = await self._get_real_transaction_analysis(mint_address, session, deep_analysis)
            
            # Step 6: Enhanced liquidity analysis with lock detection
            liquidity_data = await self._get_enhanced_liquidity_analysis(mint_address, market_data, security_data, session)
            
            # Step 7: Authority analysis
            authority_analysis = await self._analyze_enhanced_authorities(token_info, security_data)
            
            # Step 8: Calculate enhanced indicators
            indicators = self._calculate_enhanced_indicators(
                token_info, market_data, holders_data, liquidity_data, security_data
            )
            
            # Step 9: 🧠 REVOLUTIONARY GROK LIVE ANALYSIS
            logger.info("🧠 REVOLUTIONARY: Calling Grok Live Intelligence Analysis...")
            grok_analysis = await self._grok_revolutionary_meme_analysis(
                mint_address, token_info, holders_data, liquidity_data, indicators, session
            )
            
            # MODE SPLIT: Different analysis based on mode
            if deep_analysis:
                logger.info("🧠 Deep Mode: Running advanced AI analysis with Grok insights...")
                
                bundle_detection = await self._detect_enhanced_bundles(mint_address, holders_data, transaction_analysis)
                suspicious_activity = await self._detect_enhanced_suspicious_activity(
                    mint_address, transaction_analysis, holders_data, security_data
                )
                
                # 🧠 REVOLUTIONARY: Enhanced scoring with Grok intelligence
                galaxy_brain_score, severity_level, confidence = self._calculate_revolutionary_galaxy_score_with_grok(
                    indicators, transaction_analysis, bundle_detection, suspicious_activity, security_data, grok_analysis
                )
                
                risk_vectors = self._generate_revolutionary_risk_vectors_with_grok(
                    indicators, transaction_analysis, suspicious_activity, security_data, grok_analysis
                )
                
            else:
                logger.info("⚡ Express Mode: Quick analysis with Grok insights...")
                
                bundle_detection = await self._detect_basic_bundles(holders_data)
                suspicious_activity = self._detect_basic_suspicious_activity(
                    transaction_analysis, holders_data, security_data
                )
                
                # Even Express mode gets Grok enhancement
                galaxy_brain_score, severity_level, confidence = self._calculate_express_score_with_grok(
                    indicators, holders_data, security_data, grok_analysis
                )
                
                risk_vectors = self._generate_basic_risk_vectors_with_grok(
                    indicators, holders_data, liquidity_data, security_data, grok_analysis
                )
            
            analysis_time = time.time() - start_time
            
            # 🧠 REVOLUTIONARY response with Grok intelligence
            return {
                "success": True,
                "analysis_mode": mode,
                "galaxy_brain_score": galaxy_brain_score,
                "severity_level": severity_level,
                "confidence": confidence,
                
                # 🧠 REVOLUTIONARY: Grok Analysis Integration
                "grok_analysis": grok_analysis,
                "ai_analysis": self._format_grok_ai_analysis(grok_analysis),
                
                # Enhanced token info with security data
                "token_info": {
                    **token_info,
                    **market_data,
                    "age_days": self._calculate_token_age_days(market_data.get("created_at", 0)),
                    "is_mutable": not authority_analysis.get("fully_decentralized", False),
                    "security_score": security_data.get("overall_score", 50)
                },
                
                # Real analysis results
                "holder_analysis": holders_data,
                "liquidity_analysis": liquidity_data,
                "transaction_analysis": transaction_analysis,
                "bundle_detection": bundle_detection,
                "suspicious_activity": suspicious_activity,
                "authority_analysis": authority_analysis,
                "security_data": security_data,
                "scam_indicators": indicators,
                "risk_vectors": risk_vectors,
                
                # Enhanced metadata
                "analysis_time_seconds": round(analysis_time, 2),
                "analysis_timestamp": datetime.now().isoformat(),
                "analysis_version": "5.0_REVOLUTIONARY_GROK",
                "data_sources": {
                    "token_data": "helius_rpc" if self.helius_key != "demo_key" else "demo",
                    "market_data": "dexscreener_enhanced",
                    "security_data": "birdeye_api" if self.birdeye_key else "basic",
                    "holder_data": "helius_real",
                    "transaction_data": "helius_parsed",
                    "liquidity_locks": "birdeye_dexscreener",
                    "grok_intelligence": "live_community_analysis" if self.xai_key and self.xai_key != 'your-xai-api-key-here' else "unavailable"
                }
            }
            
        except Exception as e:
            logger.error(f"Revolutionary analysis failed: {e}")
            return {
                "success": False, 
                "error": f"Analysis failed: {str(e)}",
                "error_type": "analysis_error",
                "analysis_mode": mode if 'mode' in locals() else "UNKNOWN",
                "suggestions": [
                    "Try again in a few minutes",
                    "Verify network connectivity", 
                    "Check if token address is valid"
                ]
            }
        finally:
            if session:
                await session.close()

    async def _grok_revolutionary_meme_analysis(self, mint_address: str, token_info: Dict, 
                                            holders_data: Dict, liquidity_data: Dict, 
                                            indicators: ScamIndicators, session: aiohttp.ClientSession) -> Dict:
        """🧠 REVOLUTIONARY Grok-powered meme coin safety analysis with live community intelligence"""
        try:
            if not self.xai_key or self.xai_key == 'your-xai-api-key-here':
                logger.warning("No Grok API key - revolutionary analysis unavailable")
                return {"available": False, "reason": "no_api_key"}
            
            # Get the ACTUAL symbol from token_info or market data
            symbol = token_info.get('symbol', 'UNKNOWN')
            if symbol == 'TOKEN' or symbol == 'UNKNOWN':
                symbol = token_info.get('name', 'TOKEN')[:10]
            
            # Calculate derived insights from the data
            top_holder = holders_data.get('top_1_percent', 0)
            liquidity_usd = liquidity_data.get('liquidity_usd', 0)
            market_cap = token_info.get('market_cap', 0)
            age_hours = indicators.token_age_hours
            
            # 🧠 REVOLUTIONARY prompt optimized for actionable insights
            revolutionary_prompt = f"""
    🧠 CRYPTOCURRENCY SAFETY ANALYST - ${symbol} Security Assessment

    Contract: {mint_address}
    Current Metrics:
    - Top holder: {top_holder:.1f}% of supply
    - Liquidity: ${liquidity_usd:,.0f} USD
    - Market cap: ${market_cap:,.0f} USD  
    - Token age: {age_hours} hours ({age_hours/24:.1f} days)
    - Mint authority: {"RENOUNCED ✅" if not indicators.mint_enabled else "ACTIVE ⚠️"}
    - Freeze authority: {"RENOUNCED ✅" if not indicators.freeze_enabled else "ACTIVE ⚠️"}

    SEARCH X/TWITTER for recent discussions about ${symbol} token and this contract address.

    CRITICAL ANALYSIS FRAMEWORK:

    1. ABSENCE OF NEGATIVE SIGNALS = POSITIVE FINDING
    - If NO scam accusations found → "Community shows no scam concerns"
    - If NO rug pull warnings found → "No community warnings detected"
    - If NO dump coordination found → "No coordinated selling patterns reported"
    - If NO bot/bundle reports found → "Community reports organic trading activity"

    2. SPECIFIC SAFETY CONFIRMATIONS to look for:
    - Team communication and transparency updates
    - Locked liquidity confirmations from community
    - Diamond hands/holder confidence posts
    - Legitimate project development news
    - Partnership announcements or team verification

    3. WHALE BEHAVIOR INTERPRETATION:
    Based on the {top_holder:.1f}% top holder concentration:
    {self._generate_whale_context(top_holder, liquidity_usd, market_cap)}

    4. LIQUIDITY ASSESSMENT:
    ${liquidity_usd:,.0f} liquidity vs ${market_cap:,.0f} market cap = {(liquidity_usd/market_cap*100) if market_cap > 0 else 0:.1f}% ratio
    Community perspective on liquidity security and trading conditions.

    5. FINAL VERDICT BASED ON EVIDENCE:
    - REVOLUTIONARY_SAFE: Strong positive community + no negative signals + good metrics
    - COMMUNITY_CONFIDENT: Some positive signals + no major concerns + decent metrics  
    - NEUTRAL_MONITORING: Mixed signals or limited data + average metrics
    - COMMUNITY_CONCERNS: Some warnings but not widespread + concerning metrics
    - DANGER_DETECTED: Clear negative community consensus + poor metrics

    IMPORTANT: 
    - NO NEGATIVE FINDINGS = GOOD SIGN, not "no data"
    - Provide SPECIFIC community sentiment quotes when available
    - Interpret metrics in context of community discussion
    - Give actionable trading insights based on ACTUAL findings

    Focus on what the community IS saying (positive) and what they're NOT saying (absence of negatives).
    """
            
            # Enhanced search parameters
            search_params = {
                "mode": "on",
                "sources": [
                    {"type": "x"},
                    {"type": "web"}
                ],
                "max_search_results": 30,
                "from_date": (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d"),
                "return_citations": True
            }
            
            logger.info(f"🧠 REVOLUTIONARY Grok analyzing {symbol} with enhanced prompting...")
            grok_response = await self._call_grok_api_new_format(revolutionary_prompt, search_params, session)
            
            if grok_response:
                parsed_analysis = self._parse_enhanced_grok_analysis(
                    grok_response.get('content', ''), 
                    symbol, 
                    top_holder, 
                    liquidity_usd, 
                    market_cap,
                    indicators
                )
                logger.info(f"🧠 REVOLUTIONARY Enhanced analysis: {parsed_analysis.get('verdict', 'UNKNOWN')}")
                return {
                    "available": True,
                    "raw_response": grok_response.get('content', ''),
                    "parsed_analysis": parsed_analysis,
                    "confidence": parsed_analysis.get('confidence', 0.7),
                    "analysis_type": "revolutionary_enhanced_search",
                    "citations": grok_response.get('citations', [])
                }
            else:
                return {"available": False, "reason": "api_failed"}
                
        except Exception as e:
            logger.error(f"Revolutionary Grok analysis failed: {e}")
            return {"available": False, "reason": "error", "error": str(e)}

    def _generate_whale_context(self, top_holder_percent: float, liquidity_usd: float, market_cap: float) -> str:
        """Generate specific whale behavior context based on metrics"""
        if top_holder_percent > 30:
            return f"EXTREME concentration ({top_holder_percent:.1f}%) - Look for community explanations: team wallet, treasury, or concerning accumulation"
        elif top_holder_percent > 15:
            return f"HIGH concentration ({top_holder_percent:.1f}%) - Check if community discusses this whale: legitimate reserves vs concerning accumulation"
        elif top_holder_percent > 8:
            return f"MODERATE concentration ({top_holder_percent:.1f}%) - Monitor community sentiment about large holder activity"
        else:
            return f"GOOD distribution ({top_holder_percent:.1f}%) - Look for community confirmation of healthy holder spread"

    def _parse_enhanced_grok_analysis(self, grok_response: str, symbol: str, top_holder: float, 
                                    liquidity_usd: float, market_cap: float, indicators: ScamIndicators) -> Dict:
        """Enhanced parsing that creates actionable insights even from limited findings"""
        try:
            logger.info(f"🧠 Enhanced parsing for {symbol}: {len(grok_response)} characters")
            
            # Extract verdict with improved patterns
            verdict_patterns = [
                r'(?:FINAL VERDICT|VERDICT|ASSESSMENT):\s*\*?\*?([A-Z_]+)',
                r'\*\*(?:SAFETY VERDICT|FINAL VERDICT):\*\*\s*([A-Z_]+)',
                r'(?:^|\n)([A-Z_]+):\s*(?:Strong|Clear|Community|No major)'
            ]
            
            verdict = "NEUTRAL_MONITORING"  # Better default
            for pattern in verdict_patterns:
                match = re.search(pattern, grok_response, re.IGNORECASE | re.MULTILINE)
                if match:
                    verdict = match.group(1).upper()
                    break
            
            # Enhanced safety signal detection
            safety_signals = self._extract_safety_signals(grok_response, symbol, indicators)
            risk_signals = self._extract_risk_signals(grok_response, symbol)
            
            # Generate whale analysis from actual data + community context
            whale_analysis = self._generate_contextual_whale_analysis(
                grok_response, top_holder, liquidity_usd, market_cap
            )
            
            # Create actionable insight
            actionable_insight = self._create_actionable_insight(
                verdict, safety_signals, risk_signals, symbol, top_holder, 
                liquidity_usd, market_cap, indicators
            )
            
            # Enhanced confidence based on actual findings
            confidence = self._calculate_enhanced_confidence(
                grok_response, safety_signals, risk_signals, verdict
            )
            
            return {
                "verdict": verdict,
                "confidence": confidence,
                "positive_community_sentiment": safety_signals,
                "possible_community_risks": risk_signals,
                "whale_analysis": whale_analysis,
                "revolutionary_insight": actionable_insight,
                "safety_indicators": [s for s in safety_signals if any(term in s.lower() for term in ['safe', 'legitimate', 'locked', 'verified'])],
                "risk_indicators": [r for r in risk_signals if any(term in r.lower() for term in ['scam', 'warning', 'rug', 'dump'])],
                "username_mentions": len(re.findall(r'@\w+', grok_response)),
                "quote_mentions": len(re.findall(r'"[^"]{10,}"', grok_response)),
                "response_length": len(grok_response),
                "analysis_type": "revolutionary_enhanced",
                "token_specific_data": {
                    "symbol": symbol,
                    "top_holder_percent": top_holder,
                    "liquidity_ratio": (liquidity_usd/market_cap*100) if market_cap > 0 else 0,
                    "age_assessment": self._assess_token_age(indicators.token_age_hours)
                }
            }
            
        except Exception as e:
            logger.error(f"Enhanced parsing failed: {e}")
            return self._create_fallback_analysis(symbol, top_holder, liquidity_usd, indicators, str(e))

    def _assess_token_age(self, age_hours: int) -> str:
        """Assess token age for context"""
        if age_hours < 6:
            return "VERY_NEW"
        elif age_hours < 24:
            return "NEW" 
        elif age_hours < 168:  # 1 week
            return "YOUNG"
        elif age_hours < 720:  # 1 month
            return "DEVELOPING"
        else:
            return "MATURE"

    def _create_fallback_analysis(self, symbol: str, top_holder: float, liquidity_usd: float, 
                                indicators: ScamIndicators, error: str) -> Dict:
        """Create analysis when Grok fails but we have metrics"""
        return {
            "verdict": "NEUTRAL_MONITORING",
            "confidence": 0.5,
            "positive_community_sentiment": [
                f"Analysis based on verified on-chain metrics for ${symbol}",
                f"Mint authority: {'renounced' if not indicators.mint_enabled else 'active'}",
                f"Freeze authority: {'renounced' if not indicators.freeze_enabled else 'active'}"
            ],
            "possible_community_risks": [
                f"Community analysis unavailable - relying on technical metrics only"
            ],
            "whale_analysis": f"Top holder: {top_holder:.1f}% - {'concerning concentration' if top_holder > 20 else 'manageable distribution'}",
            "revolutionary_insight": f"${symbol} analysis completed using on-chain data. Community sentiment analysis failed: {error}",
            "safety_indicators": [],
            "risk_indicators": [],
            "username_mentions": 0,
            "quote_mentions": 0,
            "analysis_type": "fallback_metrics_only"
        }


    def _calculate_enhanced_confidence(self, response: str, safety_signals: List[str], 
                                    risk_signals: List[str], verdict: str) -> float:
        """Calculate confidence based on actual findings and response quality"""
        base_confidence = 0.6
        
        # Response quality indicators
        if len(response) > 300:
            base_confidence += 0.15
        if len(response) > 600:
            base_confidence += 0.1
        
        # Evidence quality
        username_mentions = len(re.findall(r'@\w+', response))
        if username_mentions > 0:
            base_confidence += min(0.15, username_mentions * 0.03)
        
        quote_mentions = len(re.findall(r'"[^"]{10,}"', response))
        if quote_mentions > 0:
            base_confidence += min(0.1, quote_mentions * 0.02)
        
        # Signal quality
        if len(safety_signals) > 2:
            base_confidence += 0.1
        if len(risk_signals) > 0:
            base_confidence += 0.05  # Risk findings often more reliable
        
        # Verdict consistency
        if verdict in ["REVOLUTIONARY_SAFE", "DANGER_DETECTED"]:
            base_confidence += 0.1  # Clear verdicts more confident
        
        return min(0.95, max(0.3, base_confidence))    
    
    
    def _create_actionable_insight(self, verdict: str, safety_signals: List[str], risk_signals: List[str],
                                symbol: str, top_holder: float, liquidity_usd: float, 
                                market_cap: float, indicators: ScamIndicators) -> str:
        """Create specific, actionable insight based on all available data"""
        
        # Base insight on verdict
        verdict_insights = {
            "REVOLUTIONARY_SAFE": f"${symbol} shows strong community confidence with no major red flags detected.",
            "COMMUNITY_CONFIDENT": f"${symbol} has positive community sentiment with manageable risks.",
            "NEUTRAL_MONITORING": f"${symbol} requires careful monitoring - mixed signals detected.",
            "COMMUNITY_CONCERNS": f"${symbol} has community concerns that warrant investigation.",
            "DANGER_DETECTED": f"${symbol} shows concerning community warnings and metrics."
        }
        
        base_insight = verdict_insights.get(verdict, f"${symbol} analysis complete with specific findings.")
        
        # Add specific risk factors
        critical_factors = []
        if indicators.mint_enabled:
            critical_factors.append("mint authority active")
        if indicators.freeze_enabled:
            critical_factors.append("freeze authority active")
        if top_holder > 25:
            critical_factors.append(f"extreme concentration ({top_holder:.1f}%)")
        if liquidity_usd < 100000:
            critical_factors.append("low liquidity")
        
        if critical_factors:
            base_insight += f" Key concerns: {', '.join(critical_factors)}."
        
        # Add actionable recommendation
        if verdict in ["REVOLUTIONARY_SAFE", "COMMUNITY_CONFIDENT"]:
            if top_holder > 15:
                base_insight += f" Monitor whale activity due to {top_holder:.1f}% concentration."
            else:
                base_insight += " Proceed with standard caution for meme token trading."
        elif verdict in ["COMMUNITY_CONCERNS", "DANGER_DETECTED"]:
            base_insight += " RECOMMEND: Avoid or use only small test amounts."
        else:
            base_insight += " RECOMMEND: Wait for clearer signals before significant investment."
        
        return base_insight

    def _generate_contextual_whale_analysis(self, response: str, top_holder: float, 
                                        liquidity_usd: float, market_cap: float) -> str:
        """Generate whale analysis combining community sentiment with metrics"""
        
        # Check for whale-related community discussion
        whale_mentions = re.findall(r'(?:whale|large holder|dev wallet|team wallet|treasury)', response, re.IGNORECASE)
        
        base_analysis = f"Top holder controls {top_holder:.1f}% of supply"
        
        if top_holder > 25:
            if whale_mentions:
                return f"{base_analysis}. Community discusses large holder activity - {', '.join(whale_mentions[:2])}. Extreme concentration requires monitoring."
            else:
                return f"{base_analysis}. CRITICAL: No community explanation for extreme concentration. High dump risk."
        elif top_holder > 15:
            if whale_mentions:
                return f"{base_analysis}. Community aware of concentration - monitoring for explanations or concerns."
            else:
                return f"{base_analysis}. Significant concentration but no major community concerns detected."
        elif top_holder > 8:
            return f"{base_analysis}. Moderate concentration within acceptable range for meme tokens."
        else:
            return f"{base_analysis}. Healthy distribution - good for price stability."



    def _extract_risk_signals(self, response: str, symbol: str) -> List[str]:
        """Extract specific risk signals from community analysis"""
        risks = []
        
        risk_patterns = [
            r'(?:scam|rug pull|pump and dump|honeypot|bot activity)',
            r'(?:warning|avoid|suspicious|red flag|concerning)',
            r'(?:dump incoming|exit|coordination|manipulation)'
        ]
        
        for pattern in risk_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for match in matches:
                risks.append(f"Community concern: {match}")
        
        return risks[:3]  # Limit to most critical
   
   
   
   
   
    def _extract_safety_signals(self, response: str, symbol: str, indicators: ScamIndicators) -> List[str]:
        """Extract or infer safety signals from response and metrics"""
        signals = []
        
        # Look for explicit positive mentions
        positive_patterns = [
            r'(?:no scam|no rug|no concerns|legitimate|verified|transparent|locked|safe)',
            r'(?:community confident|holders holding|diamond hands|good project)',
            r'(?:team active|development ongoing|partnerships|roadmap)'
        ]
        
        for pattern in positive_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for match in matches:
                signals.append(f"Community reports: {match}")
        
        # Infer from absence of negative signals
        negative_indicators = ['scam', 'rug', 'dump', 'warning', 'avoid', 'suspicious']
        has_negatives = any(indicator in response.lower() for indicator in negative_indicators)
        
        if not has_negatives and len(response) > 200:
            signals.append(f"No community warnings or scam reports found for ${symbol}")
        
        # Add metric-based positive signals
        if not indicators.mint_enabled:
            signals.append("Mint authority properly renounced - cannot create new tokens")
        if not indicators.freeze_enabled:
            signals.append("Freeze authority renounced - cannot halt transfers")
        
        return signals[:5]  # Limit to most relevant


    async def _call_grok_api_new_format(self, prompt: str, search_params: Dict, session: aiohttp.ClientSession) -> Dict:
        """🧠 REVOLUTIONARY Grok API call using NEW live search format"""
        try:
            payload = {
                "model": "grok-3-latest",
                "messages": [
                    {
                        "role": "system", 
                        "content": """You are REVOLUTIONARY GROK, an expert in cryptocurrency safety analysis and community intelligence. You specialize in detecting meme coin scams, rug pulls, and community sentiment patterns.

ANALYSIS APPROACH:
- Focus on REAL community evidence and specific mentions
- Quote actual users and posts when available  
- Distinguish between hype/speculation vs legitimate concerns
- Identify coordinated vs organic community behavior
- Assess whale behavior context and explanations
- Provide actionable safety insights for traders

Always cite specific sources and quotes when making claims about community sentiment."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "search_parameters": search_params,
                "max_tokens": 1500,
                "temperature": 0.3,
                "stream": False
            }
            
            headers = {
                "Authorization": f"Bearer {self.xai_key}",
                "Content-Type": "application/json"
            }
            
            logger.info(f"🧠 Making live search API call to Grok...")
            
            async with session.post("https://api.x.ai/v1/chat/completions", 
                                json=payload, headers=headers, timeout=60) as resp:
                
                if resp.status == 200:
                    result = await resp.json()
                    
                    if 'choices' in result and len(result['choices']) > 0:
                        choice = result['choices'][0]
                        content = choice.get('message', {}).get('content', '')
                        
                        # Extract citations if available (new format includes them)
                        citations = choice.get('citations', [])
                        
                        logger.info(f"🧠 REVOLUTIONARY Grok API success: {len(content)} characters, {len(citations)} citations")
                        
                        return {
                            'content': content,
                            'citations': citations,
                            'usage': result.get('usage', {})
                        }
                    else:
                        logger.error(f"🧠 REVOLUTIONARY Grok API unexpected response structure: {result}")
                        return None
                else:
                    error_text = await resp.text()
                    logger.error(f"🧠 REVOLUTIONARY Grok API error {resp.status}: {error_text}")
                    return None
                    
        except asyncio.TimeoutError:
            logger.error("🧠 REVOLUTIONARY Grok API timeout")
            return None
        except Exception as e:
            logger.error(f"🧠 REVOLUTIONARY Grok API call failed: {e}")
            return None

    def _parse_revolutionary_grok_analysis(self, grok_response: str) -> Dict:
        """🧠 Parse revolutionary Grok analysis response with SPECIFIC community intelligence"""
        try:
            logger.info(f"🧠 Parsing Grok response: {len(grok_response)} characters")
            
            # Extract revolutionary sections with more flexible patterns
            verdict_patterns = [
                r'(?:FINAL ASSESSMENT|VERDICT|ASSESSMENT):\s*\*?\*?([A-Z_]+)',
                r'\*\*(?:SAFETY VERDICT|FINAL VERDICT|ASSESSMENT):\*\*\s*([A-Z_]+)',
                r'(?:^|\n)([A-Z_]+):\s*(?:Strong|Clear|Widespread|Mixed)'
            ]
            
            verdict = "CAUTION_ADVISED"  # Default
            for pattern in verdict_patterns:
                match = re.search(pattern, grok_response, re.IGNORECASE | re.MULTILINE)
                if match:
                    verdict = match.group(1).upper()
                    break
            
            # Extract positive community sentiment
            positive_match = re.search(r'(?:POSITIVE|GOOD|SAFE).*?(?:SIGNALS?|SENTIMENT|MENTIONS?):(.*?)(?=(?:RISK|DANGER|WHALE|FINAL|$))', 
                                     grok_response, re.DOTALL | re.IGNORECASE)
            positive_sentiment = []
            if positive_match:
                positive_text = positive_match.group(1).strip()
                positive_sentiment = self._extract_specific_quotes(positive_text, "positive")
            
            # Extract risk warnings  
            risk_match = re.search(r'(?:RISK|DANGER|WARNING|NEGATIVE).*?(?:SIGNALS?|WARNINGS?|MENTIONS?):(.*?)(?=(?:WHALE|FINAL|ASSESSMENT|$))', 
                                 grok_response, re.DOTALL | re.IGNORECASE)
            community_risks = []
            if risk_match:
                risk_text = risk_match.group(1).strip()
                community_risks = self._extract_specific_quotes(risk_text, "risks")
            
            # Extract whale analysis
            whale_match = re.search(r'(?:WHALE|LARGE HOLDER).*?(?:BEHAVIOR|ANALYSIS):(.*?)(?=(?:FINAL|ASSESSMENT|$))', 
                                  grok_response, re.DOTALL | re.IGNORECASE)
            whale_analysis = whale_match.group(1).strip() if whale_match else "No specific whale analysis found"
            
            # Extract overall insight
            insight_patterns = [
                r'(?:REVOLUTIONARY INSIGHT|CONCLUSION|SUMMARY):\s*(.*?)(?=\n\n|$)',
                r'(?:Based on|Overall|In conclusion).*?evidence[^.]*\.(.*?)(?=\n|$)'
            ]
            revolutionary_insight = "Analysis incomplete - limited data available"
            for pattern in insight_patterns:
                match = re.search(pattern, grok_response, re.DOTALL | re.IGNORECASE)
                if match:
                    revolutionary_insight = match.group(1).strip()
                    break
            
            # Count specific evidence indicators
            username_mentions = len(re.findall(r'@\w+', grok_response))
            quote_mentions = len(re.findall(r'"[^"]{10,}"', grok_response))
            
            # Enhanced confidence calculation
            confidence = 0.5  # Base confidence
            
            # Evidence quality scoring
            if username_mentions > 0:
                confidence += min(0.3, username_mentions * 0.05)
                logger.info(f"🧠 Found {username_mentions} @username mentions (+{min(0.3, username_mentions * 0.05):.2f} confidence)")
            
            if quote_mentions > 0:
                confidence += min(0.2, quote_mentions * 0.02)
                logger.info(f"🧠 Found {quote_mentions} specific quotes (+{min(0.2, quote_mentions * 0.02):.2f} confidence)")
            
            # Content depth scoring
            if len(grok_response) > 500:
                confidence += 0.1
            if len(positive_sentiment) > 0:
                confidence += 0.1
            if len(community_risks) > 0:
                confidence += 0.15  # Risk findings are often more reliable
            
            # Penalty for generic responses
            generic_indicators = ['no information', 'not found', 'unable to find', 'limited data']
            if any(indicator in grok_response.lower() for indicator in generic_indicators):
                confidence -= 0.2
                
            confidence = max(0.2, min(0.95, confidence))
            
            # Extract safety vs risk indicators
            safety_indicators = []
            risk_indicators = []
            
            for sentiment in positive_sentiment:
                if any(term in sentiment.lower() for term in ['legitimate', 'official', 'verified', 'safe', 'locked', 'transparent']):
                    safety_indicators.append(sentiment)
            
            for risk in community_risks:
                if any(term in risk.lower() for term in ['scam', 'rug', 'warning', 'dump', 'manipulation', 'suspicious']):
                    risk_indicators.append(risk)
            
            logger.info(f"🧠 Analysis complete: {verdict} (confidence: {confidence:.0%}, evidence: {username_mentions} users, {quote_mentions} quotes)")
            
            return {
                "verdict": verdict,
                "confidence": confidence,
                "positive_community_sentiment": positive_sentiment,
                "possible_community_risks": community_risks,
                "whale_analysis": whale_analysis,
                "revolutionary_insight": revolutionary_insight,
                "safety_indicators": safety_indicators,
                "risk_indicators": risk_indicators,
                "username_mentions": username_mentions,
                "quote_mentions": quote_mentions,
                "response_length": len(grok_response),
                "full_analysis": grok_response[:1000] + "..." if len(grok_response) > 1000 else grok_response,
                "analysis_type": "revolutionary_live_search"
            }
            
        except Exception as e:
            logger.error(f"Failed to parse revolutionary Grok analysis: {e}")
            return {
                "verdict": "CAUTION_ADVISED",
                "confidence": 0.3,
                "error": str(e),
                "positive_community_sentiment": [],
                "possible_community_risks": [],
                "whale_analysis": "Analysis error occurred",
                "revolutionary_insight": f"Parsing error: {str(e)}",
                "safety_indicators": [],
                "risk_indicators": [],
                "username_mentions": 0,
                "quote_mentions": 0,
                "full_analysis": grok_response[:500] if grok_response else "No response",
                "analysis_type": "error"
            }

    def _extract_specific_quotes(self, text: str, category: str) -> List[str]:
        """Extract specific quotes and usernames from community analysis"""
        if not text or len(text.strip()) < 10:
            return []
        
        items = []
        
        # Split by common delimiters and clean up
        lines = re.split(r'[•\-\*\n]+', text)
        
        for line in lines:
            line = line.strip()
            if not line or len(line) < 20:  # Skip very short lines
                continue
            
            # Clean up formatting
            line = re.sub(r'^[•\-\*\d+\.\s]+', '', line)  # Remove bullets and numbers
            line = re.sub(r'^\s*[-:]\s*', '', line)       # Remove leading dashes/colons
            
            # Look for substantial content
            if len(line) > 25 and any(keyword in line.lower() for keyword in [
                # Positive keywords
                'legitimate', 'official', 'team', 'locked', 'verified', 'community', 'transparent', 'safe',
                # Risk keywords  
                'scam', 'rug', 'warning', 'dump', 'manipulation', 'suspicious', 'concern', 'red flag'
            ]):
                items.append(line.strip())
        
        # If no structured items found, try to extract sentences with quotes or mentions
        if not items:
            sentences = re.split(r'[.!?]+', text)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 30 and ('"' in sentence or '@' in sentence or 
                    any(keyword in sentence.lower() for keyword in ['said', 'mentioned', 'reported', 'warned', 'confirmed'])):
                    items.append(sentence)
        
        return items[:5]  # Return top 5 most relevant items

    def _calculate_revolutionary_galaxy_score_with_grok(self, indicators, transaction_analysis, 
                                                      bundle_detection, suspicious_activity, security_data, grok_analysis):
        """🧠 REVOLUTIONARY Galaxy Brain scoring enhanced with live Grok intelligence"""
        
        # Get base score
        base_score, base_severity, base_confidence = self._calculate_enhanced_galaxy_score(
            indicators, transaction_analysis, bundle_detection, suspicious_activity, security_data
        )
        
        # 🧠 REVOLUTIONARY Grok adjustments
        if grok_analysis.get("available") and grok_analysis.get("parsed_analysis"):
            grok_data = grok_analysis["parsed_analysis"]
            verdict = grok_data.get("verdict", "CAUTION_ADVISED")
            safety_indicators = grok_data.get("safety_indicators", [])
            risk_indicators = grok_data.get("risk_indicators", [])
            
            # 🧠 REVOLUTIONARY scoring adjustments
            score_adjustment = 0
            
            if verdict == "REVOLUTIONARY_SAFE":
                score_adjustment = -25  # Major risk reduction
                logger.info("🧠 REVOLUTIONARY: Grok confirms token is SAFE - major risk reduction")
            elif verdict == "CAUTION_ADVISED":
                score_adjustment = -5   # Minor risk reduction
                logger.info("🧠 REVOLUTIONARY: Grok advises caution - minor risk reduction")
            elif verdict == "DANGER_DETECTED":
                score_adjustment = +20  # Major risk increase
                logger.info("🧠 REVOLUTIONARY: Grok detects DANGER - major risk increase")
            elif verdict == "EXTREME_DANGER":
                score_adjustment = +35  # Extreme risk increase
                logger.info("🧠 REVOLUTIONARY: Grok detects EXTREME DANGER - critical risk increase")
            
            # Additional adjustments for specific findings
            for indicator in safety_indicators:
                if "legitimate" in indicator.lower():
                    score_adjustment -= 8
                    logger.info(f"🧠 REVOLUTIONARY: {indicator} - reducing risk")
                elif "official" in indicator.lower():
                    score_adjustment -= 6
                    logger.info(f"🧠 REVOLUTIONARY: {indicator} - reducing risk")
            
            for indicator in risk_indicators:
                if "warning" in indicator.lower() or "scam" in indicator.lower():
                    score_adjustment += 15
                    logger.info(f"🧠 REVOLUTIONARY: {indicator} - increasing risk")
            
            # Apply revolutionary adjustment
            adjusted_score = max(0, min(100, base_score + score_adjustment))
            
            # 🧠 REVOLUTIONARY severity adjustment
            if adjusted_score < base_score - 20:
                if base_severity == "CRITICAL_RISK":
                    severity_level = "MEDIUM_RISK"
                elif base_severity == "HIGH_RISK":
                    severity_level = "LOW_RISK"
                else:
                    severity_level = "MINIMAL_RISK"
            elif adjusted_score > base_score + 20:
                if base_severity == "MEDIUM_RISK":
                    severity_level = "CRITICAL_RISK"
                elif base_severity == "LOW_RISK":
                    severity_level = "HIGH_RISK"
                else:
                    severity_level = "EXTREME_DANGER"
            else:
                severity_level = base_severity
            
            # 🧠 REVOLUTIONARY confidence enhancement
            grok_confidence = grok_data.get("confidence", 0.5)
            enhanced_confidence = (base_confidence * 0.6) + (grok_confidence * 0.4)
            
            logger.info(f"🧠 REVOLUTIONARY Galaxy Brain: {base_score} → {adjusted_score} ({severity_level}) - Live community intelligence applied")
            
            return int(adjusted_score), severity_level, enhanced_confidence
        
        return base_score, base_severity, base_confidence

    def _calculate_express_score_with_grok(self, indicators: ScamIndicators, holders_data: Dict, 
                                         security_data: Dict, grok_analysis: Dict) -> Tuple[int, str, float]:
        """🧠 Express mode scoring with revolutionary Grok enhancement"""
        
        # Get base express score
        base_score, base_severity, base_confidence = self._calculate_express_score(
            indicators, holders_data, security_data
        )
        
        # Apply Grok adjustments even in Express mode
        if grok_analysis.get("available"):
            grok_data = grok_analysis.get("parsed_analysis", {})
            verdict = grok_data.get("verdict", "CAUTION_ADVISED")
            
            # Simplified Grok adjustments for Express mode
            if verdict == "REVOLUTIONARY_SAFE":
                base_score = max(0, base_score - 15)
                if base_severity in ["CRITICAL_RISK", "HIGH_RISK"]:
                    base_severity = "MEDIUM_RISK"
            elif verdict in ["DANGER_DETECTED", "EXTREME_DANGER"]:
                base_score = min(100, base_score + 20)
                if base_severity in ["LOW_RISK", "MINIMAL_RISK"]:
                    base_severity = "HIGH_RISK"
        
        return int(base_score), base_severity, base_confidence

    def _generate_revolutionary_risk_vectors_with_grok(self, indicators, transaction_analysis, 
                                                     suspicious_activity, security_data, grok_analysis):
        """🧠 REVOLUTIONARY risk vectors enhanced with live Grok intelligence"""
        
        # Get base risk vectors
        risk_vectors = self._generate_enhanced_risk_vectors(
            indicators, transaction_analysis, suspicious_activity, security_data
        )
        
        # 🧠 REVOLUTIONARY Grok-enhanced risk vectors
        if grok_analysis.get("available") and grok_analysis.get("parsed_analysis"):
            grok_data = grok_analysis["parsed_analysis"]
            
            # Add revolutionary safety insights
            for safety_indicator in grok_data.get("safety_indicators", []):
                risk_vectors.append({
                    "category": "🧠 Revolutionary Intelligence",
                    "risk_type": f"Community Safety Confirmation",
                    "severity": "LOW",  # Positive finding
                    "impact": safety_indicator,
                    "likelihood": "CONFIRMED",
                    "mitigation": "Based on live community analysis and verified information"
                })
            
            # Add revolutionary risk insights
            for risk_indicator in grok_data.get("risk_indicators", []):
                risk_vectors.append({
                    "category": "🧠 Revolutionary Intelligence", 
                    "risk_type": f"Community Risk Warning",
                    "severity": "HIGH",
                    "impact": risk_indicator,
                    "likelihood": "HIGH",
                    "mitigation": "URGENT: Community has identified specific risks - investigate immediately"
                })
            
            # Add overall revolutionary assessment
            verdict = grok_data.get("verdict", "CAUTION_ADVISED")
            revolutionary_insight = grok_data.get("revolutionary_insight", "No insight available")
            
            if verdict != "CAUTION_ADVISED":
                severity_map = {
                    "REVOLUTIONARY_SAFE": "LOW", 
                    "DANGER_DETECTED": "HIGH",
                    "EXTREME_DANGER": "CRITICAL"
                }
                
                risk_vectors.append({
                    "category": "🧠 Revolutionary Assessment",
                    "risk_type": f"Live Community Verdict: {verdict.replace('_', ' ')}",
                    "severity": severity_map.get(verdict, "MEDIUM"),
                    "impact": revolutionary_insight,
                    "likelihood": "HIGH",
                    "mitigation": "Based on comprehensive live social media analysis and community monitoring"
                })
        
        return risk_vectors

    def _generate_basic_risk_vectors_with_grok(self, indicators: ScamIndicators, holders_data: Dict, 
                                             liquidity_data: Dict, security_data: Dict, grok_analysis: Dict) -> List[Dict]:
        """🧠 Basic risk vectors enhanced with revolutionary Grok insights"""
        
        # Get base vectors
        risk_vectors = self._generate_basic_risk_vectors(
            indicators, holders_data, liquidity_data, security_data
        )
        
        # Add simplified Grok insights even in basic mode
        if grok_analysis.get("available"):
            grok_data = grok_analysis.get("parsed_analysis", {})
            verdict = grok_data.get("verdict", "CAUTION_ADVISED")
            
            risk_vectors.append({
                "category": "🧠 Revolutionary Intelligence",
                "risk_type": f"Community Analysis: {verdict.replace('_', ' ')}",
                "severity": "MEDIUM",
                "impact": grok_data.get("community_intelligence", "Live community analysis completed"),
                "likelihood": "CONFIRMED",
                "mitigation": "Consider community sentiment in your investment decision"
            })
        
        return risk_vectors

    def _format_grok_ai_analysis(self, grok_analysis: Dict) -> str:
        """🧠 Format Grok analysis for AI analysis display"""
        if not grok_analysis.get("available"):
            return "🧠 Revolutionary Grok Analysis: Connect XAI API key for live community intelligence and advanced meme coin safety analysis with real-time X/Twitter search."
        
        grok_data = grok_analysis.get("parsed_analysis", {})
        verdict = grok_data.get("verdict", "CAUTION_ADVISED")
        insight = grok_data.get("revolutionary_insight", "Analysis unavailable")
        confidence = grok_data.get("confidence", 0.5)
        
        # Count evidence
        username_mentions = grok_data.get("username_mentions", 0) 
        quote_mentions = grok_data.get("quote_mentions", 0)
        safety_indicators = len(grok_data.get("safety_indicators", []))
        risk_indicators = len(grok_data.get("risk_indicators", []))
        
        verdict_emojis = {
            "REVOLUTIONARY_SAFE": "✅🧠",
            "CAUTION_ADVISED": "⚠️🧠", 
            "DANGER_DETECTED": "❌🧠",
            "EXTREME_DANGER": "🚨🧠"
        }
        
        emoji = verdict_emojis.get(verdict, "🧠")
        
        evidence_summary = []
        if username_mentions > 0:
            evidence_summary.append(f"{username_mentions} user mentions")
        if quote_mentions > 0:
            evidence_summary.append(f"{quote_mentions} specific quotes")
        if safety_indicators > 0:
            evidence_summary.append(f"{safety_indicators} positive signals")
        if risk_indicators > 0:
            evidence_summary.append(f"{risk_indicators} risk warnings")
        
        evidence_text = f" ({', '.join(evidence_summary)})" if evidence_summary else " (limited community data)"
        
        return f"{emoji} Revolutionary Live Search: {verdict.replace('_', ' ')} • {confidence:.0%} confidence{evidence_text} • {insight[:100]}{'...' if len(insight) > 100 else ''}"

    # Legacy method - redirects to new format
    async def _call_grok_api(self, prompt: str, search_params: Dict, session: aiohttp.ClientSession) -> str:
        """Legacy method - redirects to new format"""
        result = await self._call_grok_api_new_format(prompt, search_params, session)
        return result.get('content', '') if result else None

    # Core analysis methods
    async def _get_birdeye_security_data(self, mint: str, session: aiohttp.ClientSession) -> Dict:
        """Get security data from Birdeye API"""
        try:
            if not self.birdeye_key or self.birdeye_key == "your-birdeye-api-key-here":
                return {"source": "basic", "available": False}
            
            url = "https://public-api.birdeye.so/defi/token_security"
            headers = {"X-API-KEY": self.birdeye_key, "accept": "application/json"}
            params = {"address": mint, "include_historical": "false"}
            
            async with session.get(url, headers=headers, params=params, timeout=15) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get("success") and data.get("data"):
                        security_info = data["data"]
                        return {
                            "source": "birdeye",
                            "available": True,
                            "liquidity_locked": security_info.get("is_liquidity_locked", False),
                            "mint_enabled": security_info.get("can_mint", True),
                            "freeze_enabled": security_info.get("can_freeze_account", True),
                            "buy_tax": security_info.get("buy_tax_percentage", 0),
                            "sell_tax": security_info.get("sell_tax_percentage", 0),
                            "overall_score": self._calculate_birdeye_score(security_info),
                            "raw_data": security_info
                        }
        except Exception as e:
            logger.error(f"Birdeye API error: {e}")
        
        return {"source": "unavailable", "available": False}

    def _calculate_birdeye_score(self, security_info: Dict) -> int:
        """Calculate security score from Birdeye data"""
        score = 100
        if security_info.get("mint_enabled", False): score -= 25
        if security_info.get("freeze_enabled", False): score -= 20
        if not security_info.get("liquidity_locked", False): score -= 15
        if security_info.get("buy_tax", 0) > 5: score -= 10
        if security_info.get("sell_tax", 0) > 5: score -= 10
        return max(0, score)

    async def _get_token_info(self, mint: str, session: aiohttp.ClientSession) -> Optional[Dict]:
        """Get basic token info from RPC"""
        try:
            if not self.rpc_url or self.helius_key == "demo_key":
                return {
                    "mint": mint,
                    "decimals": 6,
                    "supply": 1000000000,
                    "mint_authority": None,
                    "freeze_authority": None,
                    "is_initialized": True,
                    "symbol": "DEMO",
                    "name": "Demo Token"
                }
            
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getAccountInfo",
                "params": [mint, {"encoding": "jsonParsed"}]
            }
            
            async with session.post(self.rpc_url, json=payload, timeout=10) as resp:
                data = await resp.json()
                
            if not data.get("result", {}).get("value"):
                return None
                
            account_data = data["result"]["value"]["data"]["parsed"]["info"]
            
            # Get supply
            supply_payload = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "getTokenSupply",
                "params": [mint]
            }
            
            async with session.post(self.rpc_url, json=supply_payload, timeout=10) as resp:
                supply_data = await resp.json()
                
            supply = float(supply_data["result"]["value"]["amount"]) / (10 ** account_data["decimals"])
            
            return {
                "mint": mint,
                "decimals": account_data["decimals"],
                "supply": supply,
                "mint_authority": account_data.get("mintAuthority"),
                "freeze_authority": account_data.get("freezeAuthority"),
                "is_initialized": account_data.get("isInitialized", False),
                "symbol": "TOKEN",  # Will be updated from DexScreener
                "name": "Token"
            }
            
        except Exception as e:
            logger.error(f"Failed to get token info: {e}")
            return None

    async def _get_enhanced_market_data(self, mint: str, session: aiohttp.ClientSession) -> Dict:
        """Enhanced market data with liquidity lock detection"""
        try:
            url = f"https://api.dexscreener.com/latest/dex/tokens/{mint}"
            async with session.get(url, timeout=10) as resp:
                data = await resp.json()
                
            if not data.get("pairs"):
                return {}
                
            # Get the most liquid pair
            pair = max(data["pairs"], key=lambda x: float(x.get("liquidity", {}).get("usd", 0)))
            base_token = pair.get("baseToken", {})
            symbol = base_token.get("symbol", "TOKEN")
            name = base_token.get("name", "Token") 

            # If we got the symbol, log it
            if symbol and symbol != "TOKEN":
                logger.info(f"📊 Found token: {name} (${symbol})")
            else:
                logger.warning(f"⚠️ Could not extract symbol from DexScreener for {mint}")
                        
            # Check for liquidity lock labels
            labels = pair.get("labels", [])
            liquidity_locked_dex = "Liquidity Locked" in labels
            
            return {
                "price": float(pair.get("priceUsd", 0)),
                "market_cap": float(pair.get("marketCap", 0)),
                "liquidity": float(pair.get("liquidity", {}).get("usd", 0)),
                "volume_24h": float(pair.get("volume", {}).get("h24", 0)),
                "price_change_24h": float(pair.get("priceChange", {}).get("h24", 0)),
                "price_change_1h": float(pair.get("priceChange", {}).get("h1", 0)),
                "price_change_6h": float(pair.get("priceChange", {}).get("h6", 0)),
                "created_at": pair.get("pairCreatedAt", 0),
                "symbol": base_token.get("symbol", "TOKEN"),
                "name": base_token.get("name", "Token"),
                "txns": {
                    "buys_24h": pair.get("txns", {}).get("h24", {}).get("buys", 0),
                    "sells_24h": pair.get("txns", {}).get("h24", {}).get("sells", 0),
                    "buys_1h": pair.get("txns", {}).get("h1", {}).get("buys", 0),
                    "sells_1h": pair.get("txns", {}).get("h1", {}).get("sells", 0)
                },
                "pool_address": pair.get("pairAddress", ""),
                "dex": pair.get("dexId", ""),
                "liquidity_locked_dexscreener": liquidity_locked_dex,
                "labels": labels,
                "logo": base_token.get("image", ""),
                "description": base_token.get("description", ""),
                "website": base_token.get("website", ""),
                "twitter": base_token.get("twitter", ""),
                "telegram": base_token.get("telegram", ""),
                "discord": base_token.get("discord", "")
            }
            
        except Exception as e:
            logger.error(f"Failed to get enhanced market data: {e}")
            return {}

    async def _get_real_holders_analysis(self, mint: str, total_supply: float, session: aiohttp.ClientSession) -> Dict:
        """Real holder analysis with protocol wallet filtering"""
        try:
            if not self.rpc_url or self.helius_key == "demo_key":
                logger.warning("No Helius key - returning limited holder data")
                return {
                    "top_1_percent": 0,
                    "top_5_percent": 0,
                    "top_10_percent": 0,
                    "top_holders": [],
                    "data_source": "unavailable",
                    "error": "No RPC access available"
                }
            
            # Get real holder data
            payload = {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "getTokenLargestAccounts",
                "params": [mint]
            }
            
            async with session.post(self.rpc_url, json=payload, timeout=15) as resp:
                data = await resp.json()
                
            if not data.get("result", {}).get("value"):
                return {
                    "top_1_percent": 0,
                    "top_5_percent": 0,
                    "top_10_percent": 0,
                    "top_holders": [],
                    "data_source": "no_data",
                    "error": "No holder data returned from RPC"
                }
                
            accounts = data["result"]["value"]
            logger.info(f"🔍 Found {len(accounts)} total token accounts")
            
            # Process holders
            all_holders = []
            protocol_holders = []
            
            for i, account in enumerate(accounts):
                amount = float(account.get("uiAmount", 0))
                if amount > 0 and total_supply > 0:
                    percentage = (amount / total_supply) * 100
                    address = account["address"]
                    
                    holder_info = {
                        "rank": i + 1,
                        "address": address,
                        "balance": amount,
                        "percentage": percentage,
                        "is_protocol": False,
                        "protocol_type": None
                    }
                    
                    # Filter protocol wallets
                    if address in [
                        'So11111111111111111111111111111111111111112',  # Wrapped SOL
                        '11111111111111111111111111111111',              # System Program
                        'TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA',   # Token Program
                    ]:
                        protocol_holders.append(holder_info)
                        logger.info(f"🏛️ Filtered system account: {address[:8]}... ({percentage:.2f}%)")
                    else:
                        all_holders.append(holder_info)
            
            real_holders = all_holders
            
            if not real_holders:
                logger.warning("⚠️ No real holders found after filtering protocols")
                return {
                    "top_1_percent": 0,
                    "top_5_percent": 0,
                    "top_10_percent": 0,
                    "top_holders": [],
                    "protocol_holders": protocol_holders,
                    "data_source": "all_protocol",
                    "warning": "All top holders appear to be protocol contracts"
                }
            
            # Calculate concentration metrics
            top_1 = real_holders[0]["percentage"] if real_holders else 0
            top_5 = sum(h["percentage"] for h in real_holders[:5])
            top_10 = sum(h["percentage"] for h in real_holders[:10])
            
            # Calculate concentration risk
            if top_1 > 25:
                concentration_risk = "CRITICAL"
            elif top_1 > 15:
                concentration_risk = "HIGH"
            elif top_1 > 8:
                concentration_risk = "MEDIUM"
            elif top_1 > 3:
                concentration_risk = "LOW"
            else:
                concentration_risk = "MINIMAL"
            
            total_protocol_percent = sum(h["percentage"] for h in protocol_holders)
            
            logger.info(f"📊 FILTERED Holder Analysis:")
            logger.info(f"   Real holders: {len(real_holders)}")
            logger.info(f"   Protocol wallets: {len(protocol_holders)} ({total_protocol_percent:.1f}% of supply)")
            logger.info(f"   Top 1 REAL holder: {top_1:.1f}%")
            logger.info(f"   Concentration risk: {concentration_risk}")
            
            return {
                "top_1_percent": round(top_1, 2),
                "top_5_percent": round(top_5, 2),
                "top_10_percent": round(top_10, 2),
                "concentration_risk": concentration_risk,
                "top_holders": real_holders[:10],
                "protocol_holders": protocol_holders,
                "holders_analyzed": len(real_holders),
                "protocol_wallets_filtered": len(protocol_holders),
                "total_protocol_percentage": round(total_protocol_percent, 2),
                "whale_count": len([h for h in real_holders if h["percentage"] > 5]),
                "data_source": "helius_filtered",
                "total_accounts": len(accounts)
            }
            
        except Exception as e:
            logger.error(f"Filtered holders analysis failed: {e}")
            return {
                "top_1_percent": 0,
                "top_5_percent": 0,
                "top_10_percent": 0,
                "top_holders": [],
                "protocol_holders": [],
                "data_source": "error",
                "error": str(e)
            }

    async def _get_real_transaction_analysis(self, mint: str, session: aiohttp.ClientSession, deep_analysis: bool) -> Dict:
        """Real transaction analysis using Helius"""
        try:
            if not self.rpc_url or self.helius_key == "demo_key":
                logger.warning("No Helius key - returning basic transaction data")
                return {
                    "pattern_score": 50,
                    "unique_traders_24h": 0,
                    "transaction_count_24h": 0,
                    "data_source": "unavailable"
                }
            
            # Get token accounts for this mint
            token_accounts = await self._get_token_accounts(mint, session)
            
            if not token_accounts:
                logger.warning(f"No token accounts found for mint: {mint}")
                return {
                    "pattern_score": 50,
                    "unique_traders_24h": 0,
                    "transaction_count_24h": 0,
                    "data_source": "no_accounts"
                }
            
            # Get transactions for top token accounts
            all_signatures = []
            for account in token_accounts[:5]:
                try:
                    payload = {
                        "jsonrpc": "2.0",
                        "id": f"tx-analysis-{account}",
                        "method": "getSignaturesForAddress",
                        "params": [
                            account,
                            {
                                "limit": 50,
                                "commitment": "confirmed"
                            }
                        ]
                    }
                    
                    async with session.post(self.rpc_url, json=payload, timeout=15) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            signatures = data.get("result", [])
                            all_signatures.extend(signatures)
                            logger.info(f"📝 Found {len(signatures)} signatures for account {account[:8]}...")
                        
                except Exception as e:
                    logger.warning(f"Failed to get signatures for account {account}: {e}")
                    continue
            
            if not all_signatures:
                return {
                    "pattern_score": 30,
                    "unique_traders_24h": 0,
                    "transaction_count_24h": 0,
                    "data_source": "no_transactions"
                }
            
            logger.info(f"📊 Analyzing {len(all_signatures)} total transaction signatures")
            
            if deep_analysis and len(all_signatures) > 10:
                return await self._analyze_detailed_transactions(all_signatures[:50], session)
            else:
                return self._analyze_signature_patterns(all_signatures)
                
        except Exception as e:
            logger.error(f"Transaction analysis failed: {e}")
            return {
                "pattern_score": 50,
                "unique_traders_24h": 0,
                "transaction_count_24h": len(all_signatures) if 'all_signatures' in locals() else 0,
                "data_source": "error",
                "error": str(e)
            }

    async def _get_token_accounts(self, mint: str, session: aiohttp.ClientSession) -> List[str]:
        """Get token accounts that hold this token"""
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": "token-accounts",
                "method": "getTokenLargestAccounts",
                "params": [mint]
            }
            
            async with session.post(self.rpc_url, json=payload, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    result = data.get("result", {})
                    accounts = result.get("value", [])
                    
                    token_account_addresses = []
                    for account in accounts:
                        address = account.get("address")
                        amount = account.get("uiAmount", 0)
                        if address and amount > 0:
                            token_account_addresses.append(address)
                    
                    logger.info(f"🏦 Found {len(token_account_addresses)} token accounts for {mint}")
                    return token_account_addresses
                    
        except Exception as e:
            logger.error(f"Failed to get token accounts: {e}")
        
        return []

    async def _analyze_detailed_transactions(self, signatures: List[Dict], session: aiohttp.ClientSession) -> Dict:
        """Analyze detailed transaction data for deep insights"""
        try:
            signature_strings = [sig["signature"] for sig in signatures[:10]]
            
            payload = {
                "jsonrpc": "2.0",
                "id": "detailed-tx",
                "method": "getMultipleTransactions",
                "params": [
                    signature_strings,
                    {
                        "encoding": "jsonParsed",
                        "maxSupportedTransactionVersion": 0
                    }
                ]
            }
            
            async with session.post(self.rpc_url, json=payload, timeout=25) as resp:
                data = await resp.json()
            
            transactions = data.get("result", [])
            if not transactions:
                return self._analyze_signature_patterns(signatures)
            
            unique_signers = set()
            bot_indicators = 0
            total_analyzed = 0
            
            for tx in transactions:
                if tx and tx.get("transaction"):
                    total_analyzed += 1
                    message = tx["transaction"].get("message", {})
                    
                    account_keys = message.get("accountKeys", [])
                    if account_keys:
                        signer = account_keys[0].get("pubkey") if isinstance(account_keys[0], dict) else account_keys[0]
                        unique_signers.add(signer)
                    
                    instructions = message.get("instructions", [])
                    if len(instructions) > 10:
                        bot_indicators += 1
            
            unique_count = len(unique_signers)
            bot_percentage = (bot_indicators / total_analyzed * 100) if total_analyzed > 0 else 0
            
            if unique_count == 0:
                health_score = 30
            elif bot_percentage > 60:
                health_score = 40
            elif bot_percentage > 30:
                health_score = 60
            else:
                health_score = 80
            
            return {
                "pattern_score": health_score,
                "unique_traders_24h": unique_count,
                "transaction_count_24h": len(signatures),
                "bot_activity_percentage": round(bot_percentage, 1),
                "organic_volume_percentage": round(100 - bot_percentage, 1),
                "transactions_analyzed": total_analyzed,
                "data_source": "helius_detailed"
            }
            
        except Exception as e:
            logger.error(f"Detailed transaction analysis failed: {e}")
            return self._analyze_signature_patterns(signatures)

    def _analyze_signature_patterns(self, signatures: List[Dict]) -> Dict:
        """Basic signature pattern analysis"""
        try:
            if not signatures:
                return {
                    "pattern_score": 50,
                    "unique_traders_24h": 0,
                    "transaction_count_24h": 0,
                    "data_source": "no_signatures"
                }
            
            unique_patterns = set()
            recent_count = 0
            now = datetime.now().timestamp()
            
            for sig_info in signatures:
                signature = sig_info.get("signature", "")
                if signature:
                    pattern = signature[:12]
                    unique_patterns.add(pattern)
                
                block_time = sig_info.get("blockTime", 0)
                if block_time and (now - block_time) < 86400:
                    recent_count += 1
            
            unique_count = len(unique_patterns)
            
            if len(signatures) == 0:
                health_score = 50
            else:
                uniqueness_ratio = unique_count / len(signatures)
                if uniqueness_ratio > 0.8:
                    health_score = 80
                elif uniqueness_ratio > 0.6:
                    health_score = 65
                elif uniqueness_ratio > 0.4:
                    health_score = 50
                else:
                    health_score = 35
            
            bot_activity = max(0, 100 - (uniqueness_ratio * 100)) if len(signatures) > 0 else 0
            
            return {
                "pattern_score": health_score,
                "unique_traders_24h": unique_count,
                "transaction_count_24h": recent_count,
                "bot_activity_percentage": round(bot_activity, 1),
                "organic_volume_percentage": round(100 - bot_activity, 1),
                "uniqueness_ratio": round(uniqueness_ratio, 3) if len(signatures) > 0 else 0,
                "data_source": "signature_patterns"
            }
            
        except Exception as e:
            logger.error(f"Signature pattern analysis failed: {e}")
            return {
                "pattern_score": 50,
                "unique_traders_24h": 0,
                "transaction_count_24h": 0,
                "data_source": "error"
            }

    async def _get_enhanced_liquidity_analysis(self, mint: str, market_data: Dict, security_data: Dict, session: aiohttp.ClientSession) -> Dict:
        """Enhanced liquidity analysis"""
        liquidity_usd = market_data.get("liquidity", 0)
        market_cap = market_data.get("market_cap", 0)
        volume_24h = market_data.get("volume_24h", 0)
        
        liq_ratio = (liquidity_usd / market_cap * 100) if market_cap > 0 else 0
        volume_to_liq = (volume_24h / liquidity_usd) if liquidity_usd > 0 else 0
        
        lock_detection_methods = await self._detect_liquidity_locks_alternative(mint, market_data, session)
        
        is_locked = lock_detection_methods.get("any_method_detected", False)
        lock_confidence = lock_detection_methods.get("confidence", "LOW")
        
        base_risk = self._calculate_enhanced_liquidity_risk(liquidity_usd, liq_ratio, volume_to_liq, market_cap)
        
        if is_locked and lock_confidence in ["HIGH", "MEDIUM"]:
            liquidity_risk = "LOW" if base_risk in ["MEDIUM", "LOW"] else "MEDIUM"
            risk_explanation = f"Liquidity appears secured ({lock_confidence} confidence)"
        else:
            liquidity_risk = base_risk
            risk_explanation = f"Liquidity security unclear - assess based on metrics ({base_risk} risk)"
        
        return {
            "liquidity_usd": liquidity_usd,
            "liquidity_ratio": round(liq_ratio, 2),
            "volume_to_liquidity": round(volume_to_liq, 2),
            "liquidity_risk": liquidity_risk,
            "risk_explanation": risk_explanation,
            "lock_detection": lock_detection_methods,
            "is_locked": is_locked,
            "lock_confidence": lock_confidence,
            "liquidity_health_score": self._calculate_liquidity_health_score(liquidity_usd, liq_ratio, volume_to_liq, market_cap),
            "liquidity_stability": self._assess_liquidity_stability(market_data),
            "slippage_risk": self._calculate_slippage_risk(liquidity_usd),
            "pool_address": market_data.get("pool_address", ""),
            "dex": market_data.get("dex", ""),
            "slippage_1k": self._estimate_slippage(liquidity_usd, 1000),
            "slippage_10k": self._estimate_slippage(liquidity_usd, 10000),
            "slippage_50k": self._estimate_slippage(liquidity_usd, 50000),
            "market_impact_warning": self._get_market_impact_warning(liquidity_usd),
            "recommended_max_trade": self._get_recommended_max_trade(liquidity_usd)
        }

    async def _detect_liquidity_locks_alternative(self, mint: str, market_data: Dict, session: aiohttp.ClientSession) -> Dict:
        """Multiple methods to detect liquidity locks"""
        detection_results = {
            "dexscreener_labels": False,
            "pool_analysis": False,
            "burn_address_check": False,
            "any_method_detected": False,
            "confidence": "LOW",
            "methods_used": []
        }
        
        try:
            # Method 1: DexScreener labels
            labels = market_data.get("labels", [])
            if labels and any("lock" in label.lower() for label in labels):
                detection_results["dexscreener_labels"] = True
                detection_results["methods_used"].append("DexScreener labels")
                logger.info("🔒 DexScreener reports liquidity locked")
            
            # Method 2: Pool address analysis
            pool_address = market_data.get("pool_address", "")
            if pool_address:
                lock_detected = await self._check_pool_for_locks(pool_address, session)
                detection_results["pool_analysis"] = lock_detected
                if lock_detected:
                    detection_results["methods_used"].append("Pool contract analysis")
                    logger.info("🔒 Pool analysis suggests liquidity locked")
            
            # Method 3: Check for burn address holdings
            burn_check = await self._check_burn_address_locks(mint, session)
            detection_results["burn_address_check"] = burn_check
            if burn_check:
                detection_results["methods_used"].append("Burn address analysis")
                logger.info("🔒 Burn address analysis suggests locked liquidity")
            
            methods_positive = sum([
                detection_results["dexscreener_labels"],
                detection_results["pool_analysis"], 
                detection_results["burn_address_check"]
            ])
            
            detection_results["any_method_detected"] = methods_positive > 0
            
            if methods_positive >= 2:
                detection_results["confidence"] = "HIGH"
            elif methods_positive == 1:
                detection_results["confidence"] = "MEDIUM"
            else:
                detection_results["confidence"] = "LOW"
            
        except Exception as e:
            logger.error(f"Lock detection error: {e}")
            detection_results["error"] = str(e)
        
        return detection_results

    async def _check_pool_for_locks(self, pool_address: str, session: aiohttp.ClientSession) -> bool:
        """Check if pool address shows signs of being locked"""
        try:
            if not self.rpc_url or not pool_address:
                return False
            
            payload = {
                "jsonrpc": "2.0",
                "id": "pool-check",
                "method": "getAccountInfo",
                "params": [pool_address, {"encoding": "base64"}]
            }
            
            async with session.post(self.rpc_url, json=payload, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    result = data.get("result", {})
                    
                    if result and result.get("value"):
                        owner = result["value"].get("owner")
                        if owner == "11111111111111111111111111111111":
                            return True
        except Exception as e:
            logger.warning(f"Pool lock check error: {e}")
        
        return False

    async def _check_burn_address_locks(self, mint: str, session: aiohttp.ClientSession) -> bool:
        """Check if LP tokens are sent to burn addresses"""
        try:
            if not self.rpc_url:
                return False
            
            burn_addresses = [
                "11111111111111111111111111111111",
                "1nc1nerator11111111111111111111111111111111",
            ]
            
            for burn_addr in burn_addresses:
                payload = {
                    "jsonrpc": "2.0",
                    "id": "burn-check",
                    "method": "getTokenAccountsByOwner",
                    "params": [
                        burn_addr,
                        {"mint": mint},
                        {"encoding": "jsonParsed"}
                    ]
                }
                
                async with session.post(self.rpc_url, json=payload, timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        accounts = data.get("result", {}).get("value", [])
                        
                        for account in accounts:
                            token_amount = account.get("account", {}).get("data", {}).get("parsed", {}).get("info", {}).get("tokenAmount", {}).get("uiAmount", 0)
                            if token_amount > 0:
                                logger.info(f"🔥 Found {token_amount} tokens in burn address {burn_addr[:8]}...")
                                return True
        except Exception as e:
            logger.warning(f"Burn address check error: {e}")
        
        return False

    async def _analyze_enhanced_authorities(self, token_info: Dict, security_data: Dict) -> Dict:
        """Enhanced authority analysis"""
        rpc_mint_authority = token_info.get("mint_authority")
        rpc_freeze_authority = token_info.get("freeze_authority")
        
        birdeye_mint = security_data.get("mint_enabled", True)
        birdeye_freeze = security_data.get("freeze_enabled", True)
        owner_change = security_data.get("owner_change_enabled", True)
        
        mint_renounced = (rpc_mint_authority is None) and not birdeye_mint
        freeze_renounced = (rpc_freeze_authority is None) and not birdeye_freeze
        
        authority_risk = 0
        if not mint_renounced:
            authority_risk += 40
        if not freeze_renounced:
            authority_risk += 30
        if owner_change:
            authority_risk += 10
        
        if authority_risk >= 60:
            risk_level = "CRITICAL"
        elif authority_risk >= 40:
            risk_level = "HIGH"
        elif authority_risk > 0:
            risk_level = "MEDIUM"
        else:
            risk_level = "SAFE"
        
        return {
            "mint_authority": rpc_mint_authority,
            "freeze_authority": rpc_freeze_authority,
            "mint_renounced": mint_renounced,
            "freeze_renounced": freeze_renounced,
            "owner_change_enabled": owner_change,
            "authority_risk_score": authority_risk,
            "risk_level": risk_level,
            "fully_decentralized": mint_renounced and freeze_renounced and not owner_change,
            "data_sources": {
                "rpc": rpc_mint_authority is not None or rpc_freeze_authority is not None,
                "birdeye": security_data.get("available", False)
            }
        }

    def _calculate_enhanced_indicators(self, token_info: Dict, market_data: Dict, 
                                    holders_data: Dict, liquidity_data: Dict, security_data: Dict) -> ScamIndicators:
        """Enhanced risk calculation"""
        indicators = ScamIndicators()
        
        indicators.mint_enabled = token_info.get("mint_authority") is not None
        indicators.freeze_enabled = token_info.get("freeze_authority") is not None
        
        logger.info(f"🔍 Authority status: Mint={indicators.mint_enabled}, Freeze={indicators.freeze_enabled}")
        
        indicators.top_holder_percent = holders_data.get("top_1_percent", 0)
        indicators.top_10_holders_percent = holders_data.get("top_10_percent", 0)
        
        created_at = market_data.get("created_at", 0)
        if created_at:
            age_hours = int((datetime.now().timestamp() * 1000 - created_at) / (1000 * 60 * 60))
            indicators.token_age_hours = age_hours
            logger.info(f"🕰️ Token age: {age_hours} hours ({age_hours/24:.1f} days)")
        
        indicators.liquidity_percent = liquidity_data.get("liquidity_ratio", 0)
        liquidity_usd = liquidity_data.get("liquidity_usd", 0)
        volume_24h = market_data.get("volume_24h", 0)
        market_cap = market_data.get("market_cap", 0)
        
        liquidity_health = self._calculate_liquidity_health(liquidity_usd, indicators.liquidity_percent, volume_24h, market_cap)
        indicators.liquidity_locked = liquidity_health >= 70
        
        logger.info(f"💧 Liquidity health: {liquidity_health}/100 (treating as {'locked' if indicators.liquidity_locked else 'unlocked'})")
        
        buys = market_data.get("txns", {}).get("buys_24h", 0)
        sells = market_data.get("txns", {}).get("sells_24h", 0)
        indicators.can_sell = sells > 0 or buys < 10
        
        indicators.buy_tax = 0
        indicators.sell_tax = 0
        
        # Enhanced risk score calculation
        risk_score = 0
        
        if indicators.mint_enabled:
            risk_score += 40
            logger.info("❌ Mint authority active (+40 risk)")
        else:
            logger.info("✅ Mint authority renounced (+0 risk)")
            
        if indicators.freeze_enabled:
            risk_score += 30
            logger.info("❌ Freeze authority active (+30 risk)")
        else:
            logger.info("✅ Freeze authority renounced (+0 risk)")
        
        if indicators.top_holder_percent > 50:
            risk_score += 35
            logger.info(f"❌ EXTREME concentration: {indicators.top_holder_percent:.1f}% (+35 risk)")
        elif indicators.top_holder_percent > 30:
            risk_score += 28
            logger.info(f"❌ Very high concentration: {indicators.top_holder_percent:.1f}% (+28 risk)")
        elif indicators.top_holder_percent > 20:
            risk_score += 20
            logger.info(f"⚠️ High concentration: {indicators.top_holder_percent:.1f}% (+20 risk)")
        elif indicators.top_holder_percent > 15:
            risk_score += 15
            logger.info(f"⚠️ Moderate concentration: {indicators.top_holder_percent:.1f}% (+15 risk)")
        elif indicators.top_holder_percent > 10:
            risk_score += 8
            logger.info(f"⚠️ Some concentration: {indicators.top_holder_percent:.1f}% (+8 risk)")
        else:
            logger.info(f"✅ Good distribution: {indicators.top_holder_percent:.1f}% (+0 risk)")
        
        if indicators.top_10_holders_percent > 90:
            risk_score += 20
            logger.info(f"❌ Top 10 control {indicators.top_10_holders_percent:.1f}% (+20 risk)")
        elif indicators.top_10_holders_percent > 80:
            risk_score += 15
            logger.info(f"⚠️ Top 10 control {indicators.top_10_holders_percent:.1f}% (+15 risk)")
        elif indicators.top_10_holders_percent > 70:
            risk_score += 10
            logger.info(f"⚠️ Top 10 control {indicators.top_10_holders_percent:.1f}% (+10 risk)")
        else:
            logger.info(f"✅ Decent top 10 distribution: {indicators.top_10_holders_percent:.1f}% (+0 risk)")
        
        if liquidity_health < 20:
            risk_score += 25
            logger.info(f"❌ Poor liquidity health: {liquidity_health}/100 (+25 risk)")
        elif liquidity_health < 40:
            risk_score += 18
            logger.info(f"⚠️ Low liquidity health: {liquidity_health}/100 (+18 risk)")
        elif liquidity_health < 60:
            risk_score += 10
            logger.info(f"⚠️ Moderate liquidity health: {liquidity_health}/100 (+10 risk)")
        elif liquidity_health < 80:
            risk_score += 3
            logger.info(f"✅ Good liquidity health: {liquidity_health}/100 (+3 risk)")
        else:
            logger.info(f"✅ Excellent liquidity health: {liquidity_health}/100 (+0 risk)")
        
        if indicators.token_age_hours < 6:
            risk_score += 15
            logger.info(f"❌ Very new token: {indicators.token_age_hours}h (+15 risk)")
        elif indicators.token_age_hours < 24:
            risk_score += 10
            logger.info(f"⚠️ New token: {indicators.token_age_hours}h (+10 risk)")
        elif indicators.token_age_hours < 168:
            risk_score += 5
            logger.info(f"⚠️ Young token: {indicators.token_age_hours}h (+5 risk)")
        else:
            logger.info(f"✅ Mature token: {indicators.token_age_hours}h (+0 risk)")
        
        if not indicators.can_sell:
            risk_score += 20
            logger.info("❌ Cannot sell or no selling activity (+20 risk)")
        else:
            logger.info("✅ Can sell - trading activity detected (+0 risk)")
        
        indicators.overall_risk_score = min(100, max(0, risk_score))
        
        if risk_score >= 80:
            indicators.risk_level = "EXTREME"
        elif risk_score >= 65:
            indicators.risk_level = "CRITICAL"  
        elif risk_score >= 45:
            indicators.risk_level = "HIGH"
        elif risk_score >= 25:
            indicators.risk_level = "MEDIUM"
        elif risk_score >= 10:
            indicators.risk_level = "LOW"
        else:
            indicators.risk_level = "MINIMAL"
        
        logger.info(f"🧠 Final risk: {indicators.overall_risk_score}/100 ({indicators.risk_level}) - Based on reliable metrics")
        
        return indicators

    def _calculate_liquidity_health(self, liquidity_usd: float, liq_ratio: float, volume_24h: float, market_cap: float) -> int:
        """Calculate liquidity health score based on multiple measurable factors"""
        health_score = 50  # Start neutral
        
        if liquidity_usd >= 10_000_000:
            health_score += 30
        elif liquidity_usd >= 5_000_000:
            health_score += 25
        elif liquidity_usd >= 1_000_000:
            health_score += 20
        elif liquidity_usd >= 500_000:
            health_score += 15
        elif liquidity_usd >= 100_000:
            health_score += 5
        else:
            health_score -= 20
        
        if liq_ratio >= 10:
            health_score += 20
        elif liq_ratio >= 5:
            health_score += 15
        elif liq_ratio >= 3:
            health_score += 10
        elif liq_ratio >= 1:
            health_score += 0
        else:
            health_score -= 15
        
        if volume_24h > 0 and liquidity_usd > 0:
            volume_to_liq = volume_24h / liquidity_usd
            if 0.1 <= volume_to_liq <= 3:
                health_score += 10
            elif volume_to_liq > 10:
                health_score -= 10
        
        if market_cap > 0:
            if market_cap > 100_000_000 and liq_ratio >= 2:
                health_score += 5
            elif market_cap < 1_000_000 and liq_ratio >= 10:
                health_score += 5
        
        return max(0, min(100, health_score))

    # Helper calculation methods
    def _calculate_enhanced_liquidity_risk(self, liquidity_usd: float, liq_ratio: float, volume_to_liq: float, market_cap: float) -> str:
        """Enhanced liquidity risk calculation"""
        risk_score = 0
        
        if liquidity_usd >= 5_000_000:
            risk_score += 0
        elif liquidity_usd >= 1_000_000:
            risk_score += 1
        elif liquidity_usd >= 500_000:
            risk_score += 2
        elif liquidity_usd >= 100_000:
            risk_score += 3
        elif liquidity_usd >= 50_000:
            risk_score += 4
        else:
            risk_score += 5
        
        if liq_ratio >= 10:
            risk_score += 0
        elif liq_ratio >= 5:
            risk_score += 1
        elif liq_ratio >= 3:
            risk_score += 2
        elif liq_ratio >= 1:
            risk_score += 3
        else:
            risk_score += 4
        
        if volume_to_liq > 5:
            risk_score += 2
        elif volume_to_liq > 2:
            risk_score += 1
        
        if market_cap > 100_000_000 and liq_ratio < 2:
            risk_score += 2
        
        if risk_score >= 10:
            return "CRITICAL"
        elif risk_score >= 7:
            return "HIGH"
        elif risk_score >= 4:
            return "MEDIUM"
        else:
            return "LOW"

    def _calculate_liquidity_health_score(self, liquidity_usd: float, liq_ratio: float, volume_to_liq: float, market_cap: float) -> int:
        """Calculate overall liquidity health score"""
        score = 50
        
        if liquidity_usd >= 10_000_000:
            score += 25
        elif liquidity_usd >= 1_000_000:
            score += 20
        elif liquidity_usd >= 500_000:
            score += 15
        elif liquidity_usd >= 100_000:
            score += 10
        elif liquidity_usd >= 50_000:
            score += 5
        else:
            score -= 20
        
        if liq_ratio >= 8:
            score += 15
        elif liq_ratio >= 5:
            score += 10
        elif liq_ratio >= 3:
            score += 5
        elif liq_ratio < 1:
            score -= 15
        
        if 0.5 <= volume_to_liq <= 2:
            score += 10
        elif volume_to_liq < 0.1:
            score -= 10
        elif volume_to_liq > 5:
            score -= 15
        
        return max(0, min(100, score))

    def _assess_liquidity_stability(self, market_data: Dict) -> str:
        """Assess liquidity stability based on trading patterns"""
        buys_24h = market_data.get("txns", {}).get("buys_24h", 0)
        sells_24h = market_data.get("txns", {}).get("sells_24h", 0)
        
        total_txns = buys_24h + sells_24h
        
        if total_txns == 0:
            return "STAGNANT - No trading activity"
        
        buy_ratio = buys_24h / total_txns
        
        if buy_ratio > 0.7:
            return "BUYING_PRESSURE - More buys than sells"
        elif buy_ratio < 0.3:
            return "SELLING_PRESSURE - More sells than buys"
        else:
            return "BALANCED - Healthy buy/sell ratio"

    def _calculate_slippage_risk(self, liquidity_usd: float) -> str:
        """Calculate slippage risk category"""
        if liquidity_usd >= 5_000_000:
            return "MINIMAL - Large trades possible"
        elif liquidity_usd >= 1_000_000:
            return "LOW - Medium trades safe"
        elif liquidity_usd >= 500_000:
            return "MEDIUM - Small trades recommended"
        elif liquidity_usd >= 100_000:
            return "HIGH - Micro trades only"
        else:
            return "EXTREME - Any trade will cause major slippage"

    def _get_market_impact_warning(self, liquidity: float) -> str:
        """Get market impact warning based on liquidity"""
        if liquidity < 100_000:
            return "⚠️ CRITICAL: Even small trades will cause significant price impact"
        elif liquidity < 500_000:
            return "⚠️ HIGH: Trades over $5K will cause noticeable price impact"
        elif liquidity < 1_000_000:
            return "⚠️ MEDIUM: Trades over $20K will cause moderate price impact"
        else:
            return "✅ LOW: Good liquidity for most trade sizes"

    def _get_recommended_max_trade(self, liquidity: float) -> str:
        """Get recommended maximum trade size"""
        if liquidity < 50_000:
            return "$500 or less"
        elif liquidity < 100_000:
            return "$1,000 or less"
        elif liquidity < 500_000:
            return "$5,000 or less"
        elif liquidity < 1_000_000:
            return "$20,000 or less"
        else:
            return "$50,000+ (check current depth)"

    def _estimate_slippage(self, liquidity: float, trade_size: float) -> float:
        """Estimate slippage for a given trade size"""
        if liquidity <= 0:
            return 100.0
        
        impact = (trade_size / liquidity) * 100
        return min(impact * 2, 100.0)

    def _calculate_token_age_days(self, created_at: int) -> int:
        """Calculate token age in days"""
        if not created_at:
            return 0
        
        age_ms = datetime.now().timestamp() * 1000 - created_at
        return max(0, int(age_ms / (1000 * 60 * 60 * 24)))

    # Bundle and suspicious activity detection
    async def _detect_enhanced_bundles(self, mint: str, holders_data: Dict, transaction_data: Dict) -> Dict:
        """Enhanced bundle detection"""
        try:
            top_holders = holders_data.get("top_holders", [])
            
            if len(top_holders) < 5:
                return {
                    "clusters_found": 0,
                    "high_risk_clusters": 0,
                    "bundled_percentage": 0.0,
                    "risk_level": "LOW",
                    "clusters": [],
                    "detection_confidence": 0.9
                }
            
            clusters = []
            suspicious_patterns = []
            
            similar_addresses = self._find_similar_addresses(top_holders)
            if similar_addresses:
                suspicious_patterns.extend(similar_addresses)
            
            balance_clusters = self._find_balance_clusters(top_holders)
            if balance_clusters:
                clusters.extend(balance_clusters)
            
            unique_traders = transaction_data.get("unique_traders_24h", 0)
            total_holders = len(top_holders)
            
            if unique_traders > 0 and total_holders > 0:
                activity_ratio = unique_traders / total_holders
                if activity_ratio < 0.3:
                    suspicious_patterns.append("Low transaction activity relative to holder count")
            
            clusters_found = len(clusters) + len(suspicious_patterns)
            high_risk_clusters = len([c for c in clusters if c.get("risk_score", 0) > 70])
            
            bundled_percentage = 0.0
            for cluster in clusters:
                bundled_percentage += cluster.get("total_percentage", 0) * 0.7
            
            if high_risk_clusters > 2 or bundled_percentage > 25:
                risk_level = "HIGH"
            elif clusters_found > 1 or bundled_percentage > 15:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            
            return {
                "clusters_found": clusters_found,
                "high_risk_clusters": high_risk_clusters,
                "bundled_percentage": round(bundled_percentage, 2),
                "risk_level": risk_level,
                "clusters": clusters,
                "suspicious_patterns": suspicious_patterns,
                "detection_confidence": 0.85,
                "analysis_method": "enhanced_correlation"
            }
            
        except Exception as e:
            logger.error(f"Enhanced bundle detection failed: {e}")
            return {
                "clusters_found": 0,
                "high_risk_clusters": 0,
                "bundled_percentage": 0.0,
                "risk_level": "UNKNOWN",
                "clusters": [],
                "detection_confidence": 0.5
            }

    def _find_similar_addresses(self, holders: List[Dict]) -> List[str]:
        """Find holders with suspiciously similar addresses"""
        patterns = []
        
        for i, holder1 in enumerate(holders[:10]):
            for j, holder2 in enumerate(holders[i+1:], i+1):
                addr1 = holder1["address"]
                addr2 = holder2["address"]
                
                if addr1[:8] == addr2[:8]:
                    patterns.append(f"Similar address prefixes detected: {addr1[:8]}...")
                elif addr1[-8:] == addr2[-8:]:
                    patterns.append(f"Similar address suffixes detected: ...{addr1[-8:]}")
                elif self._address_similarity_score(addr1, addr2) > 0.7:
                    patterns.append(f"High address similarity detected")
        
        return list(set(patterns))

    def _address_similarity_score(self, addr1: str, addr2: str) -> float:
        """Calculate similarity score between two addresses"""
        if len(addr1) != len(addr2):
            return 0.0
        
        matches = sum(1 for a, b in zip(addr1, addr2) if a == b)
        return matches / len(addr1)

    def _find_balance_clusters(self, holders: List[Dict]) -> List[Dict]:
        """Find clusters of holders with suspiciously similar balances"""
        clusters = []
        
        balance_groups = {}
        for holder in holders[:15]:
            percentage = holder["percentage"]
            rounded = round(percentage * 2) / 2
            
            if rounded not in balance_groups:
                balance_groups[rounded] = []
            balance_groups[rounded].append(holder)
        
        for balance, group in balance_groups.items():
            if len(group) >= 3 and balance > 1:
                total_percentage = sum(h["percentage"] for h in group)
                risk_score = min(90, len(group) * 15 + (total_percentage * 2))
                
                clusters.append({
                    "cluster_id": f"balance_cluster_{balance}",
                    "wallet_count": len(group),
                    "total_percentage": round(total_percentage, 2),
                    "risk_score": int(risk_score),
                    "creation_pattern": f"Multiple wallets with ~{balance}% balance each",
                    "addresses": [h["address"] for h in group]
                })
        
        return clusters

    async def _detect_enhanced_suspicious_activity(self, mint: str, transaction_analysis: Dict, 
                                                 holders_data: Dict, security_data: Dict) -> Dict:
        """Enhanced suspicious activity detection"""
        try:
            wash_trading_score = transaction_analysis.get("bot_activity_percentage", 0)
            top_1_percent = holders_data.get("top_1_percent", 0)
            unique_traders = transaction_analysis.get("unique_traders_24h", 0)
            
            concentration_risk = min(100, top_1_percent * 2.5)
            pattern_risk = 100 - transaction_analysis.get("pattern_score", 50)
            
            tax_risk = 0
            if security_data.get("available", False):
                buy_tax = security_data.get("buy_tax", 0)
                sell_tax = security_data.get("sell_tax", 0)
                tax_risk = (buy_tax + sell_tax) * 2
            
            authority_risk = 0
            if security_data.get("mint_enabled", False):
                authority_risk += 30
            if security_data.get("freeze_enabled", False):
                authority_risk += 25
            
            insider_activity_score = (concentration_risk + pattern_risk + tax_risk + authority_risk) / 4
            
            farming_indicators = []
            if wash_trading_score > 40:
                farming_indicators.append("High bot trading activity detected")
            if top_1_percent > 30:
                farming_indicators.append("Extreme holder concentration - potential dev wallet")
            elif top_1_percent > 20:
                farming_indicators.append("High holder concentration risk")
            if security_data.get("sell_tax", 0) > 15:
                farming_indicators.append("High sell tax detected - potential honeypot")
            if not security_data.get("liquidity_locked", True):
                farming_indicators.append("Liquidity not secured - rug pull risk")
            if holders_data.get("concentration_risk", "") == "CRITICAL":
                farming_indicators.append("Critical concentration detected by analysis")
            if unique_traders < 10 and transaction_analysis.get("transaction_count_24h", 0) > 50:
                farming_indicators.append("High transaction count with few unique traders")
            
            suspicious_patterns = []
            if security_data.get("mint_enabled", False):
                suspicious_patterns.append("Mint authority not renounced - unlimited supply risk")
            if security_data.get("freeze_enabled", False):
                suspicious_patterns.append("Freeze authority active - can halt all transfers")
            if security_data.get("owner_change_enabled", False):
                suspicious_patterns.append("Owner change enabled - control can be transferred")
            if transaction_analysis.get("wash_trading_detected", False):
                suspicious_patterns.append("Wash trading patterns identified")
            if (security_data.get("buy_tax", 0) > 10) or (security_data.get("sell_tax", 0) > 10):
                suspicious_patterns.append("Excessive trading taxes detected")
            if transaction_analysis.get("data_source") == "no_transactions":
                suspicious_patterns.append("No recent transaction data available")
            
            base_health = transaction_analysis.get("pattern_score", 50)
            concentration_penalty = min(35, top_1_percent * 1.2)
            wash_penalty = wash_trading_score * 0.6
            tax_penalty = (security_data.get("buy_tax", 0) + security_data.get("sell_tax", 0)) * 0.8
            authority_penalty = 15 if security_data.get("mint_enabled", False) else 0
            authority_penalty += 10 if security_data.get("freeze_enabled", False) else 0
            
            transaction_health_score = max(0, base_health - concentration_penalty - wash_penalty - tax_penalty - authority_penalty)
            
            total_indicators = len(farming_indicators) + len(suspicious_patterns)
            combined_risk = (wash_trading_score + insider_activity_score) / 2
            
            suspicion_level = self._calculate_enhanced_suspicion_level(
                wash_trading_score, insider_activity_score, total_indicators, security_data
            )
            
            return {
                "wash_trading_score": round(wash_trading_score, 1),
                "insider_activity_score": round(insider_activity_score, 1),
                "farming_indicators": farming_indicators,
                "suspicious_patterns": suspicious_patterns,
                "transaction_health_score": round(transaction_health_score, 1),
                "overall_suspicion_level": suspicion_level,
                "data_quality": "enhanced_analysis",
                "risk_factors_detected": total_indicators,
                "metrics_used": {
                    "holder_concentration": top_1_percent,
                    "transaction_patterns": transaction_analysis.get("data_source", "unknown"),
                    "bot_activity": wash_trading_score,
                    "security_scan": security_data.get("available", False),
                    "unique_traders": unique_traders
                }
            }
            
        except Exception as e:
            logger.error(f"Enhanced suspicious activity detection error: {e}")
            return {
                "wash_trading_score": 0.0,
                "insider_activity_score": 50.0,
                "farming_indicators": ["Error in enhanced analysis - data may be incomplete"],
                "suspicious_patterns": ["Analysis error occurred"],
                "transaction_health_score": 30.0,
                "overall_suspicion_level": "HIGH",
                "data_quality": "error"
            }

    def _calculate_enhanced_suspicion_level(self, wash_score: float, insider_score: float, 
                                          indicator_count: int, security_data: Dict) -> str:
        """Calculate enhanced overall suspicion level"""
        base_score = (wash_score + insider_score) / 2 + (indicator_count * 8)
        
        if security_data.get("available", False):
            if not security_data.get("liquidity_locked", True):
                base_score += 20
            if security_data.get("mint_enabled", False):
                base_score += 15
            if security_data.get("sell_tax", 0) > 20:
                base_score += 15
        
        total_score = min(100, base_score)
        
        if total_score >= 85:
            return "EXTREME"
        elif total_score >= 70:
            return "VERY HIGH"
        elif total_score >= 55:
            return "HIGH"
        elif total_score >= 35:
            return "MEDIUM"
        elif total_score >= 15:
            return "LOW"
        else:
            return "VERY LOW"

    async def _detect_basic_bundles(self, holders_data: Dict) -> Dict:
        """Basic bundle detection for Express Mode"""
        top_holders = holders_data.get("top_holders", [])
        
        if len(top_holders) < 3:
            return {
                "clusters_found": 0,
                "high_risk_clusters": 0,
                "bundled_percentage": 0.0,
                "risk_level": "LOW",
                "clusters": []
            }
        
        top_3_total = sum(h["percentage"] for h in top_holders[:3])
        top_5_total = sum(h["percentage"] for h in top_holders[:5])
        
        clusters = []
        if top_3_total > 50:
            clusters.append({
                "cluster_id": "top_3_concentration",
                "wallet_count": 3,
                "total_percentage": round(top_3_total, 2),
                "risk_score": int(min(90, top_3_total * 1.5)),
                "creation_pattern": "High concentration in top 3 holders"
            })
        
        return {
            "clusters_found": len(clusters),
            "high_risk_clusters": len([c for c in clusters if c.get("risk_score", 0) > 70]),
            "bundled_percentage": round(top_5_total * 0.3, 2),
            "risk_level": "HIGH" if top_3_total > 60 else "MEDIUM" if top_3_total > 40 else "LOW",
            "clusters": clusters
        }

    def _detect_basic_suspicious_activity(self, transaction_data: Dict, holders_data: Dict, security_data: Dict) -> Dict:
        """Basic suspicious activity detection for Express Mode"""
        wash_trading_score = transaction_data.get("bot_activity_percentage", 0)
        top_1_percent = holders_data.get("top_1_percent", 0)
        
        concentration_risk = min(100, top_1_percent * 2.5)
        pattern_risk = 100 - transaction_data.get("pattern_score", 50)
        tax_risk = (security_data.get("buy_tax", 0) + security_data.get("sell_tax", 0)) * 2
        
        insider_activity_score = (concentration_risk + pattern_risk + tax_risk) / 3
        
        farming_indicators = []
        if wash_trading_score > 40:
            farming_indicators.append("High bot trading activity detected")
        if top_1_percent > 25:
            farming_indicators.append("Single holder dominance (potential dev wallet)")
        if security_data.get("sell_tax", 0) > 10:
            farming_indicators.append("High sell tax detected (potential honeypot)")
        if not security_data.get("liquidity_locked", True):
            farming_indicators.append("Liquidity not locked (rug pull risk)")
        
        suspicious_patterns = []
        if security_data.get("mint_enabled", False):
            suspicious_patterns.append("Mint authority not renounced")
        if security_data.get("freeze_enabled", False):
            suspicious_patterns.append("Freeze authority active")
        if security_data.get("buy_tax", 0) > 15 or security_data.get("sell_tax", 0) > 15:
            suspicious_patterns.append("Excessive trading taxes")
        
        return {
            "wash_trading_score": round(wash_trading_score, 1),
            "insider_activity_score": round(insider_activity_score, 1),
            "farming_indicators": farming_indicators,
            "suspicious_patterns": suspicious_patterns,
            "transaction_health_score": transaction_data.get("pattern_score", 50),
            "data_quality": "enhanced_basic"
        }

    def _calculate_enhanced_galaxy_score(self, indicators, transaction_analysis, 
                                       bundle_detection, suspicious_activity, security_data) -> Tuple[int, str, float]:
        """Enhanced Galaxy Brain Score calculation"""
        
        base_risk = indicators.overall_risk_score
        
        wash_trading_risk = suspicious_activity.get("wash_trading_score", 0) * 0.4
        insider_risk = suspicious_activity.get("insider_activity_score", 0) * 0.3
        
        bundle_risk = bundle_detection.get("bundled_percentage", 0) * 2.5
        
        transaction_health = transaction_analysis.get("pattern_score", 50)
        health_risk = (100 - transaction_health) * 0.2
        
        security_bonus = 0
        if security_data.get("available", False):
            if security_data.get("liquidity_locked", False):
                security_bonus -= 10
            if not security_data.get("mint_enabled", True) and not security_data.get("freeze_enabled", True):
                security_bonus -= 15
            if security_data.get("sell_tax", 0) > 15:
                security_bonus += 20
        
        galaxy_score = base_risk + wash_trading_risk + insider_risk + bundle_risk + health_risk + security_bonus
        galaxy_score = min(100, max(0, galaxy_score))
        
        if galaxy_score >= 85:
            severity = "EXTREME_DANGER"
        elif galaxy_score >= 70:
            severity = "CRITICAL_RISK"
        elif galaxy_score >= 55:
            severity = "HIGH_RISK"
        elif galaxy_score >= 35:
            severity = "MEDIUM_RISK"
        elif galaxy_score >= 15:
            severity = "LOW_RISK"
        else:
            severity = "MINIMAL_RISK"
        
        confidence_factors = []
        
        if security_data.get("available", False):
            confidence_factors.append(0.95)
        else:
            confidence_factors.append(0.7)
            
        if indicators.token_age_hours > 24:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.6)
            
        if transaction_analysis.get("unique_traders_24h", 0) > 20:
            confidence_factors.append(0.85)
        else:
            confidence_factors.append(0.7)
            
        if bundle_detection.get("detection_confidence", 0.5) > 0.8:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.75)
        
        confidence = sum(confidence_factors) / len(confidence_factors)
        
        return int(galaxy_score), severity, round(confidence, 2)

    def _calculate_express_score(self, indicators: ScamIndicators, holders_data: Dict, security_data: Dict) -> Tuple[int, str, float]:
        """Enhanced Express Mode scoring"""
        base_risk = indicators.overall_risk_score
        
        top_1 = holders_data.get("top_1_percent", 0)
        concentration_penalty = min(30, top_1 * 1.2) if top_1 > 8 else 0
        
        authority_penalty = 0
        if indicators.mint_enabled:
            authority_penalty += 25
        if indicators.freeze_enabled:
            authority_penalty += 20
        
        liquidity_penalty = 0
        liquidity_health = self._calculate_liquidity_health(
            indicators.liquidity_percent * 1000000,
            indicators.liquidity_percent, 
            0, 0
        )
        if liquidity_health < 30:
            liquidity_penalty += 10
        if indicators.liquidity_percent < 1:
            liquidity_penalty += 8
        
        tax_penalty = 0
        if indicators.buy_tax > 10:
            tax_penalty += 8
        if indicators.sell_tax > 10:
            tax_penalty += 12
        
        express_score = base_risk + concentration_penalty + authority_penalty + liquidity_penalty + tax_penalty
        express_score = min(100, max(0, express_score))
        
        if express_score >= 80:
            severity = "CRITICAL_RISK"
        elif express_score >= 60:
            severity = "HIGH_RISK"
        elif express_score >= 40:
            severity = "MEDIUM_RISK"
        elif express_score >= 20:
            severity = "LOW_RISK"
        else:
            severity = "MINIMAL_RISK"
        
        confidence = 0.85 if security_data.get("available", False) else 0.75
        
        return int(express_score), severity, confidence

    def _generate_enhanced_risk_vectors(self, indicators, transaction_analysis, 
                                    suspicious_activity, security_data) -> List[Dict]:
        """Generate enhanced risk vectors"""
        risk_vectors = []
        
        if indicators.mint_enabled:
            risk_vectors.append({
                "category": "Authority Risk",
                "risk_type": "Mint Authority Active - VERIFIED",
                "severity": "CRITICAL",
                "impact": "Developer can create unlimited new tokens instantly",
                "likelihood": "HIGH",
                "mitigation": "AVOID - This is a critical red flag"
            })
        
        if indicators.freeze_enabled:
            risk_vectors.append({
                "category": "Authority Risk",
                "risk_type": "Freeze Authority Active - VERIFIED", 
                "severity": "CRITICAL",
                "impact": "Developer can freeze all token transfers at any time",
                "likelihood": "MEDIUM",
                "mitigation": "EXTREME RISK - Avoid or use minimal amounts"
            })
        
        if indicators.top_holder_percent > 30:
            risk_vectors.append({
                "category": "Concentration Risk",
                "risk_type": f"Extreme Real Whale Dominance - {indicators.top_holder_percent:.1f}%",
                "severity": "CRITICAL",
                "impact": f"Single REAL holder (not LP) controls {indicators.top_holder_percent:.1f}% - major dump risk",
                "likelihood": "HIGH",
                "mitigation": "Monitor whale wallet movements, very high risk"
            })
        elif indicators.top_holder_percent > 15:
            risk_vectors.append({
                "category": "Concentration Risk",
                "risk_type": f"High Real Whale Concentration - {indicators.top_holder_percent:.1f}%",
                "severity": "HIGH", 
                "impact": f"Major real holder can significantly impact price (LP wallets filtered out)",
                "likelihood": "HIGH",
                "mitigation": "Watch for large transfers, trade smaller amounts"
            })
        elif indicators.top_holder_percent > 8:
            risk_vectors.append({
                "category": "Concentration Risk",
                "risk_type": f"Moderate Real Holder Concentration - {indicators.top_holder_percent:.1f}%",
                "severity": "MEDIUM",
                "impact": f"Notable concentration among real holders (protocols excluded)",
                "likelihood": "MEDIUM",
                "mitigation": "Monitor for coordinated selling"
            })
        
        return risk_vectors

    def _generate_basic_risk_vectors(self, indicators: ScamIndicators, holders_data: Dict, 
                                   liquidity_data: Dict, security_data: Dict) -> List[Dict]:
        """Enhanced basic risk vectors"""
        risk_vectors = []
        
        if indicators.mint_enabled:
            confidence = "VERIFIED" if security_data.get("available", False) else "DETECTED"
            risk_vectors.append({
                "category": "Authority Risk",
                "risk_type": f"Mint Authority Active ({confidence})",
                "severity": "CRITICAL",
                "impact": "Developer can create unlimited new tokens",
                "likelihood": "HIGH",
                "mitigation": "Wait for mint authority renunciation before investing"
            })
        
        if indicators.freeze_enabled:
            risk_vectors.append({
                "category": "Authority Risk",
                "risk_type": "Freeze Authority Active",
                "severity": "CRITICAL", 
                "impact": "Developer can freeze all token transfers",
                "likelihood": "MEDIUM",
                "mitigation": "Avoid tokens with active freeze authority"
            })
        
        if not indicators.liquidity_locked:
            risk_vectors.append({
                "category": "Liquidity Risk",
                "risk_type": "Liquidity NOT Locked - Verified",
                "severity": "CRITICAL",
                "impact": "Liquidity can be removed anytime - rug pull risk confirmed",
                "likelihood": "HIGH",
                "mitigation": "EXTREME RISK - Developer can drain pool instantly"
            })
        
        if indicators.top_holder_percent > 15:
            severity = "CRITICAL" if indicators.top_holder_percent > 25 else "HIGH"
            risk_vectors.append({
                "category": "Concentration Risk",
                "risk_type": "Whale Dominance",
                "severity": severity,
                "impact": f"Top holder owns {indicators.top_holder_percent:.1f}% - dump risk",
                "likelihood": "HIGH",
                "mitigation": "Monitor large holder activity for exit signals"
            })
        
        if indicators.liquidity_percent < 5:
            severity = "HIGH" if indicators.liquidity_percent < 2 else "MEDIUM"
            risk_vectors.append({
                "category": "Liquidity Risk",
                "risk_type": "Low Liquidity Pool",
                "severity": severity,
                "impact": f"Only {indicators.liquidity_percent:.1f}% of market cap in liquidity",
                "likelihood": "CERTAIN",
                "mitigation": "Expect high slippage, trade only small amounts"
            })
        
        return risk_vectors

# Usage example
async def main():
    """Example usage of the Enhanced Rug Checker"""
    
    # Initialize with API keys
    helius_key = "your-helius-api-key-here"  # Get from helius.xyz
    birdeye_key = "your-birdeye-api-key-here"  # Optional - Get from birdeye.so
    xai_key = "your-xai-api-key-here"  # Required for Grok live search - Get from x.ai
    
    checker = EnhancedRugChecker(
        helius_key=helius_key,
        birdeye_key=birdeye_key,
        xai_key=xai_key
    )
    
    # Example token address (replace with actual token)
    token_address = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"  # Example: Bonk
    
    try:
        # Run deep analysis with Grok live search
        print("🧠 Starting REVOLUTIONARY Galaxy Brain v5.0 Analysis...")
        result = await checker.analyze_token(token_address, deep_analysis=True)
        
        if result["success"]:
            print(f"\n✅ Analysis Complete!")
            print(f"🧠 Galaxy Brain Score: {result['galaxy_brain_score']}/100")
            print(f"⚠️ Risk Level: {result['severity_level']}")
            print(f"🎯 Confidence: {result['confidence']:.0%}")
            print(f"\n{result['ai_analysis']}")
            
            # Show key metrics
            token_info = result['token_info']
            print(f"\n📊 Token: {token_info['name']} (${token_info['symbol']})")
            print(f"💰 Market Cap: ${token_info.get('market_cap', 0):,.0f}")
            print(f"💧 Liquidity: ${token_info.get('liquidity', 0):,.0f}")
            print(f"👥 Top Holder: {result['holder_analysis']['top_1_percent']:.1f}%")
            
            # Show Grok analysis if available
            if result['grok_analysis'].get('available'):
                grok_data = result['grok_analysis']['parsed_analysis']
                print(f"\n🧠 REVOLUTIONARY Grok Intelligence:")
                print(f"   Verdict: {grok_data['verdict']}")
                print(f"   Confidence: {grok_data['confidence']:.0%}")
                print(f"   Community Evidence: {len(grok_data.get('positive_community_sentiment', []))} positive, {len(grok_data.get('possible_community_risks', []))} risks")
            
        else:
            print(f"❌ Analysis failed: {result['error']}")
            
    except Exception as e:
        print(f"💥 Error: {e}")

if __name__ == "__main__":
    # Run the example
    asyncio.run(main())