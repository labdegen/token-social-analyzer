import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json
import random
import time
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("enhanced_rugcheck")

# Wallet addresses to exclude from holder analysis
EXCLUDED_WALLETS = {
    "HFNekd429efPTM8eux66dDcgAC6WcSRoRNLtn6e81cEn",  # User requested exclusion
    "11111111111111111111111111111111",  # System program
    "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",  # Token program
}

@dataclass
class ScamIndicators:
    """Enhanced indicators for scam detection with Solana Tracker data"""
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
    
    # Enhanced with Solana Tracker
    bundle_detected: bool = False
    deployer_reputation: str = "UNKNOWN"
    first_buyer_concentration: float = 0.0
    
    # Risk scores from SolanaTracker
    solana_tracker_risk_score: int = 0
    rugged_status: bool = False
    
    # Age & Activity
    token_age_hours: int = 0
    unique_holders: int = 0
    
    # Scores
    overall_risk_score: int = 0
    risk_level: str = "UNKNOWN"

class EnhancedRugCheckerWithSolanaTracker:
    def __init__(self, solana_tracker_key: str, xai_key: str = None):
        self.solana_tracker_key = solana_tracker_key
        self.xai_key = xai_key
        self.solana_tracker_base = "https://data.solanatracker.io"
        self.last_api_call = 0 
        
        # Debug logging
        logger.info(f"🚀 Solana Tracker API: {'✅ Connected' if self.solana_tracker_key else '❌ No Key'}")
        logger.info(f"🧠 XAI API Key: {'✅ Available' if self.xai_key and self.xai_key != 'your-xai-api-key-here' else '❌ Missing'}")
        
    async def analyze_token(self, token_address: str, deep_analysis: bool = True) -> Dict:
        """
        🧠 REVOLUTIONARY Galaxy Brain v6.1 Analysis - FIXED with Proper Market Cap, Filtered Wallets, and Focused Grok
        """
        session = None
        try:
            start_time = time.time()
            mode = "DEEP" if deep_analysis else "EXPRESS"
            logger.info(f"🧠 REVOLUTIONARY Galaxy Brain v6.1 {mode} Mode analyzing: {token_address}")
            
            session = aiohttp.ClientSession()
            headers = {"x-api-key": self.solana_tracker_key}
            
            # Step 1: Get comprehensive token data from Solana Tracker (FIXED ENDPOINT)
            token_data = await self._get_solana_tracker_token_data(token_address, session, headers)
            if not token_data.get("success"):
                return {
                    "success": False,
                    "error": token_data.get("error", "Token not found"),
                    "suggestions": [
                        "Verify the token address is correct",
                        "Ensure token exists on Solana mainnet", 
                        "Check if token has been deployed recently"
                    ]
                }
            
            # Step 2: Get holder analysis with FILTERED wallets
            holders_data = await self._get_enhanced_holders_analysis(token_address, session, headers)
            
            # Step 3: Get trading analysis
            trading_data = await self._get_trading_analysis(token_address, session, headers, deep_analysis)
            
            # Step 4: Get deployer analysis
            deployer_analysis = await self._get_deployer_analysis(token_data["token_info"], session, headers)
            
            # Step 5: Get first buyer analysis
            first_buyers_data = await self._get_first_buyers_analysis(token_address, session, headers)
            
            # Step 6: Get performance data
            performance_data = await self._get_performance_analysis(token_address, session, headers)
            
            # Step 7: Calculate enhanced indicators with SolanaTracker risk data
            indicators = self._calculate_enhanced_indicators_v61(
                token_data["token_info"], holders_data, trading_data, 
                deployer_analysis, first_buyers_data, performance_data
            )
            
            # Step 8: 🧠 FOCUSED GROK ANALYSIS - X/Twitter sentiment only
            logger.info("🧠 REVOLUTIONARY: Calling Grok for X/Twitter sentiment analysis...")
            grok_analysis = await self._grok_focused_sentiment_analysis_v61(
                token_address, token_data["token_info"], indicators, session
            )
            
            # Step 9: Enhanced bundle and manipulation detection
            if deep_analysis:
                bundle_detection = await self._detect_enhanced_bundles_v61(
                    token_data["token_info"], holders_data, trading_data, first_buyers_data
                )
                manipulation_detection = await self._detect_manipulation_patterns(
                    token_address, trading_data, performance_data, session, headers
                )
                
                # Enhanced scoring with all data sources
                galaxy_brain_score, severity_level, confidence = self._calculate_revolutionary_galaxy_score_v61(
                    indicators, trading_data, bundle_detection, manipulation_detection, 
                    deployer_analysis, grok_analysis
                )
                
                risk_vectors = self._generate_enhanced_risk_vectors_v61(
                    indicators, trading_data, bundle_detection, manipulation_detection, 
                    deployer_analysis, grok_analysis
                )
                
            else:
                bundle_detection = await self._detect_basic_bundles_v61(holders_data, first_buyers_data)
                manipulation_detection = self._detect_basic_manipulation(trading_data, performance_data)
                
                galaxy_brain_score, severity_level, confidence = self._calculate_express_score_v61(
                    indicators, holders_data, trading_data, deployer_analysis, grok_analysis
                )
                
                risk_vectors = self._generate_basic_risk_vectors_v61(
                    indicators, holders_data, trading_data, deployer_analysis, grok_analysis
                )
            
            analysis_time = time.time() - start_time
            
            # 🧠 REVOLUTIONARY response with FIXED formatting for frontend
            return {
                "success": True,
                "analysis_mode": mode,
                "galaxy_brain_score": galaxy_brain_score,
                "severity_level": severity_level,
                "confidence": confidence,
                
                # 🧠 REVOLUTIONARY: Enhanced Grok Analysis
                "grok_analysis": grok_analysis,
                "ai_analysis": self._format_grok_ai_analysis_v61(grok_analysis),
                
                # FIXED token info for frontend with PROPER market cap
                "token_info": {
                    "address": token_address,
                    "mint": token_address,
                    "name": token_data["token_info"].get("name", "Unknown Token"),
                    "symbol": token_data["token_info"].get("symbol", "UNKNOWN"),
                    "decimals": token_data["token_info"].get("decimals", 6),
                    "supply": token_data["token_info"].get("supply", 0),
                    "market_cap": token_data["token_info"].get("market_cap", 0),  # FIXED
                    "price": token_data["token_info"].get("price", 0),
                    "liquidity": token_data["token_info"].get("liquidity", 0),
                    "volume_24h": token_data["token_info"].get("volume_24h", 0),
                    "price_change_24h": token_data["token_info"].get("price_change_24h", 0),
                    "holders": token_data["token_info"].get("holders", 0),
                    "age_days": indicators.token_age_hours / 24 if indicators.token_age_hours > 0 else 0,
                    "logo": token_data["token_info"].get("logo"),
                    "website": token_data["token_info"].get("website"),
                    "twitter": token_data["token_info"].get("twitter"),
                    "telegram": token_data["token_info"].get("telegram"),
                    "is_mutable": indicators.mint_enabled or indicators.freeze_enabled,
                    "deployer_reputation": deployer_analysis.get("reputation", "UNKNOWN"),
                    "bundle_detected": indicators.bundle_detected,
                    "market": token_data["token_info"].get("market", "unknown"),
                    "rugged_status": indicators.rugged_status,
                    "solana_tracker_risk_score": indicators.solana_tracker_risk_score
                },
                
                # FIXED analysis results for frontend
                "holder_analysis": self._format_holder_analysis_for_frontend(holders_data),
                "trading_analysis": trading_data,
                "deployer_analysis": deployer_analysis,
                "first_buyers_analysis": first_buyers_data,
                "performance_analysis": performance_data,
                "bundle_detection": bundle_detection,
                "manipulation_detection": manipulation_detection,
                "scam_indicators": indicators,
                "risk_vectors": risk_vectors,
                
                # FIXED: Add missing analysis sections for frontend
                "transaction_analysis": self._create_transaction_analysis(trading_data, performance_data),
                "suspicious_activity": self._create_suspicious_activity_analysis(trading_data, manipulation_detection),
                "liquidity_analysis": self._create_liquidity_analysis(token_data["token_info"], trading_data),
                
                # Enhanced metadata
                "analysis_time_seconds": round(analysis_time, 2),
                "analysis_timestamp": datetime.now().isoformat(),
                "analysis_version": "6.1_REVOLUTIONARY_FIXED_FILTERED_GROK",
                "data_sources": {
                    "primary": "solana_tracker_api_v2_fixed",
                    "token_data": "solana_tracker_comprehensive",
                    "holder_data": "solana_tracker_filtered" if holders_data.get("data_source") == "solana_tracker_live" else "limited",
                    "trading_data": "solana_tracker_real_time" if trading_data.get("data_source") == "solana_tracker_live" else "estimated",
                    "risk_data": "solana_tracker_integrated",
                    "grok_intelligence": "focused_x_twitter_sentiment" if grok_analysis.get("available") else "unavailable",
                    "api_credits_used": "estimated_4_10"
                }
            }
            
        except Exception as e:
            logger.error(f"Revolutionary v6.1 analysis failed: {e}")
            return {
                "success": False,
                "error": f"Analysis failed: {str(e)}",
                "error_type": "analysis_error",
                "analysis_mode": mode if 'mode' in locals() else "UNKNOWN",
                "suggestions": [
                    "Check Solana Tracker API key validity",
                    "Verify network connectivity",
                    "Ensure token address is valid Solana address",
                    "Try again in a few moments (rate limit may be active)"
                ]
            }
        finally:
            if session:
                await session.close()

    async def _get_solana_tracker_token_data(self, token_address: str, session: aiohttp.ClientSession, headers: Dict) -> Dict:
        """Get comprehensive token data with FIXED market cap extraction and risk data integration"""
        try:
            # Rate limit compliance
            await self._rate_limit_delay(delay=1.5)
            
            # CORRECT ENDPOINT: /tokens/{tokenAddress}
            url = f"{self.solana_tracker_base}/tokens/{token_address}"
            
            async with session.get(url, headers=headers, timeout=20) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    logger.info(f"🚀 Solana Tracker token data retrieved successfully")
                    
                    # FIXED: Parse the CORRECT response structure from API docs
                    token = data.get("token", {})
                    pools = data.get("pools", [])
                    risk_data = data.get("risk", {})  # Extract risk data
                    
                    # Get primary pool data safely
                    primary_pool = pools[0] if pools and len(pools) > 0 else {}
                    liquidity_data = primary_pool.get("liquidity", {}) if primary_pool else {}
                    
                    # FIXED: Extract creation data properly
                    creation_data = token.get("creation", {})
                    created_time = creation_data.get("created_time", 0)
                    if created_time:
                        created_time = int(created_time) * 1000  # Convert to milliseconds
                    
                    # FIXED: Extract market cap from MULTIPLE possible locations
                    market_cap = 0
                    market_cap_sources = [
                        data.get("marketCap"),
                        data.get("market_cap"), 
                        primary_pool.get("marketCap"),
                        primary_pool.get("market_cap"),
                        token.get("marketCap"),
                        token.get("market_cap")
                    ]
                    
                    for source in market_cap_sources:
                        if source is not None:
                            market_cap = self._safe_float(source)
                            if market_cap > 0:
                                break
                    
                    # Extract other price and market data
                    price = self._safe_float(data.get("price", primary_pool.get("price", {}).get("usd", 0)))
                    volume_24h = self._safe_float(data.get("volume24h", primary_pool.get("volume24h", 0)))
                    holders_count = self._safe_int(data.get("holders", token.get("holders", 0)))
                    
                    # FIXED: Extract security/risk data properly with SolanaTracker risk integration
                    freeze_authority = risk_data.get("freezeAuthority") or primary_pool.get("security", {}).get("freezeAuthority")
                    mint_authority = risk_data.get("mintAuthority") or primary_pool.get("security", {}).get("mintAuthority")
                    
                    # Extract SolanaTracker risk assessment
                    rugged_status = risk_data.get("rugged", False)
                    risk_score = risk_data.get("score", 0)
                    risk_warnings = risk_data.get("risks", [])
                    
                    # FIXED: Extract social links properly
                    links = token.get("links", {}) if isinstance(token.get("links"), dict) else {}
                    
                    # Process the token data with CORRECT structure and FIXED market cap
                    token_info = {
                        "address": token_address,
                        "name": token.get("name", "Unknown Token"),
                        "symbol": token.get("symbol", "UNKNOWN"),
                        "decimals": self._safe_int(token.get("decimals", 6)),
                        "supply": self._safe_float(token.get("supply", primary_pool.get("tokenSupply", 0))),
                        "market_cap": market_cap,  # FIXED - now properly extracted
                        "price": price,
                        "liquidity": self._safe_float(liquidity_data.get("usd", 0)),
                        "volume_24h": volume_24h,
                        "price_change_24h": self._safe_float(data.get("priceChange24h", 0)),
                        "holders": holders_count,
                        "created_at": created_time,
                        "mint_authority": mint_authority,
                        "freeze_authority": freeze_authority,
                        "bundleId": primary_pool.get("bundleId") if primary_pool else None,
                        "market": primary_pool.get("market", "unknown") if primary_pool else "unknown",
                        "deployer": creation_data.get("creator"),
                        "logo": token.get("image"),
                        "description": token.get("description", ""),
                        "website": links.get("website"),
                        "twitter": links.get("twitter"),
                        "telegram": links.get("telegram"),
                        # SolanaTracker risk data integration
                        "rugged_status": rugged_status,
                        "risk_score": risk_score,
                        "risk_warnings": risk_warnings
                    }
                    
                    logger.info(f"✅ Token parsed: {token_info['name']} ({token_info['symbol']}) - Market Cap: ${market_cap:,.0f}")
                    return {"success": True, "token_info": token_info}
                    
                elif resp.status == 429:
                    logger.warning("Rate limited by Solana Tracker - using fallback data")
                    return self._create_fallback_token_data(token_address)
                elif resp.status == 404:
                    return {"success": False, "error": "Token not found in Solana Tracker database"}
                else:
                    error_text = await resp.text()
                    logger.error(f"Solana Tracker API error {resp.status}: {error_text}")
                    return self._create_fallback_token_data(token_address)
                    
        except Exception as e:
            logger.error(f"Solana Tracker token data error: {e}")
            return self._create_fallback_token_data(token_address)

    async def _get_enhanced_holders_analysis(self, token_address: str, session: aiohttp.ClientSession, headers: Dict) -> Dict:
        """Get holder analysis with FILTERED wallets (excluding specified addresses)"""
        try:
            # Rate limit compliance
            await self._rate_limit_delay(delay=1.5)
            
            # CORRECT ENDPOINT: /tokens/{tokenAddress}/holders/top
            url = f"{self.solana_tracker_base}/tokens/{token_address}/holders/top"
            
            async with session.get(url, headers=headers, timeout=15) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    
                    # FIXED: Parse CORRECT response structure - data is directly an array
                    if isinstance(data, list):
                        holders = data
                    else:
                        holders = data.get("holders", [])
                    
                    if not holders:
                        logger.warning("No holders data returned")
                        return self._create_fallback_holders_data()
                    
                    logger.info(f"✅ Retrieved {len(holders)} holders before filtering")
                    
                    # FILTER OUT EXCLUDED WALLETS
                    filtered_holders = []
                    excluded_count = 0
                    
                    for i, holder in enumerate(holders):
                        if isinstance(holder, dict):
                            owner = holder.get("owner", holder.get("address", ""))
                            
                            # Skip excluded wallets
                            if owner in EXCLUDED_WALLETS:
                                excluded_count += 1
                                logger.info(f"🚫 Filtered out excluded wallet: {owner[:8]}...")
                                continue
                            
                            percentage = self._safe_float(holder.get("percentage", 0))
                            balance = self._safe_float(holder.get("balance", 0))
                            
                            filtered_holders.append({
                                "rank": len(filtered_holders) + 1,  # Rerank after filtering
                                "address": owner,
                                "balance": balance,
                                "percentage": percentage
                            })
                    
                    logger.info(f"✅ After filtering: {len(filtered_holders)} holders ({excluded_count} excluded)")
                    
                    # Recalculate concentration metrics with filtered data
                    top_1 = filtered_holders[0]["percentage"] if filtered_holders else 0
                    top_5 = sum(h["percentage"] for h in filtered_holders[:5])
                    top_10 = sum(h["percentage"] for h in filtered_holders[:10])
                    
                    return {
                        "top_1_percent": round(top_1, 2),
                        "top_5_percent": round(top_5, 2),
                        "top_10_percent": round(top_10, 2),
                        "top_holders": filtered_holders,
                        "total_holders": len(filtered_holders),
                        "excluded_wallets_count": excluded_count,
                        "concentration_risk": self._assess_concentration_risk(top_1, top_5),
                        "data_source": "solana_tracker_live"
                    }
                else:
                    logger.warning(f"Failed to get holders data: {resp.status}")
                    return self._create_fallback_holders_data()
                    
        except Exception as e:
            logger.error(f"Enhanced holders analysis error: {e}")
            return self._create_fallback_holders_data()

    # Continue with other methods...
    async def _get_trading_analysis(self, token_address: str, session: aiohttp.ClientSession, headers: Dict, deep_analysis: bool) -> Dict:
        """Get trading analysis using CORRECT endpoint /trades/{tokenAddress} with FIXED parsing"""
        try:
            # Rate limit compliance
            await self._rate_limit_delay(delay=1.5)
            
            # CORRECT ENDPOINT: /trades/{tokenAddress}
            url = f"{self.solana_tracker_base}/trades/{token_address}"
            params = {"limit": 100 if deep_analysis else 50}
            
            async with session.get(url, headers=headers, params=params, timeout=15) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    
                    # FIXED: Parse CORRECT response structure
                    if isinstance(data, list):
                        trades = data
                    else:
                        trades = data.get("trades", [])
                    
                    if not trades:
                        logger.warning("No trades data returned")
                        return self._create_fallback_trading_data()
                    
                    logger.info(f"✅ Retrieved {len(trades)} trades")
                    
                    # FIXED: Analyze trading patterns with proper field names
                    unique_traders = set()
                    buy_trades = 0
                    sell_trades = 0
                    total_volume = 0
                    
                    for trade in trades:
                        if isinstance(trade, dict):
                            trader = trade.get("owner", trade.get("wallet", ""))
                            unique_traders.add(trader)
                            
                            # FIXED: Check trade type from correct fields
                            trade_type = trade.get("type", "").lower()
                            is_buy = trade.get("is_buy", False)
                            
                            if trade_type in ["buy", "purchase"] or is_buy:
                                buy_trades += 1
                            else:
                                sell_trades += 1
                                
                            # FIXED: Get volume from correct field
                            volume = self._safe_float(trade.get("volumeUsd", trade.get("volume_usd", 0)))
                            total_volume += volume
                    
                    # Calculate wash trading indicators
                    trade_ratio = len(trades) / len(unique_traders) if unique_traders else 0
                    wash_trading_score = min(90, max(0, (trade_ratio - 1) * 20))
                    
                    return {
                        "total_trades": len(trades),
                        "unique_traders": len(unique_traders),
                        "buy_trades": buy_trades,
                        "sell_trades": sell_trades,
                        "buy_sell_ratio": buy_trades / max(sell_trades, 1),
                        "total_volume_usd": total_volume,
                        "avg_trade_size": total_volume / len(trades) if trades else 0,
                        "wash_trading_score": round(wash_trading_score, 1),
                        "trade_frequency": len(trades) / len(unique_traders) if unique_traders else 0,
                        "data_source": "solana_tracker_live"
                    }
                else:
                    logger.warning(f"Failed to get trading data: {resp.status}")
                    return self._create_fallback_trading_data()
                    
        except Exception as e:
            logger.error(f"Trading analysis error: {e}")
            return self._create_fallback_trading_data()

    # Deployer, first buyers, performance analysis methods remain the same...
    async def _get_deployer_analysis(self, token_info: Dict, session: aiohttp.ClientSession, headers: Dict) -> Dict:
        """Analyze deployer using CORRECT endpoint /deployer/{wallet} with FIXED parsing"""
        try:
            deployer = token_info.get("deployer")
            if not deployer:
                return {"reputation": "UNKNOWN", "reason": "No deployer information", "data_source": "no_data"}
            
            # Rate limit compliance
            await self._rate_limit_delay(delay=1.5)
            
            # CORRECT ENDPOINT: /deployer/{wallet}
            url = f"{self.solana_tracker_base}/deployer/{deployer}"
            
            async with session.get(url, headers=headers, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    
                    # FIXED: Parse CORRECT response structure
                    if isinstance(data, list):
                        tokens = data
                    else:
                        tokens = data.get("tokens", [])
                    
                    total_tokens = len(tokens)
                    successful_tokens = 0
                    rugged_tokens = 0
                    
                    # FIXED: Analyze tokens with proper field access
                    for token in tokens:
                        if isinstance(token, dict):
                            market_cap = self._safe_float(token.get("marketCap", token.get("market_cap", 0)))
                            liquidity = self._safe_float(token.get("liquidity", 0))
                            
                            if market_cap > 50000:
                                successful_tokens += 1
                            if liquidity == 0 and market_cap > 0:
                                rugged_tokens += 1
                    
                    # Calculate reputation score
                    if total_tokens == 0:
                        reputation = "NEW_DEPLOYER"
                    elif total_tokens == 1:
                        reputation = "FIRST_TOKEN"
                    elif total_tokens > 0 and rugged_tokens / total_tokens > 0.7:
                        reputation = "HIGH_RISK_DEPLOYER"
                    elif total_tokens > 0 and rugged_tokens / total_tokens > 0.4:
                        reputation = "MODERATE_RISK_DEPLOYER"
                    elif total_tokens > 0 and successful_tokens / total_tokens > 0.3:
                        reputation = "EXPERIENCED_DEPLOYER"
                    else:
                        reputation = "AVERAGE_DEPLOYER"
                    
                    logger.info(f"✅ Deployer analysis: {reputation} ({total_tokens} tokens)")
                    
                    return {
                        "deployer_address": deployer,
                        "reputation": reputation,
                        "total_tokens_deployed": total_tokens,
                        "successful_tokens": successful_tokens,
                        "rugged_tokens": rugged_tokens,
                        "success_rate": successful_tokens / total_tokens if total_tokens > 0 else 0,
                        "rug_rate": rugged_tokens / total_tokens if total_tokens > 0 else 0,
                        "data_source": "solana_tracker"
                    }
                else:
                    logger.warning(f"Failed to get deployer data: {resp.status}")
                    return {"reputation": "UNKNOWN", "reason": f"API error {resp.status}", "data_source": "api_error"}
                    
        except Exception as e:
            logger.error(f"Deployer analysis error: {e}")
            return {"reputation": "ERROR", "reason": str(e), "data_source": "error"}

    async def _get_first_buyers_analysis(self, token_address: str, session: aiohttp.ClientSession, headers: Dict) -> Dict:
        """Get first buyers using CORRECT endpoint /first-buyers/{token} with FIXED parsing"""
        try:
            # Rate limit compliance
            await self._rate_limit_delay(delay=1.5)
            
            # CORRECT ENDPOINT: /first-buyers/{token}
            url = f"{self.solana_tracker_base}/first-buyers/{token_address}"
            
            async with session.get(url, headers=headers, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    
                    # FIXED: Parse CORRECT response structure
                    if isinstance(data, list):
                        first_buyers = data
                    else:
                        first_buyers = data.get("buyers", data.get("data", []))
                    
                    if not first_buyers:
                        return {"data_source": "no_first_buyers", "first_buyers_count": 0, "first_buyer_concentration": 0}
                    
                    logger.info(f"✅ Retrieved {len(first_buyers)} first buyers")
                    
                    # FIXED: Analyze first buyer patterns with proper field access
                    total_bought = 0
                    valid_buyers = []
                    
                    for buyer in first_buyers:
                        if isinstance(buyer, dict):
                            amount = self._safe_float(buyer.get("amount", buyer.get("value", 0)))
                            wallet = buyer.get("wallet", buyer.get("address", buyer.get("owner", "")))
                            timestamp = self._safe_int(buyer.get("timestamp", buyer.get("time", 0)))
                            
                            total_bought += amount
                            valid_buyers.append({
                                "address": wallet,
                                "amount": amount,
                                "timestamp": timestamp
                            })
                    
                    first_buyer_concentration = (valid_buyers[0]["amount"] / total_bought * 100) if total_bought > 0 and valid_buyers else 0
                    
                    # Check for suspicious patterns
                    suspicious_patterns = []
                    if len(valid_buyers) < 5:
                        suspicious_patterns.append("Very few first buyers - possible insider knowledge")
                    if first_buyer_concentration > 50:
                        suspicious_patterns.append("Single buyer dominated early purchases")
                    
                    return {
                        "first_buyers_count": len(valid_buyers),
                        "first_buyer_concentration": round(first_buyer_concentration, 2),
                        "total_early_volume": total_bought,
                        "suspicious_patterns": suspicious_patterns,
                        "first_buyers": valid_buyers[:10],  # Top 10 for analysis
                        "data_source": "solana_tracker"
                    }
                else:
                    logger.warning(f"Failed to get first buyers data: {resp.status}")
                    return {"data_source": "solana_tracker_error", "error": f"Status {resp.status}", "first_buyers_count": 0, "first_buyer_concentration": 0}
                    
        except Exception as e:
            logger.error(f"First buyers analysis error: {e}")
            return {"data_source": "error", "error": str(e), "first_buyers_count": 0, "first_buyer_concentration": 0}

    async def _get_performance_analysis(self, token_address: str, session: aiohttp.ClientSession, headers: Dict) -> Dict:
        """Get performance analysis using available endpoints with FIXED parsing"""
        try:
            # Rate limit compliance
            await self._rate_limit_delay(delay=1.5)
            
            # Use top-traders endpoint to get performance data
            url = f"{self.solana_tracker_base}/top-traders/{token_address}"
            
            async with session.get(url, headers=headers, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    
                    # FIXED: Parse response structure
                    if isinstance(data, list):
                        traders = data
                    else:
                        traders = data.get("traders", data.get("data", []))
                    
                    if not traders:
                        return self._create_fallback_performance_data()
                    
                    # FIXED: Calculate performance metrics with proper field access
                    total_traders = len(traders)
                    profitable_traders = 0
                    total_pnl = 0
                    
                    for trader in traders:
                        if isinstance(trader, dict):
                            pnl = self._safe_float(trader.get("pnl", trader.get("profit", 0)))
                            total_pnl += pnl
                            if pnl > 0:
                                profitable_traders += 1
                    
                    avg_pnl = total_pnl / total_traders if total_traders > 0 else 0
                    
                    logger.info(f"✅ Retrieved performance data: {profitable_traders}/{total_traders} profitable")
                    
                    return {
                        "total_traders": total_traders,
                        "profitable_traders": profitable_traders,
                        "profit_rate": profitable_traders / total_traders if total_traders > 0 else 0.5,
                        "average_pnl": avg_pnl,
                        "data_source": "solana_tracker"
                    }
                else:
                    return self._create_fallback_performance_data()
                    
        except Exception as e:
            logger.error(f"Performance analysis error: {e}")
            return self._create_fallback_performance_data()

    def _calculate_enhanced_indicators_v61(self, token_info: Dict, holders_data: Dict, 
                                          trading_data: Dict, deployer_analysis: Dict, 
                                          first_buyers_data: Dict, performance_data: Dict) -> ScamIndicators:
        """Calculate enhanced indicators v6.1 with SolanaTracker risk data integration"""
        indicators = ScamIndicators()
        
        # Authority analysis
        indicators.mint_enabled = token_info.get("mint_authority") is not None
        indicators.freeze_enabled = token_info.get("freeze_authority") is not None
        
        # Holder concentration
        indicators.top_holder_percent = holders_data.get("top_1_percent", 0)
        indicators.top_10_holders_percent = holders_data.get("top_10_percent", 0)
        
        # Bundle detection
        indicators.bundle_detected = token_info.get("bundleId") is not None
        
        # Deployer reputation
        indicators.deployer_reputation = deployer_analysis.get("reputation", "UNKNOWN")
        
        # First buyer concentration
        indicators.first_buyer_concentration = first_buyers_data.get("first_buyer_concentration", 0)
        
        # SolanaTracker risk data integration
        indicators.solana_tracker_risk_score = token_info.get("risk_score", 0)
        indicators.rugged_status = token_info.get("rugged_status", False)
        
        # Token age
        created_at = token_info.get("created_at", 0)
        if created_at:
            age_hours = int((datetime.now().timestamp() * 1000 - created_at) / (1000 * 60 * 60))
            indicators.token_age_hours = max(0, age_hours)  # Ensure non-negative
        
        # Liquidity analysis
        market_cap = token_info.get("market_cap", 0)
        liquidity = token_info.get("liquidity", 0)
        indicators.liquidity_percent = (liquidity / market_cap * 100) if market_cap > 0 else 0
        
        # Trading health
        buy_trades = trading_data.get("buy_trades", 0)
        sell_trades = trading_data.get("sell_trades", 0)
        indicators.can_sell = sell_trades > 0 or buy_trades < 10
        
        # Calculate overall risk score with SolanaTracker integration
        risk_score = self._calculate_risk_score_v61(indicators, deployer_analysis, performance_data)
        indicators.overall_risk_score = risk_score
        
        # Determine risk level
        if risk_score >= 80 or indicators.rugged_status:
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
        
        return indicators

    def _calculate_risk_score_v61(self, indicators: ScamIndicators, deployer_analysis: Dict, performance_data: Dict) -> int:
        """Enhanced risk score calculation v6.1 with SolanaTracker risk integration"""
        risk_score = 0
        
        # SolanaTracker risk score integration (HIGH WEIGHT)
        if indicators.rugged_status:
            risk_score += 50  # Major penalty for rugged status
        
        tracker_risk = indicators.solana_tracker_risk_score
        if tracker_risk > 5000:  # Very high risk from SolanaTracker
            risk_score += 40
        elif tracker_risk > 1000:
            risk_score += 25
        elif tracker_risk > 100:
            risk_score += 15
        elif tracker_risk > 10:
            risk_score += 5
        
        # Authority risks
        if indicators.mint_enabled:
            risk_score += 40
        if indicators.freeze_enabled:
            risk_score += 30
        
        # Concentration risks
        if indicators.top_holder_percent > 50:
            risk_score += 35
        elif indicators.top_holder_percent > 30:
            risk_score += 28
        elif indicators.top_holder_percent > 20:
            risk_score += 20
        elif indicators.top_holder_percent > 15:
            risk_score += 15
        elif indicators.top_holder_percent > 10:
            risk_score += 8
        
        # Bundle detection
        if indicators.bundle_detected:
            risk_score += 25
        
        # Deployer reputation
        deployer_rep = indicators.deployer_reputation
        if deployer_rep == "HIGH_RISK_DEPLOYER":
            risk_score += 30
        elif deployer_rep == "MODERATE_RISK_DEPLOYER":
            risk_score += 20
        elif deployer_rep == "NEW_DEPLOYER":
            risk_score += 10
        elif deployer_rep == "EXPERIENCED_DEPLOYER":
            risk_score -= 5
        
        # First buyer concentration
        if indicators.first_buyer_concentration > 70:
            risk_score += 20
        elif indicators.first_buyer_concentration > 50:
            risk_score += 15
        elif indicators.first_buyer_concentration > 30:
            risk_score += 10
        
        # Performance analysis
        profit_rate = performance_data.get("profit_rate", 0.5)
        if profit_rate < 0.2:
            risk_score += 15
        elif profit_rate < 0.4:
            risk_score += 8
        elif profit_rate > 0.7:
            risk_score -= 5
        
        # Token age
        if indicators.token_age_hours < 6:
            risk_score += 15
        elif indicators.token_age_hours < 24:
            risk_score += 10
        elif indicators.token_age_hours < 168:
            risk_score += 5
        
        # Liquidity
        if indicators.liquidity_percent < 1:
            risk_score += 20
        elif indicators.liquidity_percent < 3:
            risk_score += 10
        elif indicators.liquidity_percent < 5:
            risk_score += 5
        
        return max(0, min(100, risk_score))

    # FOCUSED GROK ANALYSIS - X/Twitter sentiment only
    async def _grok_focused_sentiment_analysis_v61(self, token_address: str, token_info: Dict, 
                                                   indicators: ScamIndicators, session: aiohttp.ClientSession) -> Dict:
        """FOCUSED Grok analysis - X/Twitter sentiment search for contract address only"""
        try:
            if not self.xai_key or self.xai_key == 'your-xai-api-key-here':
                return {"available": False, "reason": "no_api_key"}
            
            symbol = token_info.get('symbol', 'UNKNOWN')
            name = token_info.get('name', 'Unknown Token')
            market_cap = token_info.get('market_cap', 0)
            
            # FOCUSED prompt - just X/Twitter sentiment for the contract address
            focused_prompt = f"""
Search X/Twitter for recent posts mentioning this Solana token contract address: {token_address}

Also search for: ${symbol} token, {name}

FOCUS ON:
- What are people saying about this token on X/Twitter?
- Are there any warning signs or red flags mentioned?
- Is there positive community sentiment or concerns?
- Any mentions of scams, rugs, or safety issues?
- Overall community vibe - bullish, bearish, or neutral?

CURRENT TOKEN INFO:
- Symbol: ${symbol}
- Name: {name}
- Market Cap: ${market_cap:,.0f} USD
- Contract: {token_address}

Provide a BRIEF summary of X/Twitter sentiment and any notable mentions.

VERDICT OPTIONS:
- COMMUNITY_BULLISH: Positive sentiment, no warnings
- COMMUNITY_NEUTRAL: Mixed or limited sentiment
- COMMUNITY_CONCERNS: Some warning signs detected
- COMMUNITY_BEARISH: Negative sentiment or warnings
- NO_MENTIONS: Little to no discussion found
"""
            
            # Simple search parameters focused on X/Twitter
            search_params = {
                "mode": "on",
                "sources": [{"type": "x"}],
                "max_search_results": 15,
                "from_date": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
                "return_citations": True
            }
            
            logger.info(f"🧠 Grok focused sentiment analysis for {symbol}...")
            grok_response = await self._call_grok_api_enhanced(focused_prompt, search_params, session)
            
            if grok_response:
                parsed_analysis = self._parse_focused_grok_sentiment_v61(
                    grok_response.get('content', ''), symbol, token_address
                )
                logger.info(f"🧠 Grok sentiment analysis complete: {parsed_analysis.get('verdict', 'UNKNOWN')}")
                return {
                    "available": True,
                    "raw_response": grok_response.get('content', ''),
                    "parsed_analysis": parsed_analysis,
                    "confidence": parsed_analysis.get('confidence', 0.7),
                    "analysis_type": "focused_x_twitter_sentiment_v61",
                    "citations": grok_response.get('citations', [])
                }
            else:
                return {"available": False, "reason": "api_failed"}
                
        except Exception as e:
            logger.error(f"Focused Grok sentiment analysis failed: {e}")
            return {"available": False, "reason": "error", "error": str(e)}

    async def _call_grok_api_enhanced(self, prompt: str, search_params: Dict, session: aiohttp.ClientSession) -> Dict:
        """Enhanced Grok API call"""
        try:
            payload = {
                "model": "grok-3-latest",
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are a cryptocurrency sentiment analyst focused on X/Twitter community discussions."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "search_parameters": search_params,
                "max_tokens": 1000,  # Reduced for focused analysis
                "temperature": 0.3,
                "stream": False
            }
            
            headers = {
                "Authorization": f"Bearer {self.xai_key}",
                "Content-Type": "application/json"
            }
            
            async with session.post("https://api.x.ai/v1/chat/completions", 
                                json=payload, headers=headers, timeout=60) as resp:
                
                if resp.status == 200:
                    result = await resp.json()
                    
                    if 'choices' in result and len(result['choices']) > 0:
                        choice = result['choices'][0]
                        content = choice.get('message', {}).get('content', '')
                        citations = choice.get('citations', [])
                        
                        logger.info(f"🧠 Grok API success: {len(content)} characters")
                        
                        return {
                            'content': content,
                            'citations': citations,
                            'usage': result.get('usage', {})
                        }
                    else:
                        logger.error(f"🧠 Grok API unexpected response: {result}")
                        return None
                else:
                    error_text = await resp.text()
                    logger.error(f"🧠 Grok API error {resp.status}: {error_text}")
                    return None
                    
        except Exception as e:
            logger.error(f"🧠 Grok API call failed: {e}")
            return None

    def _parse_focused_grok_sentiment_v61(self, grok_response: str, symbol: str, token_address: str) -> Dict:
        """Parse focused Grok sentiment analysis response"""
        try:
            # Extract verdict
            verdict_patterns = [
                r'(?:VERDICT|SENTIMENT):\s*([A-Z_]+)',
                r'\*\*([A-Z_]+)\*\*',
                r'(COMMUNITY_BULLISH|COMMUNITY_NEUTRAL|COMMUNITY_CONCERNS|COMMUNITY_BEARISH|NO_MENTIONS)'
            ]
            
            verdict = "COMMUNITY_NEUTRAL"
            for pattern in verdict_patterns:
                match = re.search(pattern, grok_response, re.IGNORECASE)
                if match:
                    verdict = match.group(1).upper()
                    break
            
            # Extract mentions and sentiment indicators
            positive_indicators = []
            negative_indicators = []
            
            # Look for positive sentiment
            positive_patterns = ['bullish', 'positive', 'good', 'safe', 'legitimate', 'strong community']
            for pattern in positive_patterns:
                if pattern in grok_response.lower():
                    positive_indicators.append(f"Mentions: {pattern}")
            
            # Look for negative sentiment
            negative_patterns = ['bearish', 'scam', 'rug', 'warning', 'avoid', 'dump', 'suspicious']
            for pattern in negative_patterns:
                if pattern in grok_response.lower():
                    negative_indicators.append(f"Warning: {pattern}")
            
            # Count mentions
            contract_mentions = len(re.findall(token_address[:8], grok_response, re.IGNORECASE))
            symbol_mentions = len(re.findall(f'\\${symbol}', grok_response, re.IGNORECASE))
            
            # Calculate confidence based on response quality
            confidence = 0.6  # Base confidence
            if len(grok_response) > 200:
                confidence += 0.1
            if contract_mentions > 0 or symbol_mentions > 0:
                confidence += 0.1
            if positive_indicators or negative_indicators:
                confidence += 0.1
            
            return {
                "verdict": verdict,
                "confidence": min(0.9, confidence),
                "positive_sentiment": positive_indicators,
                "negative_sentiment": negative_indicators,
                "contract_mentions": contract_mentions,
                "symbol_mentions": symbol_mentions,
                "community_summary": grok_response[:300] + "..." if len(grok_response) > 300 else grok_response,
                "analysis_type": "focused_sentiment_v61"
            }
            
        except Exception as e:
            logger.error(f"Focused Grok parsing failed: {e}")
            return {
                "verdict": "COMMUNITY_NEUTRAL",
                "confidence": 0.5,
                "positive_sentiment": [],
                "negative_sentiment": [f"Analysis parsing failed: {str(e)}"],
                "contract_mentions": 0,
                "symbol_mentions": 0,
                "community_summary": "Failed to parse sentiment analysis",
                "analysis_type": "fallback_v61"
            }

    # Bundle and manipulation detection methods - updated for v6.1
    async def _detect_enhanced_bundles_v61(self, token_info: Dict, holders_data: Dict, 
                                          trading_data: Dict, first_buyers_data: Dict) -> Dict:
        """Enhanced bundle detection v6.1"""
        try:
            bundle_detected = token_info.get('bundleId') is not None
            bundle_id = token_info.get('bundleId', '')
            
            first_buyer_concentration = first_buyers_data.get('first_buyer_concentration', 0)
            first_buyers_count = first_buyers_data.get('first_buyers_count', 0)
            
            risk_factors = []
            if bundle_detected:
                risk_factors.append(f"Bundle ID detected: {bundle_id}")
            
            if first_buyer_concentration > 70:
                risk_factors.append(f"Extreme first buyer concentration: {first_buyer_concentration:.1f}%")
            
            if first_buyers_count < 5 and first_buyers_count > 0:
                risk_factors.append("Very few first buyers - possible coordination")
            
            # Check SolanaTracker risk warnings for bundle-related issues
            risk_warnings = token_info.get('risk_warnings', [])
            for warning in risk_warnings:
                if isinstance(warning, dict) and 'name' in warning:
                    warning_name = warning['name'].lower()
                    if 'bundle' in warning_name or 'coordinate' in warning_name:
                        risk_factors.append(f"SolanaTracker warning: {warning['name']}")
            
            # Calculate bundle risk score
            bundle_risk_score = 0
            if bundle_detected:
                bundle_risk_score += 40
            if first_buyer_concentration > 50:
                bundle_risk_score += 30
            if first_buyers_count < 5 and first_buyers_count > 0:
                bundle_risk_score += 20
            
            # Add SolanaTracker risk component
            tracker_score = token_info.get('risk_score', 0)
            if tracker_score > 1000:
                bundle_risk_score += 20
            
            risk_level = "CRITICAL" if bundle_risk_score >= 70 else "HIGH" if bundle_risk_score >= 50 else "MEDIUM" if bundle_risk_score >= 25 else "LOW"
            
            return {
                "bundle_detected": bundle_detected,
                "bundle_id": bundle_id,
                "bundle_risk_score": min(100, bundle_risk_score),
                "risk_level": risk_level,
                "risk_factors": risk_factors,
                "clusters_found": 1 if bundle_detected else 0,
                "high_risk_clusters": 1 if bundle_detected and bundle_risk_score >= 70 else 0,
                "bundled_percentage": first_buyer_concentration if bundle_detected else 0,
                "data_source": "solana_tracker_v61"
            }
            
        except Exception as e:
            logger.error(f"Bundle detection v6.1 failed: {e}")
            return self._create_fallback_bundle_detection()

    async def _detect_basic_bundles_v61(self, holders_data: Dict, first_buyers_data: Dict) -> Dict:
        """Basic bundle detection for Express mode v6.1"""
        risk_score = 0
        bundle_indicators = []
        
        first_buyer_concentration = first_buyers_data.get('first_buyer_concentration', 0)
        if first_buyer_concentration > 60:
            bundle_indicators.append(f"High first buyer concentration: {first_buyer_concentration:.1f}%")
            risk_score += 30
        
        top_1_percent = holders_data.get('top_1_percent', 0)
        if top_1_percent > 40:
            bundle_indicators.append(f"Extreme top holder concentration: {top_1_percent:.1f}%")
            risk_score += 25
        
        risk_level = "HIGH" if risk_score >= 50 else "MEDIUM" if risk_score >= 25 else "LOW"
        
        return {
            "bundle_risk_score": min(100, risk_score),
            "risk_level": risk_level,
            "bundle_indicators": bundle_indicators,
            "clusters_found": 1 if risk_score >= 50 else 0,
            "high_risk_clusters": 1 if risk_score >= 70 else 0,
            "bundled_percentage": first_buyer_concentration,
            "data_source": "express_mode_v61"
        }

    # Continue with remaining methods...
    def _detect_basic_manipulation(self, trading_data: Dict, performance_data: Dict) -> Dict:
        """Basic manipulation detection for Express mode"""
        manipulation_indicators = []
        risk_score = 0
        
        wash_score = trading_data.get('wash_trading_score', 0)
        if wash_score > 50:
            manipulation_indicators.append(f"High wash trading: {wash_score:.0f}%")
            risk_score += 35
        
        profit_rate = performance_data.get('profit_rate', 0.5)
        if profit_rate < 0.3:
            manipulation_indicators.append(f"Low trader profitability: {profit_rate*100:.0f}%")
            risk_score += 25
        
        return {
            "manipulation_score": min(100, risk_score),
            "manipulation_level": "HIGH" if risk_score >= 50 else "MEDIUM" if risk_score >= 25 else "LOW",
            "manipulation_indicators": manipulation_indicators,
            "data_source": "express_mode_v61"
        }

    async def _detect_manipulation_patterns(self, token_address: str, trading_data: Dict, 
                                          performance_data: Dict, session: aiohttp.ClientSession, headers: Dict) -> Dict:
        """Detect manipulation patterns v6.1"""
        try:
            manipulation_indicators = []
            manipulation_score = 0
            
            # Wash trading analysis
            wash_score = trading_data.get('wash_trading_score', 0)
            if wash_score > 60:
                manipulation_indicators.append(f"High wash trading detected: {wash_score:.0f}%")
                manipulation_score += 30
            elif wash_score > 30:
                manipulation_indicators.append(f"Moderate wash trading risk: {wash_score:.0f}%")
                manipulation_score += 15
            
            # Trade frequency analysis
            unique_traders = trading_data.get('unique_traders', 0)
            total_trades = trading_data.get('total_trades', 0)
            if total_trades > 0 and unique_traders > 0:
                trades_per_trader = total_trades / unique_traders
                if trades_per_trader > 10:
                    manipulation_indicators.append(f"High trade frequency per trader: {trades_per_trader:.1f}")
                    manipulation_score += 20
            
            # Profit distribution analysis
            profit_rate = performance_data.get('profit_rate', 0.5)
            if profit_rate < 0.2:
                manipulation_indicators.append(f"Very low trader profitability: {profit_rate*100:.0f}%")
                manipulation_score += 25
            
            # Overall manipulation assessment
            if manipulation_score >= 70:
                manipulation_level = "EXTREME"
            elif manipulation_score >= 50:
                manipulation_level = "HIGH"
            elif manipulation_score >= 30:
                manipulation_level = "MEDIUM"
            else:
                manipulation_level = "LOW"
            
            return {
                "manipulation_score": min(100, manipulation_score),
                "manipulation_level": manipulation_level,
                "manipulation_indicators": manipulation_indicators,
                "data_source": "analysis_v61"
            }
            
        except Exception as e:
            logger.error(f"Manipulation detection v6.1 failed: {e}")
            return {
                "manipulation_score": 50,
                "manipulation_level": "UNKNOWN",
                "manipulation_indicators": ["Analysis failed"],
                "data_source": "error"
            }

    # Scoring methods updated for v6.1
    def _calculate_revolutionary_galaxy_score_v61(self, indicators: ScamIndicators, trading_data: Dict, 
                                                 bundle_detection: Dict, manipulation_detection: Dict, 
                                                 deployer_analysis: Dict, grok_analysis: Dict) -> Tuple[int, str, float]:
        """Calculate Galaxy Brain score v6.1 with improved SolanaTracker integration"""
        
        base_risk = indicators.overall_risk_score
        bundle_risk = bundle_detection.get('bundle_risk_score', 0) * 0.4
        manipulation_risk = manipulation_detection.get('manipulation_score', 0) * 0.3
        deployer_risk = self._calculate_deployer_risk(deployer_analysis)
        wash_score = trading_data.get('wash_trading_score', 0)
        trading_risk = wash_score * 0.2
        
        # Enhanced Grok adjustment for focused sentiment
        grok_adjustment = 0
        if grok_analysis.get("available") and grok_analysis.get("parsed_analysis"):
            verdict = grok_analysis["parsed_analysis"].get("verdict", "COMMUNITY_NEUTRAL")
            
            if verdict == "COMMUNITY_BULLISH":
                grok_adjustment = -15
            elif verdict == "COMMUNITY_NEUTRAL":
                grok_adjustment = 0
            elif verdict == "COMMUNITY_CONCERNS":
                grok_adjustment = +10
            elif verdict == "COMMUNITY_BEARISH":
                grok_adjustment = +20
            elif verdict == "NO_MENTIONS":
                grok_adjustment = +5  # Slight penalty for no community discussion
        
        galaxy_score = base_risk + bundle_risk + manipulation_risk + deployer_risk + trading_risk + grok_adjustment
        galaxy_score = max(0, min(100, galaxy_score))
        
        # Determine severity with SolanaTracker rugged status override
        if indicators.rugged_status:
            severity = "EXTREME_DANGER"
        elif galaxy_score >= 85:
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
        
        # Calculate confidence
        confidence_factors = [
            0.9,  # Solana Tracker base confidence
            0.85 if trading_data.get('data_source') == 'solana_tracker_live' else 0.7,
            0.8 if deployer_analysis.get('data_source') == 'solana_tracker' else 0.6,
            grok_analysis.get("parsed_analysis", {}).get("confidence", 0.7) if grok_analysis.get("available") else 0.6
        ]
        
        confidence = sum(confidence_factors) / len(confidence_factors)
        
        logger.info(f"🧠 Galaxy Brain v6.1: {int(galaxy_score)}/100 ({severity}) - Confidence: {confidence:.0%}")
        
        return int(galaxy_score), severity, round(confidence, 2)

    def _calculate_express_score_v61(self, indicators: ScamIndicators, holders_data: Dict, 
                                    trading_data: Dict, deployer_analysis: Dict, grok_analysis: Dict) -> Tuple[int, str, float]:
        """Express mode scoring v6.1"""
        
        base_risk = indicators.overall_risk_score
        
        if indicators.bundle_detected:
            base_risk += 20
        
        deployer_risk = self._calculate_deployer_risk(deployer_analysis) * 0.8
        wash_penalty = trading_data.get('wash_trading_score', 0) * 0.3
        
        # Focused Grok adjustment (simplified)
        grok_adjustment = 0
        if grok_analysis.get("available"):
            verdict = grok_analysis.get("parsed_analysis", {}).get("verdict", "COMMUNITY_NEUTRAL")
            if verdict == "COMMUNITY_BULLISH":
                grok_adjustment = -10
            elif verdict == "COMMUNITY_BEARISH":
                grok_adjustment = +15
        
        express_score = base_risk + deployer_risk + wash_penalty + grok_adjustment
        express_score = max(0, min(100, express_score))
        
        # Severity assessment with rugged status check
        if indicators.rugged_status:
            severity = "CRITICAL_RISK"
        elif express_score >= 80:
            severity = "CRITICAL_RISK"
        elif express_score >= 60:
            severity = "HIGH_RISK"
        elif express_score >= 40:
            severity = "MEDIUM_RISK"
        elif express_score >= 20:
            severity = "LOW_RISK"
        else:
            severity = "MINIMAL_RISK"
        
        confidence = 0.85
        
        return int(express_score), severity, confidence

    # Risk vector generation updated for v6.1
    def _generate_enhanced_risk_vectors_v61(self, indicators: ScamIndicators, trading_data: Dict, 
                                           bundle_detection: Dict, manipulation_detection: Dict, 
                                           deployer_analysis: Dict, grok_analysis: Dict) -> List[Dict]:
        """Generate comprehensive risk vectors v6.1"""
        risk_vectors = []
        
        # SolanaTracker rugged status (HIGHEST PRIORITY)
        if indicators.rugged_status:
            risk_vectors.append({
                "category": "SolanaTracker Alert",
                "risk_type": "Rugged Status Confirmed",
                "severity": "CRITICAL",
                "impact": "SolanaTracker has flagged this token as rugged",
                "likelihood": "CONFIRMED",
                "mitigation": "AVOID - Token confirmed as rugged by SolanaTracker"
            })
        
        # SolanaTracker high risk score
        if indicators.solana_tracker_risk_score > 1000:
            risk_vectors.append({
                "category": "SolanaTracker Risk",
                "risk_type": f"High Risk Score ({indicators.solana_tracker_risk_score})",
                "severity": "HIGH",
                "impact": "SolanaTracker risk algorithms detect high risk patterns",
                "likelihood": "HIGH",
                "mitigation": "Exercise extreme caution - multiple risk factors detected"
            })
        
        # Authority risks
        if indicators.mint_enabled:
            risk_vectors.append({
                "category": "Authority Risk",
                "risk_type": "Mint Authority Active",
                "severity": "CRITICAL",
                "impact": "Developer can create unlimited new tokens at any time",
                "likelihood": "HIGH",
                "mitigation": "AVOID - Critical red flag confirmed by blockchain data"
            })
        
        if indicators.freeze_enabled:
            risk_vectors.append({
                "category": "Authority Risk",
                "risk_type": "Freeze Authority Active",
                "severity": "CRITICAL",
                "impact": "Developer can halt all token transfers instantly",
                "likelihood": "MEDIUM",
                "mitigation": "EXTREME RISK - Avoid or use minimal amounts only"
            })
        
        # Bundle risks
        if indicators.bundle_detected:
            risk_vectors.append({
                "category": "Bundle Detection",
                "risk_type": "Coordinated Bundle Launch Detected",
                "severity": "HIGH",
                "impact": "Token launched as part of coordinated bundle",
                "likelihood": "CONFIRMED",
                "mitigation": "High risk of coordinated manipulation - exercise extreme caution"
            })
        
        # Deployer reputation risks
        deployer_rep = indicators.deployer_reputation
        if deployer_rep == "HIGH_RISK_DEPLOYER":
            rug_rate = deployer_analysis.get('rug_rate', 0)
            risk_vectors.append({
                "category": "Deployer Risk",
                "risk_type": f"High-Risk Deployer ({rug_rate*100:.0f}% Rug Rate)",
                "severity": "CRITICAL",
                "impact": f"Deployer has rugged {rug_rate*100:.0f}% of previous tokens",
                "likelihood": "HIGH",
                "mitigation": "AVOID - Developer has established pattern of rug pulls"
            })
        
        # Concentration risks
        if indicators.top_holder_percent > 25:
            risk_vectors.append({
                "category": "Concentration Risk",
                "risk_type": f"Extreme Holder Concentration ({indicators.top_holder_percent:.1f}%)",
                "severity": "CRITICAL",
                "impact": "Single holder can crash price with large sell orders",
                "likelihood": "HIGH",
                "mitigation": "Monitor whale wallet for movement - very high dump risk"
            })
        
        # Manipulation risks
        manipulation_score = manipulation_detection.get('manipulation_score', 0)
        if manipulation_score >= 50:
            risk_vectors.append({
                "category": "Market Manipulation",
                "risk_type": "High Manipulation Risk",
                "severity": "HIGH",
                "impact": "Trading patterns suggest artificial price inflation and volume manipulation",
                "likelihood": "HIGH",
                "mitigation": "Avoid trading - high probability of manipulation"
            })
        
        # Community sentiment risks
        if grok_analysis.get("available") and grok_analysis.get("parsed_analysis"):
            verdict = grok_analysis["parsed_analysis"].get("verdict", "COMMUNITY_NEUTRAL")
            if verdict == "COMMUNITY_BEARISH":
                risk_vectors.append({
                    "category": "Community Sentiment",
                    "risk_type": "Negative Community Sentiment",
                    "severity": "MEDIUM",
                    "impact": "X/Twitter community showing bearish or negative sentiment",
                    "likelihood": "MEDIUM",
                    "mitigation": "Monitor community discussions - sentiment may impact price"
                })
        
        return risk_vectors

    def _generate_basic_risk_vectors_v61(self, indicators: ScamIndicators, holders_data: Dict, 
                                        trading_data: Dict, deployer_analysis: Dict, grok_analysis: Dict) -> List[Dict]:
        """Generate basic risk vectors for Express mode v6.1"""
        risk_vectors = []
        
        # SolanaTracker rugged status (ALWAYS SHOW)
        if indicators.rugged_status:
            risk_vectors.append({
                "category": "SolanaTracker Alert",
                "risk_type": "Rugged Status",
                "severity": "CRITICAL",
                "impact": "Confirmed rugged by SolanaTracker",
                "likelihood": "CONFIRMED",
                "mitigation": "AVOID COMPLETELY"
            })
        
        # Critical authority risks
        if indicators.mint_enabled:
            risk_vectors.append({
                "category": "Authority Risk",
                "risk_type": "Mint Authority Active",
                "severity": "CRITICAL",
                "impact": "Unlimited token creation possible",
                "likelihood": "HIGH",
                "mitigation": "Wait for mint renunciation"
            })
        
        if indicators.freeze_enabled:
            risk_vectors.append({
                "category": "Authority Risk", 
                "risk_type": "Freeze Authority Active",
                "severity": "CRITICAL",
                "impact": "Can halt all transfers",
                "likelihood": "MEDIUM",
                "mitigation": "Avoid until authority renounced"
            })
        
        # Bundle risk
        if indicators.bundle_detected:
            risk_vectors.append({
                "category": "Bundle Risk",
                "risk_type": "Coordinated Launch Detected",
                "severity": "HIGH",
                "impact": "Part of coordinated token deployment",
                "likelihood": "CONFIRMED",
                "mitigation": "High manipulation risk - use caution"
            })
        
        # Concentration risk
        if indicators.top_holder_percent > 20:
            risk_vectors.append({
                "category": "Concentration Risk",
                "risk_type": f"High Holder Concentration ({indicators.top_holder_percent:.1f}%)",
                "severity": "HIGH",
                "impact": "Major holder can significantly impact price",
                "likelihood": "HIGH",
                "mitigation": "Monitor whale activity closely"
            })
        
        return risk_vectors

    def _format_grok_ai_analysis_v61(self, grok_analysis: Dict) -> str:
        """Format Grok analysis for display v6.1"""
        if not grok_analysis.get("available"):
            return "🧠 Revolutionary Grok v6.1: Connect XAI API key for X/Twitter sentiment analysis."
        
        grok_data = grok_analysis.get("parsed_analysis", {})
        verdict = grok_data.get("verdict", "COMMUNITY_NEUTRAL")
        confidence = grok_data.get("confidence", 0.5)
        
        verdict_emojis = {
            "COMMUNITY_BULLISH": "📈🧠",
            "COMMUNITY_NEUTRAL": "➡️🧠",
            "COMMUNITY_CONCERNS": "⚠️🧠",
            "COMMUNITY_BEARISH": "📉🧠",
            "NO_MENTIONS": "🔍🧠"
        }
        
        emoji = verdict_emojis.get(verdict, "🧠")
        
        return f"{emoji} X/Twitter Sentiment: {verdict.replace('_', ' ')} • {confidence:.0%} confidence"

    # Helper methods remain the same with updated names
    def _safe_float(self, value) -> float:
        """Safely convert value to float"""
        try:
            if value is None:
                return 0.0
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                # Remove any non-numeric characters except decimal point and minus
                clean_value = ''.join(c for c in value if c.isdigit() or c in '.-')
                return float(clean_value) if clean_value else 0.0
            return 0.0
        except (ValueError, TypeError):
            return 0.0

    def _safe_int(self, value) -> int:
        """Safely convert value to int"""
        try:
            if value is None:
                return 0
            if isinstance(value, int):
                return value
            if isinstance(value, float):
                return int(value)
            if isinstance(value, str):
                # Remove any non-numeric characters except minus
                clean_value = ''.join(c for c in value if c.isdigit() or c == '-')
                return int(clean_value) if clean_value else 0
            return 0
        except (ValueError, TypeError):
            return 0

    async def _rate_limit_delay(self, delay: float = 1.0):
        """Ensure proper delay between API calls"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call
        
        if time_since_last_call < delay:
            wait_time = delay - time_since_last_call
            logger.info(f"⏱️ Rate limiting: waiting {wait_time:.2f}s")
            await asyncio.sleep(wait_time)
        
        self.last_api_call = time.time()

    def _create_fallback_token_data(self, token_address: str) -> Dict:
        """Create fallback token data when API fails"""
        return {
            "success": True,
            "token_info": {
                "address": token_address,
                "name": "Unknown Token",
                "symbol": "UNKNOWN",
                "decimals": 6,
                "supply": 0,
                "market_cap": 0,
                "price": 0,
                "liquidity": 0,
                "volume_24h": 0,
                "price_change_24h": 0,
                "holders": 0,
                "created_at": 0,
                "mint_authority": None,
                "freeze_authority": None,
                "bundleId": None,
                "market": "unknown",
                "deployer": None,
                "logo": None,
                "description": "",
                "website": None,
                "twitter": None,
                "telegram": None,
                "rugged_status": False,
                "risk_score": 0,
                "risk_warnings": []
            }
        }

    def _create_fallback_holders_data(self) -> Dict:
        """Create fallback holder data when API fails"""
        return {
            "top_1_percent": 15.0,  # Assume moderate concentration
            "top_5_percent": 35.0,
            "top_10_percent": 50.0,
            "top_holders": [],
            "total_holders": 0,
            "excluded_wallets_count": 0,
            "concentration_risk": "UNKNOWN",
            "data_source": "fallback_estimated"
        }

    def _create_fallback_trading_data(self) -> Dict:
        """Create fallback trading data when API fails"""
        return {
            "total_trades": 0,
            "unique_traders": 0,
            "buy_trades": 0,
            "sell_trades": 0,
            "buy_sell_ratio": 1.0,
            "total_volume_usd": 0,
            "avg_trade_size": 0,
            "wash_trading_score": 50.0,  # Assume moderate risk
            "trade_frequency": 0,
            "data_source": "fallback_estimated"
        }

    def _create_fallback_performance_data(self) -> Dict:
        """Create fallback performance data"""
        return {
            "total_traders": 0,
            "profitable_traders": 0,
            "profit_rate": 0.5,  # Assume neutral
            "average_pnl": 0,
            "data_source": "fallback_estimated"
        }

    def _create_fallback_bundle_detection(self) -> Dict:
        """Create fallback bundle detection data"""
        return {
            "bundle_detected": False,
            "bundle_id": "",
            "bundle_risk_score": 0,
            "risk_level": "LOW",
            "risk_factors": [],
            "clusters_found": 0,
            "high_risk_clusters": 0,
            "bundled_percentage": 0,
            "data_source": "fallback"
        }

    def _assess_concentration_risk(self, top_1: float, top_5: float) -> str:
        """Assess concentration risk level"""
        if top_1 > 30:
            return "CRITICAL"
        elif top_1 > 20 or top_5 > 60:
            return "HIGH"
        elif top_1 > 10 or top_5 > 40:
            return "MEDIUM"
        elif top_1 > 5:
            return "LOW"
        else:
            return "MINIMAL"

    def _calculate_deployer_risk(self, deployer_analysis: Dict) -> float:
        """Calculate risk score from deployer analysis"""
        reputation = deployer_analysis.get('reputation', 'UNKNOWN')
        
        risk_scores = {
            'HIGH_RISK_DEPLOYER': 30,
            'MODERATE_RISK_DEPLOYER': 20,
            'NEW_DEPLOYER': 10,
            'FIRST_TOKEN': 8,
            'AVERAGE_DEPLOYER': 5,
            'EXPERIENCED_DEPLOYER': -5,
            'UNKNOWN': 15
        }
        
        return risk_scores.get(reputation, 15)

    # Frontend data formatting methods
    def _format_holder_analysis_for_frontend(self, holders_data: Dict) -> Dict:
        """Format holder analysis data for frontend display"""
        return {
            "top_1_percent": holders_data.get("top_1_percent", 0),
            "top_5_percent": holders_data.get("top_5_percent", 0),
            "top_10_percent": holders_data.get("top_10_percent", 0),
            "top_holders": holders_data.get("top_holders", []),
            "total_holders": holders_data.get("total_holders", 0),
            "excluded_wallets_count": holders_data.get("excluded_wallets_count", 0),
            "concentration_risk": holders_data.get("concentration_risk", "UNKNOWN"),
            "data_source": holders_data.get("data_source", "unknown")
        }

    def _create_transaction_analysis(self, trading_data: Dict, performance_data: Dict) -> Dict:
        """Create transaction analysis for frontend"""
        return {
            "total_transactions": trading_data.get("total_trades", 0),
            "unique_traders": trading_data.get("unique_traders", 0),
            "avg_transaction_size": trading_data.get("avg_trade_size", 0),
            "transaction_frequency": trading_data.get("trade_frequency", 0),
            "data_source": trading_data.get("data_source", "unknown")
        }

    def _create_suspicious_activity_analysis(self, trading_data: Dict, manipulation_detection: Dict) -> Dict:
        """Create suspicious activity analysis for frontend"""
        return {
            "transaction_health_score": 100 - trading_data.get("wash_trading_score", 0),
            "wash_trading_score": trading_data.get("wash_trading_score", 0),
            "insider_activity_score": manipulation_detection.get("manipulation_score", 0),
            "farming_indicators": manipulation_detection.get("manipulation_indicators", []),
            "suspicious_patterns": manipulation_detection.get("manipulation_indicators", []),
            "data_source": manipulation_detection.get("data_source", "unknown")
        }

    def _create_liquidity_analysis(self, token_info: Dict, trading_data: Dict) -> Dict:
        """Create liquidity analysis for frontend"""
        market_cap = token_info.get("market_cap", 0)
        liquidity = token_info.get("liquidity", 0)
        volume_24h = token_info.get("volume_24h", 0)
        
        liquidity_ratio = (liquidity / market_cap * 100) if market_cap > 0 else 0
        volume_to_liquidity = (volume_24h / liquidity) if liquidity > 0 else 0
        
        # Assess liquidity risk
        if liquidity_ratio < 1:
            liquidity_risk = "CRITICAL"
        elif liquidity_ratio < 3:
            liquidity_risk = "HIGH"
        elif liquidity_ratio < 5:
            liquidity_risk = "MEDIUM"
        else:
            liquidity_risk = "LOW"
        
        return {
            "liquidity_usd": liquidity,
            "liquidity_ratio": liquidity_ratio,
            "volume_to_liquidity": volume_to_liquidity,
            "liquidity_risk": liquidity_risk,
            "is_locked": False,  # Would need additional API call
            "lock_duration": "Unknown",
            "dex": token_info.get("market", "Unknown"),
            "slippage_1k": 0,  # Would need additional calculation
            "slippage_10k": 0,  # Would need additional calculation
            "data_source": "calculated_from_tracker"
        }


# Usage example with FIXED implementation v6.1
async def main():
    """Example usage of FIXED Enhanced Rug Checker v6.1 with proper market cap, filtered wallets, and focused Grok"""
    
    # Your API keys
    solana_tracker_key = "e8cb0621-db95-4697-b227-e12097576964"  # Your actual key
    xai_key = "your-xai-api-key-here"  # Get from x.ai for Grok X/Twitter sentiment
    
    checker = EnhancedRugCheckerWithSolanaTracker(
        solana_tracker_key=solana_tracker_key,
        xai_key=xai_key
    )
    
    # Example token address
    token_address = "6ztpBm31cmBNPwa396ocmDfaWyKKY95Bu8T664QfCe7f"
    
    try:
        print(f"\n🧠 Starting FIXED Revolutionary Galaxy Brain v6.1 Analysis for {token_address}...")
        result = await checker.analyze_token(token_address, deep_analysis=True)
        
        if result["success"]:
            print(f"\n✅ Analysis Complete!")
            print(f"🧠 Galaxy Brain Score: {result['galaxy_brain_score']}/100")
            print(f"⚠️ Risk Level: {result['severity_level']}")
            print(f"🎯 Confidence: {result['confidence']:.0%}")
            print(f"\n{result['ai_analysis']}")
            
            # Show token info with FIXED market cap
            token_info = result['token_info']
            print(f"\n📊 Token: {token_info['name']} (${token_info['symbol']})")
            print(f"💰 Market Cap: ${token_info.get('market_cap', 0):,.0f}")  # FIXED
            print(f"💧 Liquidity: ${token_info.get('liquidity', 0):,.0f}")
            print(f"📈 24h Volume: ${token_info.get('volume_24h', 0):,.0f}")
            print(f"👥 Holders: {token_info.get('holders', 0):,}")
            print(f"🏗️ Market: {token_info.get('market', 'unknown')}")
            print(f"👤 Deployer Rep: {token_info.get('deployer_reputation', 'UNKNOWN')}")
            print(f"📦 Bundle: {'YES' if token_info.get('bundle_detected') else 'NO'}")
            print(f"🚨 Rugged: {'YES' if token_info.get('rugged_status') else 'NO'}")
            print(f"📊 SolanaTracker Risk: {token_info.get('solana_tracker_risk_score', 0)}")
            
            # Show FILTERED holder analysis
            holder_analysis = result['holder_analysis']
            print(f"\n🐋 Holder Analysis (Filtered):")
            print(f"   Top 1 Holder: {holder_analysis.get('top_1_percent', 0):.1f}%")
            print(f"   Top 5 Holders: {holder_analysis.get('top_5_percent', 0):.1f}%")
            print(f"   Top 10 Holders: {holder_analysis.get('top_10_percent', 0):.1f}%")
            print(f"   Excluded Wallets: {holder_analysis.get('excluded_wallets_count', 0)}")
            print(f"   Concentration Risk: {holder_analysis.get('concentration_risk', 'UNKNOWN')}")
            
            # Show trading analysis
            trading_analysis = result['trading_analysis']
            print(f"\n🔄 Trading Analysis:")
            print(f"   Unique Traders: {trading_analysis.get('unique_traders', 0)}")
            print(f"   Total Trades: {trading_analysis.get('total_trades', 0)}")
            print(f"   Wash Trading Score: {trading_analysis.get('wash_trading_score', 0):.1f}%")
            
            # Show FOCUSED Grok sentiment
            if result['grok_analysis'].get('available'):
                grok_data = result['grok_analysis']['parsed_analysis']
                print(f"\n🧠 X/Twitter Sentiment Analysis:")
                print(f"   Verdict: {grok_data.get('verdict', 'UNKNOWN')}")
                print(f"   Contract Mentions: {grok_data.get('contract_mentions', 0)}")
                print(f"   Symbol Mentions: {grok_data.get('symbol_mentions', 0)}")
                print(f"   Summary: {grok_data.get('community_summary', 'No summary')[:100]}...")
            
            # Show risk vectors
            risk_vectors = result.get('risk_vectors', [])
            if risk_vectors:
                print(f"\n⚠️ Top Risk Vectors:")
                for i, vector in enumerate(risk_vectors[:3]):
                    print(f"   {i+1}. {vector['severity']}: {vector['risk_type']}")
            
            print(f"\n⏱️ Analysis completed in {result['analysis_time_seconds']} seconds")
            print(f"📊 Data Sources: {result['data_sources']}")
            
        else:
            print(f"❌ Analysis failed: {result['error']}")
            if 'suggestions' in result:
                print("💡 Suggestions:")
                for suggestion in result['suggestions']:
                    print(f"   • {suggestion}")
                
    except Exception as e:
        print(f"💥 Error analyzing {token_address}: {e}")

if __name__ == "__main__":
    # Run the fixed example
    asyncio.run(main())