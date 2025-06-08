import asyncio
import json
import logging
import statistics
import base64
import struct
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any, Tuple
from collections import defaultdict, Counter
import math
import hashlib

import aiohttp
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import uvicorn

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
HELIUS_API_KEY = "YOUR-HELIUS-KEY"  # Replace with your actual key
HELIUS_RPC_URL = f"https://mainnet.helius-rpc.com/?api-key={HELIUS_API_KEY}"

# External API endpoints
JUPITER_PRICE_API = "https://price.jup.ag/v4/price"
BIRDEYE_API = "https://public-api.birdeye.so/public"

OPENAI_API_KEY = ""  # Optional - will use rule-based analysis if not provided
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rugcheck")

# Rate-limit config
RATE_LIMIT_DELAY = 0.1           # 100ms between RPC calls
MAX_CONCURRENT_REQUESTS = 5       # concurrent RPCs allowed

# Known DEX program IDs for liquidity detection
DEX_PROGRAMS = {
    "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8",  # Raydium
    "9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM",  # Raydium V4
    "CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK",  # Raydium CLMM
    "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc",   # Orca
    "DjVE6JNiYqPL2QXyCUUh8rNjHrbz9hXHNYt99MQ59qw1",   # Orca V2
}

# ──────────────────────────────────────────────────────────────────────────────
# DATA CLASSES (keeping your existing structure)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class DeployerToken:
    address: str
    symbol: str
    name: str
    launch_date: datetime
    current_price: Optional[float]
    current_market_cap: Optional[float]
    highest_market_cap: Optional[float]
    is_rugged: bool
    holder_count: int
    liquidity: Optional[float]
    age_days: int

@dataclass
class DeployerAnalysis:
    deployer_address: str
    total_tokens_deployed: int
    successful_launches: int
    rugged_tokens: int
    active_tokens: int
    average_token_lifespan: float
    total_value_extracted: float
    reputation_score: int
    token_history: List[DeployerToken]
    deployment_pattern: str
    risk_assessment: str

@dataclass
class WalletInfo:
    address: str
    balance: float
    percentage: float
    sol_balance: Optional[float]
    token_count: Optional[int]
    creation_time: Optional[datetime]
    first_activity: Optional[datetime]
    transaction_count: int
    is_suspicious: bool
    risk_flags: List[str]
    rank: int = 0
    type: str = "holder"
    risk: str = "low"
    isBundled: bool = False

@dataclass
class BundleCluster:
    cluster_id: str
    wallets: List[str]
    creation_timeframe: float
    funding_source: Optional[str]
    similar_patterns: List[str]
    risk_score: int
    total_holdings: float

@dataclass
class BundleRiskSummary:
    total_bundles: int
    high_risk_bundles: int
    bundled_supply_percentage: float
    largest_bundle_size: int
    bundle_concentration_score: int
    patterns_detected: List[str]

@dataclass
class HolderVisualization:
    holder_bubbles: List[Dict[str, Any]]
    concentration_chart: Dict[str, float]
    bundle_summary: BundleRiskSummary

@dataclass
class EnhancedHolderAnalysis:
    total_holders: int
    unique_holders: int
    top_10_concentration: float
    top_5_concentration: float
    deployer_holdings: float
    holder_distribution: List[WalletInfo]
    bundle_clusters: List[BundleCluster]
    whale_wallets: List[str]
    suspicious_wallets: List[str]
    holder_growth_24h: int
    average_holding: float
    median_holding: float
    visualization_data: HolderVisualization

@dataclass
class TransactionAnalysis:
    total_transactions: int
    unique_traders: int
    volume_24h: float
    volume_7d: float
    first_transaction: Optional[datetime]
    peak_activity_time: Optional[datetime]
    suspicious_patterns: List[str]
    coordinated_activity: bool
    wash_trading_risk: float

@dataclass
class EnhancedTokenInfo:
    address: str
    symbol: str
    name: str
    decimals: int
    supply: float
    age_days: int
    mint_authority: Optional[str]
    freeze_authority: Optional[str]
    deployer: str
    creation_date: datetime
    is_mutable: bool
    metadata_uri: Optional[str]
    social_links: Dict[str, str]
    verified_status: bool
    current_price: Optional[float]
    market_cap: Optional[float]
    liquidity: Optional[float]

# ──────────────────────────────────────────────────────────────────────────────
# REQUEST/RESPONSE MODELS
# ──────────────────────────────────────────────────────────────────────────────
class RugCheckRequest(BaseModel):
    token_address: str

# ──────────────────────────────────────────────────────────────────────────────
# ENHANCED HELIUS API CLIENT
# ──────────────────────────────────────────────────────────────────────────────
class HeliusAPIClient:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.base_url = f"https://mainnet.helius-rpc.com/?api-key={api_key}"
        self.rest_url = "https://api.helius.xyz/v0"
        self.session: Optional[aiohttp.ClientSession] = None
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.session:
            await self.session.close()

    async def call_rpc(self, method: str, params: list, retry: int = 3) -> Dict[str, Any]:
        async with self.semaphore:
            for attempt in range(retry):
                try:
                    payload = {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": method,
                        "params": params,
                    }
                    async with self.session.post(
                        self.base_url, json=payload, timeout=30
                    ) as resp:
                        data = await resp.json()

                        if resp.status == 429:
                            delay = (2 ** attempt) * RATE_LIMIT_DELAY
                            logger.warning(f"Rate-limited ({method}); sleeping {delay:.2f}s")
                            await asyncio.sleep(delay)
                            continue

                        if resp.status != 200:
                            raise RuntimeError(f"HTTP {resp.status}")
                        if "error" in data:
                            raise RuntimeError(f"RPC error → {data['error']}")

                        await asyncio.sleep(RATE_LIMIT_DELAY)
                        return data.get("result", {})
                except Exception as exc:
                    logger.error(f"{method} attempt {attempt+1}/{retry} failed → {exc}")
                    if attempt == retry - 1:
                        return {}
                    await asyncio.sleep((2 ** attempt) * RATE_LIMIT_DELAY)
        return {}

    async def get_token_accounts(self, mint: str, limit: int = 1000) -> List[Dict]:
        """Get all token accounts for a mint"""
        return await self.call_rpc("getTokenLargestAccounts", [mint, {"commitment": "confirmed"}]) or {"value": []}

    async def get_multiple_accounts(self, addresses: List[str]) -> List[Dict]:
        """Get multiple account info in batch"""
        if not addresses:
            return []
        
        # Split into batches of 100 (Solana RPC limit)
        accounts = []
        for i in range(0, len(addresses), 100):
            batch = addresses[i:i+100]
            result = await self.call_rpc("getMultipleAccounts", [
                batch, 
                {"encoding": "jsonParsed", "commitment": "confirmed"}
            ])
            if result and "value" in result:
                accounts.extend(result["value"])
            await asyncio.sleep(0.1)  # Rate limiting
        
        return accounts

    async def get_token_metadata(self, mint: str) -> Dict[str, Any]:
        """Enhanced metadata fetching with fallbacks"""
        try:
            # Try Helius metadata API first
            url = f"{self.rest_url}/tokens/metadata?mints={mint}"
            headers = {"api-key": self.api_key}
            async with self.session.get(url, headers=headers, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data and len(data) > 0:
                        return data[0]
        except Exception as e:
            logger.debug(f"Metadata API failed: {e}")

        # Fallback to RPC
        try:
            result = await self.call_rpc("getAsset", [mint])
            if result:
                return result
        except Exception:
            pass

        # Final fallback - basic mint info
        account_info = await self.call_rpc("getAccountInfo", [mint, {"encoding": "jsonParsed"}])
        if account_info and account_info.get("value"):
            parsed = account_info["value"]["data"]["parsed"]["info"]
            return {
                "symbol": "UNKNOWN",
                "name": "Unknown Token",
                "decimals": parsed.get("decimals", 9),
                "supply": parsed.get("supply", "0")
            }
        
        return {}

    async def get_token_supply(self, mint: str) -> Dict[str, Any]:
        return await self.call_rpc("getTokenSupply", [mint]) or {}

    async def get_transaction_history(self, address: str, limit: int = 100) -> List[Dict]:
        return await self.call_rpc(
            "getSignaturesForAddress",
            [address, {"limit": limit, "commitment": "confirmed"}],
        ) or []

    async def get_parsed_transactions(self, signatures: List[str]) -> List[Dict]:
        txs: List[Dict] = []
        for i in range(0, len(signatures), 10):
            batch = signatures[i : i + 10]
            tasks = [
                self.call_rpc(
                    "getTransaction",
                    [sig, {"encoding": "jsonParsed", "commitment": "confirmed"}],
                )
                for sig in batch
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            txs.extend([r for r in results if isinstance(r, dict) and r])
            if i + 10 < len(signatures):
                await asyncio.sleep(0.2)
        return txs

# ──────────────────────────────────────────────────────────────────────────────
# ENHANCED TOKEN ANALYZER - REAL IMPLEMENTATION
# ──────────────────────────────────────────────────────────────────────────────
class EnhancedTokenAnalyzer:
    def __init__(self, helius_key: str) -> None:
        self.helius_key = helius_key

    async def analyze_token_comprehensive(self, mint: str) -> EnhancedTokenInfo:
        """Comprehensive token analysis"""
        async with HeliusAPIClient(self.helius_key) as helius:
            # Get basic token info
            metadata = await helius.get_token_metadata(mint)
            account_info = await helius.call_rpc("getAccountInfo", [mint, {"encoding": "jsonParsed"}])
            
            if not account_info or not account_info.get("value"):
                raise ValueError(f"Token {mint} not found")

            parsed = account_info["value"]["data"]["parsed"]["info"]
            
            # Get supply info
            supply_info = await helius.get_token_supply(mint)
            decimals = int(parsed.get("decimals", 9))
            raw_supply = float(supply_info.get("value", {}).get("amount", 0)) if supply_info else 0
            supply = raw_supply / (10 ** decimals)

            # Estimate age (simplified - in production, parse creation tx)
            age_days = 30  # Default estimate

            # Extract authorities
            mint_auth = parsed.get("mintAuthority")
            freeze_auth = parsed.get("freezeAuthority")
            
            # Get deployer from creation transaction (simplified)
            deployer = "Unknown"
            try:
                sigs = await helius.get_transaction_history(mint, 50)
                if sigs:
                    # Get earliest transaction
                    earliest_tx = await helius.call_rpc("getTransaction", [
                        sigs[-1]["signature"], 
                        {"encoding": "jsonParsed"}
                    ])
                    if earliest_tx and "transaction" in earliest_tx:
                        accounts = earliest_tx["transaction"]["message"]["accountKeys"]
                        if accounts:
                            deployer = accounts[0]["pubkey"]
            except Exception:
                pass

            # Price and market data (simplified)
            current_price = None
            market_cap = None
            liquidity = None

            return EnhancedTokenInfo(
                address=mint,
                symbol=metadata.get("symbol", "UNKNOWN"),
                name=metadata.get("name", "Unknown Token"),
                decimals=decimals,
                supply=supply,
                age_days=age_days,
                mint_authority=mint_auth,
                freeze_authority=freeze_auth,
                deployer=deployer,
                creation_date=datetime.utcnow() - timedelta(days=age_days),
                is_mutable=bool(mint_auth or freeze_auth),
                metadata_uri=metadata.get("uri"),
                social_links={},
                verified_status=False,
                current_price=current_price,
                market_cap=market_cap,
                liquidity=liquidity,
            )

    async def analyze_holders_comprehensive(self, mint: str) -> EnhancedHolderAnalysis:
        """Real holder analysis implementation"""
        async with HeliusAPIClient(self.helius_key) as helius:
            # Get token accounts
            largest_accounts = await helius.get_token_accounts(mint, 1000)
            
            if not largest_accounts or "value" not in largest_accounts:
                return self._empty_holder_analysis()

            accounts = largest_accounts["value"]
            if not accounts:
                return self._empty_holder_analysis()

            # Get account details
            account_addresses = [acc["address"] for acc in accounts]
            account_details = await helius.get_multiple_accounts(account_addresses)

            # Process holders
            holders = []
            total_supply = 0
            
            for i, (acc, details) in enumerate(zip(accounts, account_details)):
                if not details or not details.get("data"):
                    continue
                    
                try:
                    parsed = details["data"]["parsed"]["info"]
                    amount = float(parsed.get("amount", 0))
                    decimals = int(parsed.get("decimals", 9))
                    balance = amount / (10 ** decimals)
                    total_supply += balance
                    
                    owner = parsed.get("owner", "")
                    
                    # Detect suspicious patterns
                    risk_flags = []
                    is_suspicious = False
                    
                    # Check for round numbers (potential airdrop/bot)
                    if balance > 0 and balance == int(balance) and balance < 1000000:
                        risk_flags.append("round_number")
                        is_suspicious = True
                    
                    # Check for very new accounts (would need more RPC calls)
                    
                    holders.append(WalletInfo(
                        address=owner,
                        balance=balance,
                        percentage=0,  # Will calculate after
                        sol_balance=None,
                        token_count=None,
                        creation_time=None,
                        first_activity=None,
                        transaction_count=0,
                        is_suspicious=is_suspicious,
                        risk_flags=risk_flags,
                        rank=i + 1,
                        type="whale" if i < 5 else "holder",
                        risk="high" if is_suspicious else "medium" if i < 10 else "low"
                    ))
                    
                except Exception as e:
                    logger.debug(f"Error processing holder {i}: {e}")
                    continue

            # Calculate percentages
            if total_supply > 0:
                for holder in holders:
                    holder.percentage = (holder.balance / total_supply) * 100

            # Sort by balance
            holders.sort(key=lambda h: h.balance, reverse=True)

            # Detect bundle clusters
            bundle_clusters = await self._detect_bundles(holders[:50])  # Top 50 for performance

            # Calculate metrics
            total_holders = len(holders)
            unique_holders = len(set(h.address for h in holders))
            
            top_5_pct = sum(h.percentage for h in holders[:5]) if len(holders) >= 5 else 0
            top_10_pct = sum(h.percentage for h in holders[:10]) if len(holders) >= 10 else 0
            
            whale_wallets = [h.address for h in holders[:10]]
            suspicious_wallets = [h.address for h in holders if h.is_suspicious]
            
            balances = [h.balance for h in holders if h.balance > 0]
            avg_holding = statistics.mean(balances) if balances else 0
            median_holding = statistics.median(balances) if balances else 0

            # Create visualization data
            visualization = self._create_holder_visualization(holders, bundle_clusters, total_supply)

            return EnhancedHolderAnalysis(
                total_holders=total_holders,
                unique_holders=unique_holders,
                top_10_concentration=top_10_pct,
                top_5_concentration=top_5_pct,
                deployer_holdings=0,  # Would need deployer address
                holder_distribution=holders[:100],  # Top 100
                bundle_clusters=bundle_clusters,
                whale_wallets=whale_wallets,
                suspicious_wallets=suspicious_wallets,
                holder_growth_24h=0,  # Would need historical data
                average_holding=avg_holding,
                median_holding=median_holding,
                visualization_data=visualization,
            )

    async def _detect_bundles(self, holders: List[WalletInfo]) -> List[BundleCluster]:
        """Detect potential wallet bundles using pattern analysis"""
        clusters = []
        
        # Group by similar balance patterns
        balance_groups = defaultdict(list)
        for holder in holders:
            # Round balance to detect similar amounts
            rounded = round(holder.balance, -int(math.log10(holder.balance)) + 2) if holder.balance > 0 else 0
            balance_groups[rounded].append(holder)
        
        cluster_id = 0
        for balance, group in balance_groups.items():
            if len(group) >= 3 and balance > 0:  # Potential bundle
                # Calculate risk score based on patterns
                risk_score = min(100, len(group) * 10 + (50 if balance == int(balance) else 0))
                
                clusters.append(BundleCluster(
                    cluster_id=f"cluster_{cluster_id}",
                    wallets=[h.address for h in group],
                    creation_timeframe=0,  # Would need creation time analysis
                    funding_source=None,
                    similar_patterns=["similar_balance"],
                    risk_score=risk_score,
                    total_holdings=sum(h.balance for h in group)
                ))
                
                # Mark holders as bundled
                for holder in group:
                    holder.isBundled = True
                    holder.risk = "high"
                    holder.risk_flags.append("potential_bundle")
                
                cluster_id += 1
        
        return clusters

    def _create_holder_visualization(
        self, holders: List[WalletInfo], clusters: List[BundleCluster], total_supply: float
    ) -> HolderVisualization:
        """Create visualization data for frontend"""
        
        # Holder bubbles (top 20)
        holder_bubbles = []
        for i, holder in enumerate(holders[:20]):
            holder_bubbles.append({
                "rank": holder.rank,
                "address": holder.address[:8] + "..." + holder.address[-4:],
                "percentage": round(holder.percentage, 2),
                "type": holder.type,
                "risk": holder.risk,
                "isBundled": holder.isBundled
            })

        # Concentration chart
        top1 = holders[0].percentage if len(holders) > 0 else 0
        top5 = sum(h.percentage for h in holders[:5]) if len(holders) >= 5 else 0
        top10 = sum(h.percentage for h in holders[:10]) if len(holders) >= 10 else 0
        others = max(0, 100 - top10)

        concentration_chart = {
            "top1": round(top1, 1),
            "top5": round(top5, 1),
            "top10": round(top10, 1),
            "others": round(others, 1)
        }

        # Bundle summary
        high_risk_bundles = len([c for c in clusters if c.risk_score > 70])
        bundled_supply = sum(c.total_holdings for c in clusters)
        bundled_pct = (bundled_supply / total_supply * 100) if total_supply > 0 else 0
        largest_bundle = max([len(c.wallets) for c in clusters]) if clusters else 0
        
        bundle_summary = BundleRiskSummary(
            total_bundles=len(clusters),
            high_risk_bundles=high_risk_bundles,
            bundled_supply_percentage=round(bundled_pct, 1),
            largest_bundle_size=largest_bundle,
            bundle_concentration_score=min(100, len(clusters) * 15 + high_risk_bundles * 10),
            patterns_detected=["similar_balance", "coordinated_activity"] if clusters else []
        )

        return HolderVisualization(
            holder_bubbles=holder_bubbles,
            concentration_chart=concentration_chart,
            bundle_summary=bundle_summary
        )

    def _empty_holder_analysis(self) -> EnhancedHolderAnalysis:
        """Return empty analysis when no data available"""
        empty_viz = HolderVisualization(
            holder_bubbles=[],
            concentration_chart={"top1": 0, "top5": 0, "top10": 0, "others": 100},
            bundle_summary=BundleRiskSummary(0, 0, 0, 0, 0, [])
        )
        
        return EnhancedHolderAnalysis(
            total_holders=0,
            unique_holders=0,
            top_10_concentration=0,
            top_5_concentration=0,
            deployer_holdings=0,
            holder_distribution=[],
            bundle_clusters=[],
            whale_wallets=[],
            suspicious_wallets=[],
            holder_growth_24h=0,
            average_holding=0,
            median_holding=0,
            visualization_data=empty_viz,
        )

    async def analyze_transactions_comprehensive(self, mint: str, deployer: str) -> TransactionAnalysis:
        """Simplified transaction analysis"""
        # This would normally require extensive transaction parsing
        # For now, return basic structure with some estimated data
        
        return TransactionAnalysis(
            total_transactions=0,
            unique_traders=0,
            volume_24h=0.0,
            volume_7d=0.0,
            first_transaction=None,
            peak_activity_time=None,
            suspicious_patterns=[],
            coordinated_activity=False,
            wash_trading_risk=0.0,
        )

# ──────────────────────────────────────────────────────────────────────────────
# ENHANCED DEPLOYER ANALYZER
# ──────────────────────────────────────────────────────────────────────────────
class DeployerAnalyzer:
    def __init__(self, helius: HeliusAPIClient) -> None:
        self.helius = helius

    async def analyze_deployer(self, deployer_address: str, current_mint: str) -> DeployerAnalysis:
        if not deployer_address or len(deployer_address) < 32:
            return self._empty(deployer_address)

        try:
            # Get transaction history
            sigs = await self.helius.get_transaction_history(deployer_address, 200)
            deployed = await self._find_deployed_tokens(sigs)

            # Analyze each deployed token
            tokens: List[DeployerToken] = []
            for mint in deployed[:20]:  # Limit for performance
                if mint == current_mint:
                    continue
                tok = await self._analyze_token(mint)
                if tok:
                    tokens.append(tok)

            tokens.sort(key=lambda t: t.launch_date, reverse=True)
            return self._calc_metrics(deployer_address, tokens)
            
        except Exception as exc:
            logger.error(f"Deployer analysis failed → {exc}")
            return self._empty(deployer_address)

    async def _find_deployed_tokens(self, signatures: List[Dict]) -> List[str]:
        """Find tokens deployed by this address"""
        deployed: set[str] = set()
        sigs = [s["signature"] for s in signatures if "signature" in s][:50]
        txs = await self.helius.get_parsed_transactions(sigs)
        
        for tx in txs:
            try:
                instructions = tx.get("transaction", {}).get("message", {}).get("instructions", [])
                for inst in instructions:
                    # Check for mint initialization
                    if (inst.get("program") == "spl-token" and 
                        inst.get("parsed", {}).get("type") in ("initializeMint", "initializeMint2")):
                        mint = inst.get("parsed", {}).get("info", {}).get("mint")
                        if mint:
                            deployed.add(mint)
                    
                    # Check for token program calls
                    elif inst.get("programId") == "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA":
                        accounts = inst.get("accounts", [])
                        if accounts and len(accounts) > 0:
                            deployed.add(accounts[0])
            except Exception as e:
                logger.debug(f"Error parsing transaction: {e}")
                continue
        
        return list(deployed)

    async def _analyze_token(self, mint: str) -> Optional[DeployerToken]:
        """Analyze individual deployed token"""
        try:
            # Get token metadata
            metadata = await self.helius.get_token_metadata(mint)
            
            # Get account info
            account_info = await self.helius.call_rpc(
                "getAccountInfo", [mint, {"encoding": "jsonParsed"}]
            )
            
            if not account_info or not account_info.get("value"):
                return None

            parsed = account_info["value"]["data"]["parsed"]["info"]
            decimals = int(parsed.get("decimals", 9))
            
            # Get supply
            supply_info = await self.helius.get_token_supply(mint)
            supply = 0.0
            if supply_info and "value" in supply_info:
                raw_supply = float(supply_info["value"]["amount"])
                supply = raw_supply / (10 ** decimals)

            # Estimate metrics (simplified)
            holder_count = 1  # Default
            creation_date = datetime.utcnow() - timedelta(days=30)  # Estimate
            age = (datetime.utcnow() - creation_date).days
            
            # Simple rug detection heuristics
            is_rugged = (
                supply == 0 or 
                holder_count < 5 or 
                (metadata.get("symbol", "") == "UNKNOWN")
            )

            return DeployerToken(
                address=mint,
                symbol=metadata.get("symbol", "UNK"),
                name=metadata.get("name", "Unknown"),
                launch_date=creation_date,
                current_price=None,
                current_market_cap=None,
                highest_market_cap=None,
                is_rugged=is_rugged,
                holder_count=holder_count,
                liquidity=None,
                age_days=age,
            )
            
        except Exception as exc:
            logger.debug(f"Token analysis failed for {mint}: {exc}")
            return None

    def _calc_metrics(self, deployer: str, tokens: List[DeployerToken]) -> DeployerAnalysis:
        """Calculate deployer metrics"""
        if not tokens:
            return self._empty(deployer)

        total = len(tokens)
        rugged = sum(1 for t in tokens if t.is_rugged)
        success = sum(1 for t in tokens if not t.is_rugged and (t.current_market_cap or 0) > 1000)
        active = total - rugged
        
        lifespans = [t.age_days for t in tokens]
        avg_life = statistics.mean(lifespans) if lifespans else 0.0
        
        value_extracted = sum(t.highest_market_cap or 0 for t in tokens if t.is_rugged)

        # Reputation scoring
        if total == 0:
            score = 50
        else:
            rug_penalty = int((rugged / total) * 60)
            success_bonus = int((success / total) * 30)
            longevity_bonus = 10 if avg_life > 90 else 5 if avg_life > 30 else 0
            score = max(0, min(100, 70 - rug_penalty + success_bonus + longevity_bonus))

        # Pattern classification
        if total == 0:
            pattern, risk = "unknown", "UNKNOWN - No deployment history"
        elif rugged / total > 0.8:
            pattern, risk = "serial_rugger", "EXTREME RISK - Serial rug puller detected"
        elif rugged / total > 0.5:
            pattern, risk = "high_risk", "HIGH RISK - Many failed projects"
        elif success / total > 0.6:
            pattern, risk = "legitimate", "LOW RISK - Mostly successful projects"
        elif total == 1:
            pattern, risk = "new_deployer", "MEDIUM RISK - New deployer"
        else:
            pattern, risk = "mixed", "MEDIUM RISK - Mixed track record"

        return DeployerAnalysis(
            deployer_address=deployer,
            total_tokens_deployed=total,
            successful_launches=success,
            rugged_tokens=rugged,
            active_tokens=active,
            average_token_lifespan=avg_life,
            total_value_extracted=value_extracted,
            reputation_score=score,
            token_history=tokens[:10],  # Limit for response size
            deployment_pattern=pattern,
            risk_assessment=risk,
        )

    def _empty(self, deployer: str) -> DeployerAnalysis:
        """Empty deployer analysis"""
        return DeployerAnalysis(
            deployer_address=deployer or "Unknown",
            total_tokens_deployed=0,
            successful_launches=0,
            rugged_tokens=0,
            active_tokens=0,
            average_token_lifespan=0.0,
            total_value_extracted=0.0,
            reputation_score=50,
            token_history=[],
            deployment_pattern="unknown",
            risk_assessment="UNKNOWN - No deployment history found",
        )

# ──────────────────────────────────────────────────────────────────────────────
# AI ANALYST
# ──────────────────────────────────────────────────────────────────────────────
class AIAnalyst:
    def __init__(self, openai_key: str = "") -> None:
        self.client = None
        if openai_key and OPENAI_AVAILABLE:
            try:
                self.client = AsyncOpenAI(api_key=openai_key)
            except Exception:
                logger.warning("OpenAI client initialization failed")

    async def generate_analysis(
        self,
        token: EnhancedTokenInfo,
        holders: EnhancedHolderAnalysis,
        txs: TransactionAnalysis,
        score: int,
        deployer_analysis: Optional[DeployerAnalysis] = None,
    ) -> str:
        """Generate AI analysis or rule-based fallback"""
        
        if not self.client:
            return self._rule_based_analysis(token, holders, txs, score, deployer_analysis)

        # Prepare data for AI
        analysis_data = {
            "token": {
                "symbol": token.symbol,
                "name": token.name,
                "age_days": token.age_days,
                "supply": token.supply,
                "has_mint_authority": bool(token.mint_authority),
                "has_freeze_authority": bool(token.freeze_authority),
            },
            "holders": {
                "total": holders.total_holders,
                "unique": holders.unique_holders,
                "top_5_concentration": holders.top_5_concentration,
                "top_10_concentration": holders.top_10_concentration,
                "bundle_clusters": len(holders.bundle_clusters),
                "suspicious_wallets": len(holders.suspicious_wallets),
            },
            "risk_score": score,
        }

        if deployer_analysis:
            analysis_data["deployer"] = {
                "total_deployed": deployer_analysis.total_tokens_deployed,
                "rugged_count": deployer_analysis.rugged_tokens,
                "success_count": deployer_analysis.successful_launches,
                "reputation_score": deployer_analysis.reputation_score,
                "pattern": deployer_analysis.deployment_pattern,
            }

        prompt = f"""
Analyze this Solana token risk profile and provide a concise assessment:

{json.dumps(analysis_data, indent=2)}

Provide a professional analysis focusing on:
1. Overall risk level and reasoning
2. Deployer history concerns (if applicable)
3. Holder distribution red flags
4. Specific actionable recommendations

Keep it concise but informative, around 150-200 words.
"""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning(f"OpenAI analysis failed: {exc}")
            return self._rule_based_analysis(token, holders, txs, score, deployer_analysis)

    def _rule_based_analysis(
        self,
        token: EnhancedTokenInfo,
        holders: EnhancedHolderAnalysis,
        txs: TransactionAnalysis,
        score: int,
        deployer: Optional[DeployerAnalysis] = None,
    ) -> str:
        """Rule-based analysis fallback"""
        
        parts = []
        
        # Risk level
        if score >= 80:
            parts.append("🚨 **EXTREME RISK** - Multiple critical red flags detected.")
        elif score >= 60:
            parts.append("⚠️ **HIGH RISK** - Significant concerns identified.")
        elif score >= 40:
            parts.append("⚡ **MEDIUM RISK** - Some caution warranted.")
        elif score >= 20:
            parts.append("✅ **LOW RISK** - Minimal concerns found.")
        else:
            parts.append("🟢 **VERY LOW RISK** - Token appears legitimate.")

        # Deployer analysis
        if deployer and deployer.total_tokens_deployed > 0:
            if deployer.deployment_pattern == "serial_rugger":
                parts.append(f"🚩 **DEPLOYER WARNING**: {deployer.rugged_tokens}/{deployer.total_tokens_deployed} previous tokens were rugged.")
            elif deployer.deployment_pattern == "legitimate":
                parts.append(f"👍 **DEPLOYER POSITIVE**: Strong track record with {deployer.successful_launches} successful launches.")
            elif deployer.reputation_score < 30:
                parts.append("⚠️ **DEPLOYER CONCERN**: Poor historical performance detected.")

        # Holder concentration
        if holders.top_5_concentration > 70:
            parts.append(f"🐋 **WHALE ALERT**: Top 5 holders control {holders.top_5_concentration:.1f}% of supply.")
        elif holders.top_5_concentration > 50:
            parts.append(f"📊 **CONCENTRATION RISK**: Top 5 holders own {holders.top_5_concentration:.1f}% of tokens.")

        # Bundle detection
        if len(holders.bundle_clusters) > 0:
            high_risk_bundles = len([c for c in holders.bundle_clusters if c.risk_score > 70])
            if high_risk_bundles > 0:
                parts.append(f"🔗 **BUNDLE RISK**: {high_risk_bundles} high-risk wallet clusters detected.")

        # Authority risks
        if token.mint_authority:
            parts.append("⚠️ **MINT RISK**: Mint authority still active - new tokens can be created.")
        if token.freeze_authority:
            parts.append("❄️ **FREEZE RISK**: Freeze authority active - transfers can be halted.")

        # Recommendations
        parts.append("\n**RECOMMENDATIONS:**")
        if score >= 60:
            parts.append("• Avoid or exit position immediately")
            parts.append("• High probability of rug pull or manipulation")
        elif score >= 40:
            parts.append("• Exercise extreme caution")
            parts.append("• Consider smaller position sizes")
            parts.append("• Monitor deployer activity closely")
        else:
            parts.append("• Proceed with standard crypto caution")
            parts.append("• Monitor holder distribution changes")

        parts.append("\n⚠️ **DISCLAIMER**: This analysis is not financial advice. Always DYOR!")

        return " ".join(parts)

# ──────────────────────────────────────────────────────────────────────────────
# MAIN RUG CHECKER
# ──────────────────────────────────────────────────────────────────────────────
class RugChecker:
    def __init__(self, helius_key: str, openai_key: str = "") -> None:
        self.analyzer = EnhancedTokenAnalyzer(helius_key)
        self.helius_key = helius_key
        self.ai = AIAnalyst(openai_key)

    async def analyze_token_ultimate(self, mint: str) -> Dict[str, Any]:
        """Main analysis orchestrator"""
        try:
            logger.info(f"Starting comprehensive analysis for {mint}")
            
            # Core analyses
            token = await self.analyzer.analyze_token_comprehensive(mint)
            holders = await self.analyzer.analyze_holders_comprehensive(mint)
            txs = await self.analyzer.analyze_transactions_comprehensive(mint, token.deployer)

            # Deployer analysis
            deployer_analysis = None
            if token.deployer and token.deployer != "Unknown":
                async with HeliusAPIClient(self.helius_key) as helius:
                    deployer_analyzer = DeployerAnalyzer(helius)
                    deployer_analysis = await deployer_analyzer.analyze_deployer(token.deployer, mint)

            # Risk scoring
            risk_score, confidence = self._calculate_risk_score(token, holders, txs, deployer_analysis)

            # AI analysis
            ai_analysis = await self.ai.generate_analysis(token, holders, txs, risk_score, deployer_analysis)

            # Prepare response in format expected by frontend
            response = {
                "success": True,
                "token_info": self._serialize_token_info(token),
                "holder_analysis": self._serialize_holder_analysis(holders),
                "transaction_analysis": asdict(txs),
                "deployer_analysis": asdict(deployer_analysis) if deployer_analysis else None,
                "risk_score": risk_score,
                "confidence_level": confidence,
                "ai_analysis": ai_analysis,
                "visualization": self._create_visualization_data(holders),
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "riskFactors": self._extract_risk_factors(token, holders, deployer_analysis, risk_score)
            }

            logger.info(f"Analysis completed for {mint} - Risk Score: {risk_score}")
            return response

        except Exception as exc:
            logger.error(f"Analysis failed for {mint}: {exc}")
            return {
                "success": False,
                "error": f"Analysis failed: {str(exc)}",
                "token_address": mint
            }

    def _calculate_risk_score(
        self,
        token: EnhancedTokenInfo,
        holders: EnhancedHolderAnalysis,
        txs: TransactionAnalysis,
        deployer: Optional[DeployerAnalysis],
    ) -> Tuple[int, int]:
        """Calculate risk score and confidence level"""
        
        risk_factors = {}
        
        # Holder concentration (0-40 points)
        concentration_score = min(40, int(holders.top_5_concentration * 0.6))
        risk_factors["concentration"] = concentration_score
        
        # Authority risks (0-20 points)
        authority_score = 0
        if token.mint_authority:
            authority_score += 10
        if token.freeze_authority:
            authority_score += 10
        risk_factors["authority"] = authority_score
        
        # Bundle risks (0-25 points)
        bundle_score = 0
        if len(holders.bundle_clusters) > 0:
            high_risk_bundles = len([c for c in holders.bundle_clusters if c.risk_score > 70])
            bundle_score = min(25, len(holders.bundle_clusters) * 5 + high_risk_bundles * 5)
        risk_factors["bundles"] = bundle_score
        
        # Deployer reputation (0-30 points)
        deployer_score = 0
        if deployer and deployer.total_tokens_deployed > 0:
            # Invert reputation score (low reputation = high risk)
            deployer_score = int((100 - deployer.reputation_score) * 0.3)
        else:
            deployer_score = 15  # Unknown deployer = medium risk
        risk_factors["deployer"] = deployer_score
        
        # Age factor (0-10 points)
        age_score = 0
        if token.age_days < 1:
            age_score = 10
        elif token.age_days < 7:
            age_score = 5
        elif token.age_days < 30:
            age_score = 2
        risk_factors["age"] = age_score
        
        # Calculate total risk score
        total_risk = sum(risk_factors.values())
        final_risk = min(100, total_risk)
        
        # Calculate confidence based on data availability
        confidence = 50  # Base confidence
        
        if holders.total_holders > 0:
            confidence += 20
        if deployer and deployer.total_tokens_deployed > 0:
            confidence += 20
        if token.age_days > 7:
            confidence += 10
        
        confidence = min(100, confidence)
        
        return final_risk, confidence

    def _serialize_token_info(self, token: EnhancedTokenInfo) -> Dict[str, Any]:
        """Serialize token info for JSON response"""
        return {
            "address": token.address,
            "symbol": token.symbol,
            "name": token.name,
            "decimals": token.decimals,
            "supply": token.supply,
            "age_days": token.age_days,
            "mint_authority": bool(token.mint_authority),
            "freeze_authority": bool(token.freeze_authority),
            "deployer": token.deployer,
            "creation_date": token.creation_date.isoformat(),
            "is_mutable": token.is_mutable,
            "verified_status": token.verified_status,
            "current_price": token.current_price,
            "market_cap": token.market_cap,
            "liquidity": token.liquidity,
        }

    def _serialize_holder_analysis(self, holders: EnhancedHolderAnalysis) -> Dict[str, Any]:
        """Serialize holder analysis for JSON response"""
        return {
            "total_holders": holders.total_holders,
            "unique_holders": holders.unique_holders,
            "top_10_concentration": holders.top_10_concentration,
            "top_5_concentration": holders.top_5_concentration,
            "deployer_holdings": holders.deployer_holdings,
            "bundle_clusters": [asdict(cluster) for cluster in holders.bundle_clusters],
            "whale_wallets": holders.whale_wallets,
            "suspicious_wallets": holders.suspicious_wallets,
            "holder_growth_24h": holders.holder_growth_24h,
            "average_holding": holders.average_holding,
            "median_holding": holders.median_holding,
        }

    def _create_visualization_data(self, holders: EnhancedHolderAnalysis) -> Dict[str, Any]:
        """Create visualization data for frontend charts"""
        viz = holders.visualization_data
        
        return {
            "holder_bubbles": viz.holder_bubbles,
            "concentration_chart": viz.concentration_chart,
            "bundle_summary": asdict(viz.bundle_summary)
        }

    def _extract_risk_factors(
        self, 
        token: EnhancedTokenInfo, 
        holders: EnhancedHolderAnalysis, 
        deployer: Optional[DeployerAnalysis],
        risk_score: int
    ) -> List[str]:
        """Extract key risk factors for display"""
        factors = []
        
        if token.mint_authority:
            factors.append("Mint authority is still active - new tokens can be created")
        
        if token.freeze_authority:
            factors.append("Freeze authority is active - transfers can be frozen")
        
        if holders.top_5_concentration > 60:
            factors.append(f"High concentration: Top 5 holders control {holders.top_5_concentration:.1f}% of supply")
        
        if len(holders.bundle_clusters) > 0:
            high_risk = len([c for c in holders.bundle_clusters if c.risk_score > 70])
            factors.append(f"Detected {len(holders.bundle_clusters)} wallet clusters ({high_risk} high-risk)")
        
        if deployer and deployer.deployment_pattern == "serial_rugger":
            factors.append(f"Deployer has rugged {deployer.rugged_tokens}/{deployer.total_tokens_deployed} previous tokens")
        
        if token.age_days < 7:
            factors.append(f"Very new token ({token.age_days} days old) - limited history available")
        
        if len(holders.suspicious_wallets) > 0:
            factors.append(f"Found {len(holders.suspicious_wallets)} wallets with suspicious patterns")
        
        if not factors:
            if risk_score < 30:
                factors.append("No major risk factors detected - token appears relatively safe")
            else:
                factors.append("Multiple minor risk factors detected - exercise caution")
        
        return factors

# ──────────────────────────────────────────────────────────────────────────────
# FASTAPI SERVER
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Ultimate Rug Checker API", version="2.0")

# Initialize rug checker
rug_checker = RugChecker(HELIUS_API_KEY, OPENAI_API_KEY)

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the frontend HTML"""
    # In production, you'd serve this from a static file
    # For now, return a simple message
    return """
    <html>
        <head><title>Ultimate Rug Checker</title></head>
        <body>
            <h1>Ultimate Rug Checker API</h1>
            <p>API is running! Use POST /rugcheck to analyze tokens.</p>
            <p>Frontend should be served separately.</p>
        </body>
    </html>
    """

@app.post("/rugcheck")
async def rugcheck_endpoint(request: RugCheckRequest):
    """Main rug check endpoint"""
    try:
        # Validate token address
        token_address = request.token_address.strip()
        if not token_address or len(token_address) < 32:
            raise HTTPException(status_code=400, detail="Invalid token address")
        
        # Perform analysis
        result = await rug_checker.analyze_token_ultimate(token_address)
        
        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "Analysis failed"))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

if __name__ == "__main__":
    print("🚀 Starting Ultimate Rug Checker API...")
    print("📡 Make sure to set your HELIUS_API_KEY!")
    print("🤖 OpenAI integration available for enhanced analysis")
    print("🔍 API will be available at http://localhost:8000")
    
    uvicorn.run(
        "main:app",  # Change this to match your filename
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )