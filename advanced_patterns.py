"""
Advanced pattern detection for sophisticated scam identification
Goes beyond basic metrics to identify complex fraud patterns
"""

import re
import statistics
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class BundleTransaction:
    """Transaction bundle detected on-chain"""
    bundle_id: str
    transaction_count: int
    total_value: float
    timestamp: datetime
    accounts_involved: List[str]
    suspicious_score: float

@dataclass
class WalletCluster:
    """Cluster of related wallets"""
    cluster_id: str
    wallet_addresses: List[str]
    connection_strength: float
    transaction_overlap: int
    creation_timeframe: int  # hours
    purpose: str  # 'pump', 'distribution', 'wash_trading'

@dataclass
class MemeCoinProfile:
    """Profile analysis for meme coin characteristics"""
    has_meme_name: bool
    uses_trending_terms: bool
    copycat_score: float  # similarity to successful memes
    narrative_strength: float
    viral_potential: float
    authenticity_score: float

@dataclass
class LiquidityPattern:
    """Advanced liquidity analysis"""
    initial_liquidity: float
    liquidity_changes: List[Tuple[datetime, float]]
    largest_removal: float
    removal_frequency: int
    sandwich_attacks: int
    mev_activity: float

@dataclass
class VestingAnalysis:
    """Token vesting and unlock analysis"""
    has_vesting: bool
    total_vested_tokens: float
    unlock_schedule: List[Tuple[datetime, float]]
    cliff_period: Optional[int]
    linear_vesting: bool
    team_allocation: float

class AdvancedPatternDetector:
    """Detect sophisticated scam patterns and advanced threats"""
    
    def __init__(self):
        self.meme_keywords = [
            'moon', 'rocket', 'doge', 'shib', 'pepe', 'wojak', 'chad', 'diamond', 'hands',
            'ape', 'banana', 'lambo', 'tendies', 'hodl', 'wagmi', 'gm', 'fren', 'based',
            'pump', 'dump', 'rekt', 'fud', 'fomo', 'cope', 'seethe', 'ngmi', 'bullish'
        ]
        
        self.scam_indicators = [
            'safe', 'baby', 'mini', 'elon', 'biden', 'trump', 'inu', 'coin', 'token',
            'x100', 'x1000', 'moon', 'mars', 'rocket', 'gem', 'stealth', 'fair', 'launch'
        ]
        
        self.trending_patterns = [
            r'\b\w+inu\b',  # *inu coins
            r'\b\w+doge\b', # *doge coins  
            r'\b\w+moon\b', # *moon coins
            r'\bsafe\w+\b', # safe* coins
            r'\bbaby\w+\b', # baby* coins
        ]

    def analyze_meme_coin_profile(self, token_name: str, token_symbol: str) -> MemeCoinProfile:
        """Analyze if token fits meme coin patterns and authenticity"""
        name_lower = token_name.lower()
        symbol_lower = token_symbol.lower()
        
        # Check for meme characteristics
        has_meme_name = any(keyword in name_lower for keyword in self.meme_keywords)
        
        # Check for trending terms
        uses_trending_terms = any(re.search(pattern, name_lower) for pattern in self.trending_patterns)
        
        # Calculate copycat score
        copycat_score = self._calculate_copycat_score(name_lower, symbol_lower)
        
        # Narrative strength (how well it fits meme narratives)
        narrative_strength = self._calculate_narrative_strength(name_lower)
        
        # Viral potential based on name/symbol
        viral_potential = self._calculate_viral_potential(name_lower, symbol_lower)
        
        # Authenticity (original vs derivative)
        authenticity_score = self._calculate_authenticity_score(name_lower, symbol_lower)
        
        return MemeCoinProfile(
            has_meme_name=has_meme_name,
            uses_trending_terms=uses_trending_terms,
            copycat_score=copycat_score,
            narrative_strength=narrative_strength,
            viral_potential=viral_potential,
            authenticity_score=authenticity_score
        )
    
    def _calculate_copycat_score(self, name: str, symbol: str) -> float:
        """Calculate how much token copies existing successful projects"""
        known_successful = [
            'dogecoin', 'shiba', 'pepe', 'floki', 'bonk', 'wojak', 'chad',
            'bitcoin', 'ethereum', 'solana', 'avalanche', 'polygon'
        ]
        
        score = 0.0
        
        # Direct name similarity
        for known in known_successful:
            if known in name or known in symbol:
                score += 0.3
                
            # Levenshtein-like similarity for close matches
            if self._similar_strings(known, name) or self._similar_strings(known, symbol):
                score += 0.2
        
        # Check for common scam patterns
        scam_patterns = ['safe', 'baby', 'mini', 'x100', 'moon', 'mars']
        for pattern in scam_patterns:
            if pattern in name:
                score += 0.15
        
        return min(score, 1.0)
    
    def _similar_strings(self, s1: str, s2: str, threshold: float = 0.8) -> bool:
        """Check if strings are similar using simple ratio"""
        if not s1 or not s2:
            return False
        
        # Simple similarity check
        longer = s1 if len(s1) > len(s2) else s2
        shorter = s2 if len(s1) > len(s2) else s1
        
        if len(longer) == 0:
            return True
        
        # Count matching characters
        matches = 0
        for char in shorter:
            if char in longer:
                matches += 1
        
        similarity = matches / len(longer)
        return similarity > threshold
    
    def _calculate_narrative_strength(self, name: str) -> float:
        """Calculate how well token fits into current narratives"""
        narratives = {
            'ai': ['ai', 'artificial', 'intelligence', 'gpt', 'chat', 'bot'],
            'gaming': ['game', 'play', 'meta', 'verse', 'nft', 'pixel'],
            'defi': ['swap', 'yield', 'farm', 'stake', 'pool', 'vault'],
            'meme': ['doge', 'pepe', 'wojak', 'chad', 'frog', 'cat'],
            'animal': ['dog', 'cat', 'frog', 'ape', 'monkey', 'tiger']
        }
        
        strength = 0.0
        for narrative, keywords in narratives.items():
            if any(keyword in name for keyword in keywords):
                strength += 0.2
        
        return min(strength, 1.0)
    
    def _calculate_viral_potential(self, name: str, symbol: str) -> float:
        """Calculate potential for viral spread based on name characteristics"""
        viral_factors = {
            'short_symbol': 0.2 if len(symbol) <= 4 else 0.0,
            'pronounceable': 0.15 if self._is_pronounceable(symbol) else 0.0,
            'meme_keywords': 0.2 if any(kw in name for kw in self.meme_keywords[:10]) else 0.0,
            'animal_reference': 0.15 if any(animal in name for animal in ['dog', 'cat', 'frog', 'ape']) else 0.0,
            'cultural_reference': 0.1 if any(ref in name for ref in ['elon', 'trump', 'biden', 'wojak']) else 0.0,
            'action_word': 0.1 if any(action in name for action in ['moon', 'rocket', 'pump', 'launch']) else 0.0
        }
        
        return sum(viral_factors.values())
    
    def _is_pronounceable(self, text: str) -> bool:
        """Check if text is easily pronounceable"""
        vowels = 'aeiou'
        consonants = 'bcdfghjklmnpqrstvwxyz'
        
        vowel_count = sum(1 for char in text.lower() if char in vowels)
        consonant_count = sum(1 for char in text.lower() if char in consonants)
        
        # Good balance of vowels and consonants
        if len(text) == 0:
            return False
        
        vowel_ratio = vowel_count / len(text)
        return 0.2 <= vowel_ratio <= 0.6
    
    def _calculate_authenticity_score(self, name: str, symbol: str) -> float:
        """Calculate originality vs derivative score"""
        authenticity = 1.0
        
        # Penalty for obvious scam indicators
        for indicator in self.scam_indicators:
            if indicator in name:
                authenticity -= 0.1
        
        # Penalty for number inflation
        if any(mult in name for mult in ['x100', 'x1000', '100x', '1000x']):
            authenticity -= 0.3
        
        # Penalty for copycat patterns
        if any(re.search(pattern, name) for pattern in self.trending_patterns):
            authenticity -= 0.15
        
        # Bonus for original naming
        if len(name) > 10 and not any(common in name for common in ['coin', 'token', 'cash']):
            authenticity += 0.1
        
        return max(0.0, authenticity)

    def detect_wash_trading_pattern(self, transaction_history: List[Dict]) -> Tuple[bool, float]:
        """Detect wash trading patterns in transaction history"""
        if not transaction_history or len(transaction_history) < 10:
            return False, 0.0
        
        # Analyze transaction patterns
        wallets = set()
        repeated_interactions = defaultdict(int)
        circular_trades = 0
        
        for tx in transaction_history:
            # Extract wallet addresses (simplified)
            if 'accounts' in tx:
                tx_wallets = tx['accounts']
                wallets.update(tx_wallets)
                
                # Check for repeated interactions between same wallets
                for i, wallet1 in enumerate(tx_wallets):
                    for wallet2 in tx_wallets[i+1:]:
                        pair = tuple(sorted([wallet1, wallet2]))
                        repeated_interactions[pair] += 1
        
        # Calculate wash trading indicators
        total_interactions = len(repeated_interactions)
        if total_interactions == 0:
            return False, 0.0
        
        # High frequency of repeated interactions
        avg_interactions = sum(repeated_interactions.values()) / total_interactions
        max_interactions = max(repeated_interactions.values()) if repeated_interactions else 0
        
        # Suspicious if many repeated interactions
        wash_score = min(1.0, (avg_interactions - 2) / 10 + (max_interactions - 5) / 20)
        
        is_wash_trading = wash_score > 0.6
        return is_wash_trading, wash_score

    def detect_bundle_manipulation(self, recent_transactions: List[Dict]) -> List[BundleTransaction]:
        """Detect transaction bundling for price manipulation"""
        bundles = []
        
        # Group transactions by timestamp (within 5 seconds = potential bundle)
        time_groups = defaultdict(list)
        
        for tx in recent_transactions:
            if 'blockTime' in tx:
                timestamp = datetime.fromtimestamp(tx['blockTime'])
                # Round to 5-second intervals
                time_key = timestamp.replace(second=(timestamp.second // 5) * 5, microsecond=0)
                time_groups[time_key].append(tx)
        
        # Analyze groups for bundle characteristics
        for time_key, group_txs in time_groups.items():
            if len(group_txs) >= 3:  # 3+ transactions in 5 seconds
                accounts = set()
                total_value = 0
                
                for tx in group_txs:
                    if 'accounts' in tx:
                        accounts.update(tx['accounts'])
                    # Simplified value calculation
                    total_value += tx.get('value', 0)
                
                # Calculate suspicion score
                suspicious_score = min(1.0, len(group_txs) / 10 + len(accounts) / 20)
                
                if suspicious_score > 0.3:
                    bundle = BundleTransaction(
                        bundle_id=f"bundle_{time_key.isoformat()}",
                        transaction_count=len(group_txs),
                        total_value=total_value,
                        timestamp=time_key,
                        accounts_involved=list(accounts),
                        suspicious_score=suspicious_score
                    )
                    bundles.append(bundle)
        
        return bundles

    def analyze_deployer_wallet_cluster(self, deployer_address: str, 
                                      related_addresses: List[str]) -> WalletCluster:
        """Analyze if deployer is part of a coordinated wallet cluster"""
        
        # For now, create a basic cluster analysis
        # In production, this would analyze on-chain connections
        
        cluster_size = len(related_addresses) + 1  # +1 for deployer
        
        # Estimate connection strength (simplified)
        connection_strength = min(1.0, cluster_size / 20)
        
        # Estimate purpose based on patterns
        purpose = "unknown"
        if cluster_size > 10:
            purpose = "distribution"
        elif cluster_size > 5:
            purpose = "pump"
        else:
            purpose = "single_operator"
        
        return WalletCluster(
            cluster_id=f"cluster_{deployer_address[:8]}",
            wallet_addresses=[deployer_address] + related_addresses,
            connection_strength=connection_strength,
            transaction_overlap=cluster_size * 2,  # Estimated
            creation_timeframe=24,  # Assume 24 hours
            purpose=purpose
        )

    def detect_liquidity_manipulation(self, liquidity_events: List[Dict]) -> LiquidityPattern:
        """Detect suspicious liquidity manipulation patterns"""
        
        if not liquidity_events:
            return LiquidityPattern(
                initial_liquidity=0,
                liquidity_changes=[],
                largest_removal=0,
                removal_frequency=0,
                sandwich_attacks=0,
                mev_activity=0
            )
        
        # Analyze liquidity changes
        changes = []
        removals = []
        
        for event in liquidity_events:
            timestamp = datetime.fromtimestamp(event.get('timestamp', 0))
            amount = event.get('amount', 0)
            
            changes.append((timestamp, amount))
            
            if amount < 0:  # Removal
                removals.append(abs(amount))
        
        # Calculate metrics
        initial_liquidity = liquidity_events[0].get('amount', 0) if liquidity_events else 0
        largest_removal = max(removals) if removals else 0
        removal_frequency = len(removals)
        
        # Estimate sandwich attacks and MEV (simplified)
        sandwich_attacks = len([r for r in removals if r > initial_liquidity * 0.1])
        mev_activity = sum(removals) / max(initial_liquidity, 1)
        
        return LiquidityPattern(
            initial_liquidity=initial_liquidity,
            liquidity_changes=changes,
            largest_removal=largest_removal,
            removal_frequency=removal_frequency,
            sandwich_attacks=sandwich_attacks,
            mev_activity=mev_activity
        )

    def analyze_token_vesting(self, token_info: Dict, 
                            transaction_history: List[Dict]) -> VestingAnalysis:
        """Analyze token vesting schedule and team allocations"""
        
        # This is a simplified analysis - would need more sophisticated parsing
        # in production to read actual vesting contracts
        
        has_vesting = False
        total_vested = 0
        unlock_schedule = []
        team_allocation = 0
        
        # Look for vesting-related transactions or metadata
        for tx in transaction_history:
            # Simplified vesting detection
            if 'vesting' in str(tx).lower() or 'lock' in str(tx).lower():
                has_vesting = True
                break
        
        # Estimate team allocation based on initial distribution
        # (This would need actual token account analysis)
        if token_info.get('supply', 0) > 0:
            team_allocation = 0.15  # Assume 15% team allocation (common)
            total_vested = token_info['supply'] * team_allocation
        
        return VestingAnalysis(
            has_vesting=has_vesting,
            total_vested_tokens=total_vested,
            unlock_schedule=unlock_schedule,
            cliff_period=None,
            linear_vesting=True,
            team_allocation=team_allocation
        )

    def calculate_sophisticated_risk_score(self, 
                                         meme_profile: MemeCoinProfile,
                                         wash_trading: Tuple[bool, float],
                                         bundles: List[BundleTransaction],
                                         wallet_cluster: WalletCluster,
                                         liquidity_pattern: LiquidityPattern,
                                         vesting: VestingAnalysis) -> Dict[str, float]:
        """Calculate advanced risk scores based on sophisticated analysis"""
        
        risks = {}
        
        # Meme coin authenticity risk
        risks['meme_authenticity'] = (1 - meme_profile.authenticity_score) * 100
        risks['copycat_risk'] = meme_profile.copycat_score * 100
        
        # Manipulation risks
        risks['wash_trading_risk'] = wash_trading[1] * 100
        risks['bundle_manipulation'] = min(100, len(bundles) * 20)
        
        # Coordination risks
        risks['wallet_coordination'] = wallet_cluster.connection_strength * 100
        
        # Liquidity risks
        risks['liquidity_manipulation'] = min(100, liquidity_pattern.mev_activity * 50)
        
        # Vesting risks
        if vesting.has_vesting:
            risks['vesting_risk'] = max(0, (vesting.team_allocation - 0.1) * 200)  # Risk if >10%
        else:
            risks['vesting_risk'] = 30  # No vesting is suspicious
        
        return risks

def integrate_advanced_patterns(token_address: str, basic_analysis: Dict) -> Dict:
    """Integrate advanced pattern detection with basic analysis"""
    
    detector = AdvancedPatternDetector()
    
    # Extract basic info
    token_info = basic_analysis.get('token_info', {})
    token_name = getattr(token_info, 'name', 'Unknown')
    token_symbol = getattr(token_info, 'symbol', 'UNK')
    
    # Advanced analyses
    meme_profile = detector.analyze_meme_coin_profile(token_name, token_symbol)
    
    # Mock transaction data (in production, fetch real data)
    mock_transactions = []
    wash_trading = detector.detect_wash_trading_pattern(mock_transactions)
    bundles = detector.detect_bundle_manipulation(mock_transactions)
    
    # Mock related addresses (in production, analyze on-chain connections)
    related_addresses = []
    wallet_cluster = detector.analyze_deployer_wallet_cluster(
        getattr(token_info, 'deployer', 'Unknown'), related_addresses
    )
    
    # Mock liquidity events
    liquidity_events = []
    liquidity_pattern = detector.detect_liquidity_manipulation(liquidity_events)
    
    # Vesting analysis
    vesting = detector.analyze_token_vesting(token_info.__dict__ if hasattr(token_info, '__dict__') else {}, mock_transactions)
    
    # Calculate sophisticated risk scores
    advanced_risks = detector.calculate_sophisticated_risk_score(
        meme_profile, wash_trading, bundles, wallet_cluster, liquidity_pattern, vesting
    )
    
    # Integrate with basic analysis
    enhanced_analysis = basic_analysis.copy()
    enhanced_analysis['advanced_patterns'] = {
        'meme_profile': meme_profile,
        'wash_trading_detected': wash_trading[0],
        'wash_trading_score': wash_trading[1],
        'bundle_count': len(bundles),
        'wallet_cluster_size': len(wallet_cluster.wallet_addresses),
        'liquidity_manipulation_score': liquidity_pattern.mev_activity,
        'has_vesting': vesting.has_vesting,
        'advanced_risk_scores': advanced_risks
    }
    
    return enhanced_analysis

# Usage example:
if __name__ == "__main__":
    detector = AdvancedPatternDetector()
    
    # Test meme coin analysis
    profile = detector.analyze_meme_coin_profile("SafeMoonDogeCoin", "SMDOGE")
    print(f"Copycat Score: {profile.copycat_score:.2f}")
    print(f"Authenticity: {profile.authenticity_score:.2f}")
    print(f"Viral Potential: {profile.viral_potential:.2f}")