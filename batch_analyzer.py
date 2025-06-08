"""
Batch Analysis System for Multiple Token Risk Assessment
Allows users to analyze portfolios, watchlists, or groups of tokens
"""

import asyncio
import concurrent.futures
import json
import csv
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
import time
import statistics

from rugcheck import RugChecker, RiskScore, TokenInfo
from advanced_patterns import AdvancedPatternDetector, integrate_advanced_patterns

logger = logging.getLogger(__name__)

@dataclass
class BatchAnalysisRequest:
    tokens: List[str]
    analysis_type: str  # 'portfolio', 'watchlist', 'category'
    priority: str = 'normal'  # 'high', 'normal', 'low'
    include_advanced: bool = True
    max_concurrent: int = 5

@dataclass
class TokenSummary:
    address: str
    symbol: str
    name: str
    risk_score: int
    risk_level: str
    top_risk_factors: List[str]
    market_cap: float
    age_days: int
    analysis_timestamp: str
    analysis_duration: float  # seconds

@dataclass
class BatchAnalysisResult:
    request_id: str
    total_tokens: int
    successful_analyses: int
    failed_analyses: int
    analysis_duration: float
    token_summaries: List[TokenSummary]
    portfolio_risk_score: float
    risk_distribution: Dict[str, int]
    top_risks_overall: List[str]
    recommendations: List[str]

@dataclass
class PortfolioMetrics:
    total_value: float
    weighted_risk_score: float
    diversification_score: float
    correlation_matrix: Dict[str, Dict[str, float]]
    risk_adjusted_return: float
    var_95: float  # Value at Risk 95%

class BatchTokenAnalyzer:
    """Analyze multiple tokens efficiently with concurrent processing"""
    
    def __init__(self):
        self.rug_checker = RugChecker()
        self.pattern_detector = AdvancedPatternDetector()
        self.max_workers = 10
        self.rate_limit_delay = 0.5  # seconds between API calls
        
    async def analyze_batch(self, request: BatchAnalysisRequest) -> BatchAnalysisResult:
        """Analyze multiple tokens concurrently"""
        start_time = time.time()
        request_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting batch analysis {request_id} for {len(request.tokens)} tokens")
        
        # Validate tokens
        valid_tokens = self._validate_token_addresses(request.tokens)
        
        # Analyze tokens concurrently
        token_summaries = await self._analyze_tokens_concurrent(
            valid_tokens, request.max_concurrent, request.include_advanced
        )
        
        # Calculate portfolio metrics
        portfolio_metrics = self._calculate_portfolio_metrics(token_summaries)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(token_summaries, portfolio_metrics)
        
        # Create result
        analysis_duration = time.time() - start_time
        
        result = BatchAnalysisResult(
            request_id=request_id,
            total_tokens=len(request.tokens),
            successful_analyses=len(token_summaries),
            failed_analyses=len(request.tokens) - len(token_summaries),
            analysis_duration=analysis_duration,
            token_summaries=token_summaries,
            portfolio_risk_score=portfolio_metrics.weighted_risk_score,
            risk_distribution=self._calculate_risk_distribution(token_summaries),
            top_risks_overall=self._get_top_risks_overall(token_summaries),
            recommendations=recommendations
        )
        
        logger.info(f"Batch analysis {request_id} completed in {analysis_duration:.2f}s")
        return result
    
    def _validate_token_addresses(self, tokens: List[str]) -> List[str]:
        """Validate and clean token addresses"""
        valid_tokens = []
        
        for token in tokens:
            token = token.strip()
            if len(token) >= 32 and len(token) <= 44:
                valid_tokens.append(token)
            else:
                logger.warning(f"Invalid token address format: {token}")
        
        return valid_tokens
    
    async def _analyze_tokens_concurrent(self, tokens: List[str], 
                                       max_concurrent: int,
                                       include_advanced: bool) -> List[TokenSummary]:
        """Analyze tokens with controlled concurrency"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def analyze_single_token(token_address: str) -> Optional[TokenSummary]:
            async with semaphore:
                try:
                    # Add rate limiting
                    await asyncio.sleep(self.rate_limit_delay)
                    
                    # Run analysis in thread pool to avoid blocking
                    loop = asyncio.get_event_loop()
                    
                    start_time = time.time()
                    
                    # Basic analysis
                    analysis_result = await loop.run_in_executor(
                        None, self.rug_checker.analyze_token, token_address
                    )
                    
                    if not analysis_result['success']:
                        logger.error(f"Analysis failed for {token_address}: {analysis_result['error']}")
                        return None
                    
                    # Advanced analysis if requested
                    if include_advanced:
                        analysis_result = await loop.run_in_executor(
                            None, integrate_advanced_patterns, token_address, analysis_result
                        )
                    
                    analysis_duration = time.time() - start_time
                    
                    # Create summary
                    token_info = analysis_result['token_info']
                    risk_score = analysis_result['risk_score']
                    
                    summary = TokenSummary(
                        address=token_address,
                        symbol=token_info.symbol,
                        name=token_info.name,
                        risk_score=risk_score.total_score,
                        risk_level=self._get_risk_level(risk_score.total_score),
                        top_risk_factors=risk_score.risk_factors[:3],  # Top 3 risks
                        market_cap=analysis_result['metrics'].current_mcap,
                        age_days=token_info.age_days,
                        analysis_timestamp=datetime.now().isoformat(),
                        analysis_duration=analysis_duration
                    )
                    
                    logger.info(f"Analyzed {token_info.symbol}: Risk {risk_score.total_score}/100")
                    return summary
                    
                except Exception as e:
                    logger.error(f"Error analyzing token {token_address}: {e}")
                    return None
        
        # Run all analyses concurrently
        tasks = [analyze_single_token(token) for token in tokens]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        token_summaries = [
            result for result in results 
            if isinstance(result, TokenSummary)
        ]
        
        return token_summaries
    
    def _get_risk_level(self, score: int) -> str:
        """Convert risk score to risk level"""
        if score <= 20:
            return "VERY_LOW"
        elif score <= 40:
            return "LOW"
        elif score <= 60:
            return "MEDIUM"
        elif score <= 80:
            return "HIGH"
        else:
            return "VERY_HIGH"
    
    def _calculate_portfolio_metrics(self, summaries: List[TokenSummary]) -> PortfolioMetrics:
        """Calculate portfolio-level risk metrics"""
        if not summaries:
            return PortfolioMetrics(0, 0, 0, {}, 0, 0)
        
        # Calculate weighted risk score (by market cap)
        total_market_cap = sum(s.market_cap for s in summaries if s.market_cap > 0)
        
        if total_market_cap > 0:
            weighted_risk = sum(
                (s.market_cap / total_market_cap) * s.risk_score 
                for s in summaries if s.market_cap > 0
            )
        else:
            # Equal weight if no market cap data
            weighted_risk = sum(s.risk_score for s in summaries) / len(summaries)
        
        # Calculate diversification score
        diversification = self._calculate_diversification_score(summaries)
        
        # Simplified correlation matrix (would need price data for real correlation)
        correlation_matrix = self._calculate_correlation_matrix(summaries)
        
        # Risk-adjusted return (simplified)
        risk_adjusted_return = max(0, 100 - weighted_risk) / 100
        
        # Value at Risk 95% (simplified)
        risk_scores = [s.risk_score for s in summaries]
        var_95 = statistics.quantile(risk_scores, 0.95) if len(risk_scores) > 1 else weighted_risk
        
        return PortfolioMetrics(
            total_value=total_market_cap,
            weighted_risk_score=weighted_risk,
            diversification_score=diversification,
            correlation_matrix=correlation_matrix,
            risk_adjusted_return=risk_adjusted_return,
            var_95=var_95
        )
    
    def _calculate_diversification_score(self, summaries: List[TokenSummary]) -> float:
        """Calculate portfolio diversification score"""
        if len(summaries) <= 1:
            return 0.0
        
        # Age diversification
        ages = [s.age_days for s in summaries]
        age_std = statistics.stdev(ages) if len(ages) > 1 else 0
        age_score = min(1.0, age_std / 100)  # Normalize
        
        # Risk diversification
        risk_scores = [s.risk_score for s in summaries]
        risk_std = statistics.stdev(risk_scores) if len(risk_scores) > 1 else 0
        risk_score = min(1.0, risk_std / 30)  # Normalize
        
        # Market cap diversification
        market_caps = [s.market_cap for s in summaries if s.market_cap > 0]
        if len(market_caps) > 1:
            log_caps = [math.log(cap) for cap in market_caps if cap > 0]
            cap_std = statistics.stdev(log_caps) if len(log_caps) > 1 else 0
            cap_score = min(1.0, cap_std / 5)  # Normalize
        else:
            cap_score = 0.0
        
        # Combined diversification score
        return (age_score + risk_score + cap_score) / 3
    
    def _calculate_correlation_matrix(self, summaries: List[TokenSummary]) -> Dict[str, Dict[str, float]]:
        """Calculate simplified correlation matrix"""
        matrix = {}
        
        for i, summary1 in enumerate(summaries):
            matrix[summary1.symbol] = {}
            for j, summary2 in enumerate(summaries):
                if i == j:
                    correlation = 1.0
                else:
                    # Simplified correlation based on risk similarity
                    risk_diff = abs(summary1.risk_score - summary2.risk_score)
                    correlation = max(0.0, 1.0 - (risk_diff / 100))
                
                matrix[summary1.symbol][summary2.symbol] = correlation
        
        return matrix
    
    def _calculate_risk_distribution(self, summaries: List[TokenSummary]) -> Dict[str, int]:
        """Calculate distribution of risk levels"""
        distribution = {
            "VERY_LOW": 0,
            "LOW": 0,
            "MEDIUM": 0,
            "HIGH": 0,
            "VERY_HIGH": 0
        }
        
        for summary in summaries:
            distribution[summary.risk_level] += 1
        
        return distribution
    
    def _get_top_risks_overall(self, summaries: List[TokenSummary]) -> List[str]:
        """Get most common risk factors across all tokens"""
        risk_factor_counts = {}
        
        for summary in summaries:
            for factor in summary.top_risk_factors:
                risk_factor_counts[factor] = risk_factor_counts.get(factor, 0) + 1
        
        # Sort by frequency
        sorted_risks = sorted(
            risk_factor_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return [risk for risk, count in sorted_risks[:5]]
    
    def _generate_recommendations(self, summaries: List[TokenSummary], 
                                metrics: PortfolioMetrics) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # High-risk token warnings
        high_risk_tokens = [s for s in summaries if s.risk_score > 70]
        if high_risk_tokens:
            recommendations.append(
                f"⚠️ Consider removing {len(high_risk_tokens)} high-risk tokens: " +
                ", ".join([f"${s.symbol}" for s in high_risk_tokens[:3]])
            )
        
        # Portfolio risk assessment
        if metrics.weighted_risk_score > 60:
            recommendations.append(
                "🔴 Portfolio has high overall risk - consider rebalancing toward safer assets"
            )
        elif metrics.weighted_risk_score < 30:
            recommendations.append(
                "🟢 Portfolio has conservative risk profile - consider small allocation to higher potential tokens"
            )
        
        # Diversification advice
        if metrics.diversification_score < 0.3:
            recommendations.append(
                "📊 Low diversification detected - consider tokens with different risk profiles and ages"
            )
        
        # Age-based recommendations
        very_new_tokens = [s for s in summaries if s.age_days < 7]
        if len(very_new_tokens) > len(summaries) * 0.5:
            recommendations.append(
                "🕐 Many very new tokens detected - consider mixing with more established projects"
            )
        
        # Common risk factors
        common_risks = self._get_top_risks_overall(summaries)
        if "High-risk deployer with history of failed tokens" in common_risks:
            recommendations.append(
                "👤 Multiple tokens from risky deployers detected - research deployer history"
            )
        
        if not recommendations:
            recommendations.append("✅ Portfolio shows reasonable risk distribution")
        
        return recommendations

class PortfolioTracker:
    """Track and analyze token portfolios over time"""
    
    def __init__(self):
        self.analyzer = BatchTokenAnalyzer()
        self.portfolios = {}  # In production, use database
    
    def create_portfolio(self, portfolio_id: str, tokens: List[str], 
                        metadata: Dict = None) -> Dict:
        """Create a new portfolio for tracking"""
        portfolio = {
            'id': portfolio_id,
            'tokens': tokens,
            'created_at': datetime.now().isoformat(),
            'metadata': metadata or {},
            'analysis_history': []
        }
        
        self.portfolios[portfolio_id] = portfolio
        return portfolio
    
    async def analyze_portfolio(self, portfolio_id: str) -> BatchAnalysisResult:
        """Analyze an existing portfolio"""
        if portfolio_id not in self.portfolios:
            raise ValueError(f"Portfolio {portfolio_id} not found")
        
        portfolio = self.portfolios[portfolio_id]
        
        request = BatchAnalysisRequest(
            tokens=portfolio['tokens'],
            analysis_type='portfolio',
            include_advanced=True
        )
        
        result = await self.analyzer.analyze_batch(request)
        
        # Store analysis in history
        portfolio['analysis_history'].append({
            'timestamp': datetime.now().isoformat(),
            'result': asdict(result)
        })
        
        return result
    
    def export_portfolio_report(self, portfolio_id: str, format: str = 'json') -> str:
        """Export portfolio analysis as report"""
        if portfolio_id not in self.portfolios:
            raise ValueError(f"Portfolio {portfolio_id} not found")
        
        portfolio = self.portfolios[portfolio_id]
        
        if format == 'json':
            return json.dumps(portfolio, indent=2)
        elif format == 'csv':
            return self._export_csv(portfolio)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_csv(self, portfolio: Dict) -> str:
        """Export portfolio as CSV"""
        if not portfolio['analysis_history']:
            return "No analysis data available"
        
        latest_analysis = portfolio['analysis_history'][-1]['result']
        summaries = latest_analysis['token_summaries']
        
        # Create CSV content
        csv_lines = []
        csv_lines.append("Symbol,Name,Address,Risk Score,Risk Level,Market Cap,Age Days,Top Risk Factor")
        
        for summary in summaries:
            top_risk = summary['top_risk_factors'][0] if summary['top_risk_factors'] else "None"
            csv_lines.append(
                f"{summary['symbol']},{summary['name']},{summary['address']},"
                f"{summary['risk_score']},{summary['risk_level']},{summary['market_cap']},"
                f"{summary['age_days']},\"{top_risk}\""
            )
        
        return "\n".join(csv_lines)

# Usage example and testing
async def example_batch_analysis():
    """Example of how to use the batch analyzer"""
    
    analyzer = BatchTokenAnalyzer()
    
    # Example token list (mix of known tokens)
    tokens = [
        "So11111111111111111111111111111111111111112",  # SOL
        "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
        "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",  # BONK
    ]
    
    request = BatchAnalysisRequest(
        tokens=tokens,
        analysis_type='portfolio',
        include_advanced=True,
        max_concurrent=3
    )
    
    print("🔍 Starting batch analysis...")
    result = await analyzer.analyze_batch(request)
    
    print(f"\n📊 BATCH ANALYSIS RESULTS")
    print(f"{'='*50}")
    print(f"Request ID: {result.request_id}")
    print(f"Total Tokens: {result.total_tokens}")
    print(f"Successful: {result.successful_analyses}")
    print(f"Failed: {result.failed_analyses}")
    print(f"Duration: {result.analysis_duration:.2f}s")
    print(f"Portfolio Risk Score: {result.portfolio_risk_score:.1f}/100")
    
    print(f"\n📈 RISK DISTRIBUTION:")
    for risk_level, count in result.risk_distribution.items():
        print(f"  {risk_level}: {count} tokens")
    
    print(f"\n⚠️ TOP PORTFOLIO RISKS:")
    for risk in result.top_risks_overall:
        print(f"  • {risk}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    for rec in result.recommendations:
        print(f"  {rec}")
    
    print(f"\n🪙 TOKEN DETAILS:")
    for summary in result.token_summaries:
        print(f"  ${summary.symbol}: {summary.risk_score}/100 ({summary.risk_level})")

if __name__ == "__main__":
    import math  # Add missing import
    asyncio.run(example_batch_analysis())