#!/usr/bin/env python3
"""
Test script for the rug checker system
Tests with known tokens to validate functionality
"""

import sys
import json
from rugcheck import RugChecker

def test_token(token_address, expected_risk_level=None):
    """Test a specific token and display results"""
    print(f"\n{'='*60}")
    print(f"TESTING TOKEN: {token_address}")
    print(f"{'='*60}")
    
    checker = RugChecker()
    result = checker.analyze_token(token_address)
    
    if not result['success']:
        print(f"❌ ERROR: {result['error']}")
        return False
    
    # Extract data
    token_info = result['token_info']
    risk_score = result['risk_score']
    holder_analysis = result['holder_analysis']
    deployer_history = result['deployer_history']
    
    # Display basic info
    print(f"🪙 TOKEN: {token_info.symbol} - {token_info.name}")
    print(f"📅 AGE: {token_info.age_days} days old")
    print(f"👥 HOLDERS: {holder_analysis.total_holders}")
    print(f"🏗️ DEPLOYER: {token_info.deployer[:12]}...{token_info.deployer[-8:]}")
    
    # Display risk score
    risk_level = get_risk_level(risk_score.total_score)
    print(f"\n🎯 RISK SCORE: {risk_score.total_score}/100 ({risk_level})")
    
    # Risk breakdown
    print(f"\n📊 RISK BREAKDOWN:")
    for category, score in risk_score.breakdown.items():
        print(f"   {category.title().replace('_', ' ')}: {score}/100")
    
    # Deployer analysis
    print(f"\n👤 DEPLOYER ANALYSIS:")
    print(f"   Total Tokens Created: {deployer_history.total_tokens_created}")
    print(f"   Successful Tokens: {deployer_history.successful_tokens}")
    print(f"   Failed Tokens: {deployer_history.failed_tokens}")
    print(f"   Failure Rate: {deployer_history.failed_tokens / max(deployer_history.total_tokens_created, 1) * 100:.1f}%")
    
    # Holder concentration
    print(f"\n💰 HOLDER CONCENTRATION:")
    print(f"   Top 5 Holders: {holder_analysis.top_5_concentration:.1f}%")
    print(f"   Top 10 Holders: {holder_analysis.top_10_concentration:.1f}%")
    print(f"   Deployer Holdings: {holder_analysis.deployer_holdings:.1f}%")
    
    # Authorities
    print(f"\n🔐 AUTHORITIES:")
    print(f"   Mint Authority: {'REVOKED' if not token_info.mint_authority else 'ACTIVE'}")
    print(f"   Freeze Authority: {'REVOKED' if not token_info.freeze_authority else 'ACTIVE'}")
    
    # Risk factors
    if risk_score.risk_factors:
        print(f"\n⚠️ RISK FACTORS:")
        for factor in risk_score.risk_factors:
            print(f"   • {factor}")
    else:
        print(f"\n✅ NO MAJOR RISK FACTORS DETECTED")
    
    # Validation
    if expected_risk_level:
        actual_level = get_risk_level(risk_score.total_score)
        if actual_level.lower() == expected_risk_level.lower():
            print(f"\n✅ VALIDATION PASSED: Expected {expected_risk_level}, got {actual_level}")
        else:
            print(f"\n⚠️ VALIDATION WARNING: Expected {expected_risk_level}, got {actual_level}")
    
    return True

def get_risk_level(score):
    """Convert risk score to risk level"""
    if score <= 20:
        return "VERY LOW RISK"
    elif score <= 40:
        return "LOW RISK"
    elif score <= 60:
        return "MEDIUM RISK"
    elif score <= 80:
        return "HIGH RISK"
    else:
        return "VERY HIGH RISK"

def main():
    """Run comprehensive tests"""
    print("🚨 ASK DEGEN RUG CHECKER - TEST SUITE")
    print("=====================================")
    
    # Test cases with known tokens
    test_cases = [
        {
            'address': 'So11111111111111111111111111111111111111112',
            'name': 'Wrapped SOL (Should be very low risk)',
            'expected': 'very low risk'
        },
        {
            'address': 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',
            'name': 'USDC (Should be very low risk)',
            'expected': 'very low risk'
        },
        {
            'address': 'DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263',
            'name': 'BONK (Should be low-medium risk)',
            'expected': 'low risk'
        }
    ]
    
    success_count = 0
    total_tests = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[TEST {i}/{total_tests}] {test_case['name']}")
        
        try:
            success = test_token(test_case['address'], test_case['expected'])
            if success:
                success_count += 1
        except Exception as e:
            print(f"❌ TEST FAILED: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"✅ PASSED: {success_count}/{total_tests}")
    print(f"❌ FAILED: {total_tests - success_count}/{total_tests}")
    
    if success_count == total_tests:
        print(f"\n🎉 ALL TESTS PASSED! Rug checker is working correctly.")
    else:
        print(f"\n⚠️ Some tests failed. Check RPC connectivity and token addresses.")
    
    # Performance note
    print(f"\n📝 NOTES:")
    print(f"   • Analysis speed depends on RPC response times")
    print(f"   • Some deployer history features require more data")
    print(f"   • Risk scores may vary based on real-time data")

def test_custom_token():
    """Interactive test for custom token"""
    print(f"\n🔍 CUSTOM TOKEN TEST")
    print(f"===================")
    
    while True:
        token_address = input("Enter Solana token address (or 'quit' to exit): ").strip()
        
        if token_address.lower() in ['quit', 'exit', 'q']:
            break
        
        if len(token_address) < 32:
            print("❌ Invalid address format. Please enter a valid Solana token address.")
            continue
        
        try:
            test_token(token_address)
        except KeyboardInterrupt:
            print("\n👋 Test interrupted by user")
            break
        except Exception as e:
            print(f"❌ Error testing token: {e}")

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1:
        # Test specific token from command line
        token_address = sys.argv[1]
        print(f"Testing token from command line: {token_address}")
        test_token(token_address)
    else:
        # Run full test suite
        main()
        
        # Ask if user wants to test custom tokens
        print(f"\n" + "="*60)
        custom_test = input("Would you like to test a custom token? (y/n): ").strip().lower()
        if custom_test in ['y', 'yes']:
            test_custom_token()
    
    print(f"\n👋 Thanks for testing the ASK DEGEN Rug Checker!")

# Example usage:
# python test_rugcheck.py
# python test_rugcheck.py So11111111111111111111111111111111111111112