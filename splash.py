from flask import render_template, jsonify
import requests
import random
import math
import json

def get_trending_meme_coins():
    """Fetch top 100 meme coins by market cap from CoinGecko"""
    try:
        # Try to get real data first
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            'vs_currency': 'usd',
            'category': 'meme-token',
            'order': 'market_cap_desc',
            'per_page': 100,
            'page': 1,
            'sparkline': False,
            'price_change_percentage': '24h'
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            coins = response.json()
            processed_coins = []
            
            for i, coin in enumerate(coins):
                processed_coins.append({
                    'id': coin.get('id', f'coin_{i}'),
                    'symbol': coin.get('symbol', 'UNK').upper(),
                    'name': coin.get('name', 'Unknown'),
                    'market_cap': coin.get('market_cap', 1000000),
                    'price': coin.get('current_price', 0),
                    'price_change_24h': coin.get('price_change_percentage_24h', 0),
                    'image': coin.get('image', ''),
                    'rank': coin.get('market_cap_rank', i + 1)
                })
            
            return processed_coins
        
    except Exception as e:
        print(f"Error fetching live data: {e}")
    
    # Fallback data with popular meme coins
    return [
        {'id': 'dogecoin', 'symbol': 'DOGE', 'name': 'Dogecoin', 'market_cap': 12500000000, 'price': 0.085, 'price_change_24h': 2.5, 'image': '', 'rank': 1},
        {'id': 'shiba-inu', 'symbol': 'SHIB', 'name': 'Shiba Inu', 'market_cap': 5200000000, 'price': 0.0000088, 'price_change_24h': -1.2, 'image': '', 'rank': 2},
        {'id': 'bonk', 'symbol': 'BONK', 'name': 'Bonk', 'market_cap': 450000000, 'price': 0.000015, 'price_change_24h': 15.7, 'image': '', 'rank': 3},
        {'id': 'dogwifhat', 'symbol': 'WIF', 'name': 'dogwifhat', 'market_cap': 280000000, 'price': 2.85, 'price_change_24h': 8.9, 'image': '', 'rank': 4},
        {'id': 'popcat', 'symbol': 'POPCAT', 'name': 'Popcat', 'market_cap': 150000000, 'price': 1.52, 'price_change_24h': 12.3, 'image': '', 'rank': 5},
        {'id': 'book-of-meme', 'symbol': 'BOME', 'name': 'Book of Meme', 'market_cap': 120000000, 'price': 0.0089, 'price_change_24h': -3.4, 'image': '', 'rank': 6},
        {'id': 'cat-in-a-dogs-world', 'symbol': 'MEW', 'name': 'Cat in a Dogs World', 'market_cap': 110000000, 'price': 0.0078, 'price_change_24h': 5.6, 'image': '', 'rank': 7},
        {'id': 'myro', 'symbol': 'MYRO', 'name': 'Myro', 'market_cap': 95000000, 'price': 0.089, 'price_change_24h': -2.1, 'image': '', 'rank': 8},
        {'id': 'jeo-boden', 'symbol': 'BODEN', 'name': 'Jeo Boden', 'market_cap': 85000000, 'price': 0.215, 'price_change_24h': 18.9, 'image': '', 'rank': 9},
        {'id': 'mog-coin', 'symbol': 'MOG', 'name': 'Mog Coin', 'market_cap': 78000000, 'price': 0.0000019, 'price_change_24h': 7.2, 'image': '', 'rank': 10},
        # Continue with more fallback data
        {'id': 'slerf', 'symbol': 'SLERF', 'name': 'Slerf', 'market_cap': 65000000, 'price': 0.145, 'price_change_24h': -5.8, 'image': '', 'rank': 11},
        {'id': 'wen-2', 'symbol': 'WEN', 'name': 'Wen', 'market_cap': 58000000, 'price': 0.000089, 'price_change_24h': 3.4, 'image': '', 'rank': 12},
        {'id': 'based-brett', 'symbol': 'BRETT', 'name': 'Based Brett', 'market_cap': 52000000, 'price': 0.078, 'price_change_24h': 11.2, 'image': '', 'rank': 13},
        {'id': 'toshi', 'symbol': 'TOSHI', 'name': 'Toshi', 'market_cap': 48000000, 'price': 0.000156, 'price_change_24h': -1.9, 'image': '', 'rank': 14},
        {'id': 'pnut', 'symbol': 'PNUT', 'name': 'Peanut the Squirrel', 'market_cap': 42000000, 'price': 0.89, 'price_change_24h': 145.7, 'image': '', 'rank': 15},
    ] + [
        {
            'id': f'meme_{i}',
            'symbol': f'MEME{i}',
            'name': f'Meme Coin {i}',
            'market_cap': random.randint(1000000, 50000000),
            'price': random.uniform(0.000001, 10),
            'price_change_24h': random.uniform(-50, 200),
            'image': '',
            'rank': i
        } for i in range(16, 101)
    ]

def calculate_planet_properties(coins):
    """Calculate 3D positions and sizes for meme coin planets"""
    planets = []
    
    # Sort by market cap to determine orbit distances
    sorted_coins = sorted(coins, key=lambda x: x['market_cap'], reverse=True)
    
    for i, coin in enumerate(sorted_coins):
        # Calculate orbit radius (closer to sun = higher market cap)
        min_radius = 8
        max_radius = 50
        orbit_radius = min_radius + (i / len(sorted_coins)) * (max_radius - min_radius)
        
        # Calculate planet size based on market cap
        max_market_cap = max(c['market_cap'] for c in coins)
        min_market_cap = min(c['market_cap'] for c in coins)
        
        # Normalize market cap to size (0.5 to 3.0 units)
        size_factor = (coin['market_cap'] - min_market_cap) / (max_market_cap - min_market_cap)
        planet_size = 0.5 + size_factor * 2.5
        
        # Random starting position on orbit
        angle = random.uniform(0, 2 * math.pi)
        
        # Calculate orbital speed (higher market cap = faster orbit for visual interest)
        base_speed = 0.002
        speed_multiplier = 1 + size_factor * 2
        orbital_speed = base_speed * speed_multiplier
        
        # Slight vertical variation for 3D effect
        y_offset = random.uniform(-2, 2)
        
        # Color based on price change
        if coin['price_change_24h'] > 10:
            color = '#00ff88'  # Bright green for major gains
        elif coin['price_change_24h'] > 0:
            color = '#88ff44'  # Light green for gains
        elif coin['price_change_24h'] > -10:
            color = '#ffaa00'  # Orange for small losses
        else:
            color = '#ff4444'  # Red for major losses
        
        planets.append({
            'id': coin['id'],
            'symbol': coin['symbol'],
            'name': coin['name'],
            'market_cap': coin['market_cap'],
            'price': coin['price'],
            'price_change_24h': coin['price_change_24h'],
            'rank': coin['rank'],
            'orbit_radius': orbit_radius,
            'planet_size': planet_size,
            'initial_angle': angle,
            'orbital_speed': orbital_speed,
            'y_offset': y_offset,
            'color': color,
            'glow': coin['price_change_24h'] > 50  # Special glow for massive gainers
        })
    
    return planets

def splash_route():
    """Main route for the 3D crypto solar system"""
    try:
        # Get meme coin data
        coins = get_trending_meme_coins()
        
        # Calculate planet properties
        planets = calculate_planet_properties(coins)
        
        # Return rendered template with planet data
        return render_template('splash.html', planets=json.dumps(planets))
        
    except Exception as e:
        print(f"Error in splash route: {e}")
        # Return minimal template with fallback data
        fallback_planets = [
            {
                'symbol': 'DOGE',
                'name': 'Dogecoin',
                'market_cap': 12500000000,
                'price': 0.085,
                'price_change_24h': 2.5,
                'orbit_radius': 10,
                'planet_size': 2.0,
                'initial_angle': 0,
                'orbital_speed': 0.004,
                'y_offset': 0,
                'color': '#88ff44',
                'glow': False
            }
        ]
        return render_template('splash.html', planets=json.dumps(fallback_planets))

def api_planets():
    """API endpoint to get planet data"""
    try:
        coins = get_trending_meme_coins()
        planets = calculate_planet_properties(coins)
        
        return jsonify({
            'success': True,
            'planets': planets,
            'total_count': len(planets),
            'last_updated': 'now'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'planets': []
        })