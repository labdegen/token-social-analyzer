// Dashboard JavaScript for ASK DEGEN
let currentAnalysis = null;
let currentTokenAddress = '';
let chatHistory = [];
let currentTab = 'fresh-hype';

// Initialize dashboard on page load
document.addEventListener('DOMContentLoaded', function() {
    loadMarketOverview();
    loadTrendingTokens();
    loadCryptoNews();
    loadTopKOLs();
    
    // Allow Enter key for inputs
    document.getElementById('tokenAddress').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') analyzeToken();
    });
    document.getElementById('chatInput').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') sendChatMessage();
    });
});

// Market Overview Functions
async function loadMarketOverview() {
    try {
        const response = await fetch('/market-overview');
        const data = await response.json();
        
        if (data.success) {
            updateMarketOverview(data);
        }
    } catch (error) {
        console.error('Market overview error:', error);
        // Show fallback data
        updateMarketOverview({
            bitcoin_price: 95000,
            ethereum_price: 3500,
            solana_price: 180,
            market_sentiment: 'Bullish',
            fear_greed_index: 72
        });
    }
}

function updateMarketOverview(data) {
    // Update Fear & Greed Index
    document.getElementById('fearGreedIndex').textContent = data.fear_greed_index + '/100';
    
    const sentimentEl = document.getElementById('sentimentIndicator');
    sentimentEl.textContent = data.market_sentiment;
    sentimentEl.className = `sentiment-indicator sentiment-${data.market_sentiment.toLowerCase()}`;
    
    // Update crypto prices
    const pricesHtml = `
        <div class="crypto-price-card">
            <div class="crypto-symbol">BTC</div>
            <div class="crypto-price">$${data.bitcoin_price?.toLocaleString() || '95,000'}</div>
            <div class="crypto-change positive">+2.3%</div>
        </div>
        <div class="crypto-price-card">
            <div class="crypto-symbol">ETH</div>
            <div class="crypto-price">$${data.ethereum_price?.toLocaleString() || '3,500'}</div>
            <div class="crypto-change positive">+1.8%</div>
        </div>
        <div class="crypto-price-card">
            <div class="crypto-symbol">SOL</div>
            <div class="crypto-price">$${data.solana_price?.toLocaleString() || '180'}</div>
            <div class="crypto-change positive">+4.2%</div>
        </div>
        <div class="crypto-price-card">
            <div class="crypto-symbol">Market Cap</div>
            <div class="crypto-price">$${(data.total_market_cap / 1e12)?.toFixed(2) || '2.3'}T</div>
            <div class="crypto-change positive">+0.9%</div>
        </div>
    `;
    document.getElementById('cryptoPrices').innerHTML = pricesHtml;
}

// Trending Tokens Functions
async function loadTrendingTokens() {
    // Load all tabs on initial load
    await loadFreshHype();
    await loadRecentTrending();
    await loadBlueChips();
}

// Tab switching function
function switchTab(tabName) {
    currentTab = tabName;
    
    // Update tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
    
    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    document.getElementById(tabName).classList.add('active');
}

// Refresh current tab
function refreshCurrentTab() {
    const refreshBtn = document.querySelector('.section-refresh');
    if (refreshBtn) {
        refreshBtn.style.transform = 'rotate(180deg)';
        setTimeout(() => {
            refreshBtn.style.transform = '';
        }, 500);
    }
    
    switch(currentTab) {
        case 'fresh-hype':
            loadFreshHype(true);
            break;
        case 'recent-trending':
            loadRecentTrending(true);
            break;
        case 'blue-chips':
            loadBlueChips(true);
            break;
    }
}

// Fresh Hype tokens
async function loadFreshHype(forceRefresh = false) {
    try {
        const response = await fetch(`/trending-tokens?category=fresh-hype&refresh=${forceRefresh}`);
        const data = await response.json();
        
        if (data.success && data.tokens) {
            displayTokensInGrid(data.tokens, 'freshHypeGrid', 'fresh-hype');
        }
    } catch (error) {
        console.error('Fresh hype tokens error:', error);
        document.getElementById('freshHypeGrid').innerHTML = '<div style="text-align: center; color: #ef4444; padding: 40px;">Error loading fresh hype tokens</div>';
    }
}

// Recent Trending tokens
async function loadRecentTrending(forceRefresh = false) {
    try {
        const response = await fetch(`/trending-tokens?category=recent-trending&refresh=${forceRefresh}`);
        const data = await response.json();
        
        if (data.success && data.tokens) {
            displayTokensInGrid(data.tokens, 'recentTrendingGrid', 'recent-trending');
        }
    } catch (error) {
        console.error('Recent trending tokens error:', error);
        document.getElementById('recentTrendingGrid').innerHTML = '<div style="text-align: center; color: #ef4444; padding: 40px;">Error loading trending tokens</div>';
    }
}

// Blue Chip tokens
async function loadBlueChips(forceRefresh = false) {
    try {
        const response = await fetch(`/trending-tokens?category=blue-chips&refresh=${forceRefresh}`);
        const data = await response.json();
        
        if (data.success && data.tokens) {
            displayTokensInGrid(data.tokens, 'blueChipsGrid', 'blue-chip');
        }
    } catch (error) {
        console.error('Blue chip tokens error:', error);
        document.getElementById('blueChipsGrid').innerHTML = '<div style="text-align: center; color: #ef4444; padding: 40px;">Error loading blue chip tokens</div>';
    }
}

function displayTokensInGrid(tokens, gridId, category) {
    let tokensHtml = '';
    
    tokens.slice(0, 12).forEach(token => {
        const changeClass = token.price_change >= 0 ? 'positive' : 'negative';
        const changeIcon = token.price_change >= 0 ? '🚀' : '📉';
        
        tokensHtml += `
            <div class="trending-token ${category}" onclick="loadTokenAnalysis('${token.address}')">
                <div class="trending-symbol">${token.symbol}</div>
                <div class="trending-change ${changeClass}">
                    ${changeIcon} ${token.price_change >= 0 ? '+' : ''}${token.price_change.toFixed(1)}%
                </div>
                <div class="trending-mentions">${token.mentions || Math.floor(token.volume / 1000)} ${category === 'blue-chip' ? 'vol' : 'mentions'}</div>
            </div>
        `;
    });
    
    document.getElementById(gridId).innerHTML = tokensHtml;
}

// Top KOLs Functions
async function loadTopKOLs() {
    try {
        const response = await fetch('/top-kols');
        const data = await response.json();
        
        if (data.success && data.kols) {
            displayTopKOLs(data.kols);
        } else {
            const kolsList = document.getElementById('kolsList');
            if (kolsList) {
                kolsList.innerHTML = '<div style="text-align: center; color: #ef4444; padding: 20px;">Error loading KOLs</div>';
            }
        }
    } catch (error) {
        console.error('Top KOLs error:', error);
        const kolsList = document.getElementById('kolsList');
        if (kolsList) {
            kolsList.innerHTML = '<div style="text-align: center; color: #ef4444; padding: 20px;">Error loading KOLs</div>';
        }
    }
}

function displayTopKOLs(kols) {
    const kolsList = document.getElementById('kolsList');
    if (!kolsList) return;
    
    let kolsHtml = '';
    
    kols.slice(0, 20).forEach(kol => {
        const parts = kol.split(' - ');
        const accountInfo = parts[0];
        const description = parts[1];
        const url = parts[2] || '#';
        
        kolsHtml += `
            <div class="kol-item" onclick="window.open('${url}', '_blank')">
                <div class="kol-account">${accountInfo}</div>
                <div class="kol-description">${description}</div>
            </div>
        `;
    });
    
    kolsList.innerHTML = kolsHtml;
}

// Toggle Functions
function toggleNewsSection() {
    const newsContent = document.getElementById('newsContent');
    const newsToggle = document.getElementById('newsToggle');
    
    if (newsContent.style.display === 'none') {
        newsContent.style.display = 'block';
        newsToggle.textContent = '▼';
    } else {
        newsContent.style.display = 'none';
        newsToggle.textContent = '▶';
    }
}

function toggleKOLsSection() {
    const kolsContent = document.getElementById('kolsContent');
    const kolsToggle = document.getElementById('kolsToggle');
    
    if (kolsContent && kolsToggle) {
        if (kolsContent.style.display === 'none') {
            kolsContent.style.display = 'block';
            kolsToggle.textContent = '▼';
        } else {
            kolsContent.style.display = 'none';
            kolsToggle.textContent = '▶';
        }
    }
}

function loadTokenAnalysis(address) {
    if (!address) return;
    
    // Set the token address in the input field
    document.getElementById('tokenAddress').value = address;
    
    // Scroll to analyzer
    document.querySelector('.token-analyzer').scrollIntoView({ behavior: 'smooth' });
    
    // Auto-analyze after a short delay
    setTimeout(() => {
        analyzeToken();
    }, 500);
}

// Crypto News Functions
async function loadCryptoNews() {
    try {
        const response = await fetch('/crypto-news');
        const data = await response.json();
        
        if (data.success && data.articles) {
            displayCryptoNews(data.articles);
        } else {
            document.getElementById('newsList').innerHTML = '<div style="text-align: center; color: #ef4444; padding: 20px;">Error loading news</div>';
        }
    } catch (error) {
        console.error('Crypto news error:', error);
        document.getElementById('newsList').innerHTML = '<div style="text-align: center; color: #ef4444; padding: 20px;">Error loading news</div>';
    }
}

function displayCryptoNews(articles) {
    let newsHtml = '';
    
    articles.slice(0, 8).forEach(article => {
        newsHtml += `
            <div class="news-item" onclick="window.open('${article.url}', '_blank')">
                <div class="news-headline">${article.headline}</div>
                <div class="news-summary">${article.summary}</div>
                <div class="news-meta">
                    <span>${article.source}</span>
                    <span>${article.timestamp}</span>
                </div>
            </div>
        `;
    });
    
    document.getElementById('newsList').innerHTML = newsHtml;
}

// Token Analysis Functions
async function analyzeToken() {
    const tokenAddress = document.getElementById('tokenAddress').value.trim();
    
    if (!tokenAddress) {
        showError('Please enter a token address');
        return;
    }

    if (tokenAddress.length < 32 || tokenAddress.length > 44) {
        showError('Invalid Solana token address format');
        return;
    }

    currentTokenAddress = tokenAddress;
    chatHistory = [];

    showLoading();
    hideError();
    hideResults();
    showChart(tokenAddress);

    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                token_address: tokenAddress
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        // Handle streaming response
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const jsonData = line.substring(6);
                        if (jsonData.trim()) {
                            const data = JSON.parse(jsonData);
                            handleStreamMessage(data);
                        }
                    } catch (e) {
                        console.warn('Failed to parse SSE message:', line);
                    }
                }
            }
        }
        
    } catch (error) {
        console.error('Analysis error:', error);
        hideLoading();
        showError('Analysis failed: ' + error.message);
    }
}

function handleStreamMessage(data) {
    if (data.type === 'progress') {
        updateLoadingText(data.message);
    } else if (data.type === 'complete') {
        currentAnalysis = data;
        hideLoading();
        displayResults(data);
    } else if (data.type === 'error') {
        hideLoading();
        showError('Analysis error: ' + data.error);
    }
}

function showChart(tokenAddress) {
    const chartContainer = document.getElementById('chartContainer');
    const iframe = document.getElementById('dexscreenerChart');
    
    iframe.src = `https://dexscreener.com/solana/${tokenAddress}?embed=1&loadChartSettings=0&chartLeftToolbar=0&chartTheme=dark&theme=dark&chartStyle=1&chartType=usd&interval=15`;
    chartContainer.style.display = 'block';
    chartContainer.classList.add('fade-in');
}

function displayResults(data) {
    // Show token profile
    updateTokenProfile(data);
    
    // Show results section
    document.getElementById('resultsSection').style.display = 'block';
    document.getElementById('resultsSection').classList.add('fade-in');

    // Update all sections
    updateMomentumDashboard(data);
    updateTweetsSection(data);
    updateExpertAnalysis(data);
    updateAnalysisCards(data);
    enableChatWithContext();

    // Scroll to results
    document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
}

function updateTokenProfile(data) {
    const profile = document.getElementById('tokenProfile');
    
    // Update token info
    document.getElementById('tokenName').textContent = `$${data.token_symbol} - ${data.token_name || data.token_symbol + ' Token'}`;
    document.getElementById('tokenContract').textContent = `${data.token_address.slice(0, 8)}...${data.token_address.slice(-8)}`;
    
    // Update token image if available
    const tokenImage = document.getElementById('tokenImage');
    if (data.token_image) {
        tokenImage.src = data.token_image;
        tokenImage.style.display = 'block';
    } else {
        tokenImage.style.display = 'none';
    }
    
    // Update metrics
    const metricsHtml = `
        <div class="metric-card">
            <div class="metric-value">$${(data.price_usd || 0).toFixed(8)}</div>
            <div class="metric-label">Price USD</div>
        </div>
        <div class="metric-card">
            <div class="metric-value ${data.price_change_24h >= 0 ? 'positive' : 'negative'}">${data.price_change_24h >= 0 ? '+' : ''}${(data.price_change_24h || 0).toFixed(2)}%</div>
            <div class="metric-label">24h Change</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">$${((data.market_cap || 0) / 1e6).toFixed(1)}M</div>
            <div class="metric-label">Market Cap</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">$${((data.volume_24h || 0) / 1e6).toFixed(1)}M</div>
            <div class="metric-label">24h Volume</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">$${((data.liquidity || 0) / 1e6).toFixed(1)}M</div>
            <div class="metric-label">Liquidity</div>
        </div>
    `;
    document.getElementById('tokenMetrics').innerHTML = metricsHtml;
    
    profile.style.display = 'block';
}

function updateMomentumDashboard(data) {
    const momentumScore = data.social_momentum_score || 50;
    const scoreEl = document.getElementById('momentumScore');
    
    // Animate score counting up
    let currentScore = 0;
    const increment = momentumScore / 60;
    const counter = setInterval(() => {
        currentScore += increment;
        if (currentScore >= momentumScore) {
            currentScore = momentumScore;
            clearInterval(counter);
        }
        scoreEl.textContent = Math.round(currentScore);
    }, 25);

    // Update metrics with visual indicators
    const metrics = data.sentiment_metrics || {};
    const metricsData = [
        { label: 'Bullish', value: Math.round(metrics.bullish_percentage || 0), unit: '%' },
        { label: 'Viral', value: Math.round(metrics.viral_potential || 0), unit: '%' },
        { label: 'Community', value: Math.round(metrics.community_strength || 0), unit: '%' },
        { label: 'Activity', value: Math.round(metrics.volume_activity || 0), unit: '%' },
        { label: 'Quality', value: Math.round(metrics.engagement_quality || 0), unit: '%' },
        { label: 'AI Score', value: Math.round((data.confidence_score || 0.5) * 100), unit: '%' }
    ];

    let metricsHtml = '';
    metricsData.forEach(metric => {
        const percentage = metric.value;
        metricsHtml += `
            <div class="momentum-metric">
                <div
                    class="metric-visual"
                    style="background: conic-gradient(
                        #0040ff 0deg,
                        #00e5ff 72deg,
                        #00ff00 144deg,
                        #ffff00 216deg,
                        #ff7f00 288deg,
                        #ff0000 ${percentage * 3.6}deg,
                        rgba(255,255,255,0.1) ${percentage * 3.6}deg,
                        rgba(255,255,255,0.1) 360deg
                    );"
                ></div>
                <div class="metric-value-main">${metric.value}${metric.unit}</div>
                <div class="metric-label-main">${metric.label}</div>
            </div>
        `;
    });
    
    document.getElementById('momentumMetrics').innerHTML = metricsHtml;
}

function updateTweetsSection(data) {
    const tweets = data.actual_tweets || [];
    let tweetsHtml = '';
    
    if (tweets.length > 0) {
        tweets.forEach(tweet => {
            tweetsHtml += `
                <div class="tweet-item">
                    <div class="tweet-header">
                        <a href="${tweet.url || '#'}" target="_blank" class="tweet-author">@${tweet.author}</a>
                        <div class="tweet-timestamp">${tweet.timestamp}</div>
                    </div>
                    <div class="tweet-content">"${tweet.text}"</div>
                    <div class="tweet-engagement">${tweet.engagement} • 🔴 LIVE</div>
                </div>
            `;
        });
    } else {
        tweetsHtml = '<div class="tweet-item"><div class="tweet-content">Connect XAI API for real-time X/Twitter data feed with live social mentions and high-follower KOL activity.</div></div>';
    }
    
    document.getElementById('tweetsList').innerHTML = tweetsHtml;
    
    // Update contract accounts
    updateContractAccounts(data);
}

function updateContractAccounts(data) {
    const accounts = data.contract_accounts || [];
    let accountsHtml = '';
    
    if (accounts.length > 0) {
        accounts.forEach(account => {
            accountsHtml += `
                <div class="account-item" onclick="window.open('${account.url}', '_blank')">
                    <div class="account-username">@${account.username}</div>
                    <div class="account-followers">${account.followers}</div>
                    <div class="account-activity">${account.recent_activity}</div>
                </div>
            `;
        });
    } else {
        accountsHtml = `
            <div class="account-item" style="text-align: center; opacity: 0.7; cursor: default;">
                <div class="account-activity" style="font-style: italic;">
                    No accounts to recommend at this time
                </div>
            </div>
        `;
    }
    
    document.getElementById('contractAccountsList').innerHTML = accountsHtml;
}

function updateExpertAnalysis(data) {
    const analysis = data.expert_analysis || 'Connect XAI API for comprehensive expert analysis with real-time market insights and trading perspectives.';
    document.getElementById('expertAnalysis').innerHTML = analysis;
}

function updateAnalysisCards(data) {
    // Trading Signals
    const signals = data.trading_signals || [];
    let signalsHtml = '';
    if (signals.length > 0) {
        signals.forEach(signal => {
            const emoji = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '🟡', 'WATCH': '👀'}[signal.signal_type] || '⚪';
            signalsHtml += `
                <div class="content-heading">${emoji} ${signal.signal_type}</div>
                <p>${signal.reasoning}</p>
                <p><strong>Confidence:</strong> ${Math.round(signal.confidence * 100)}%</p>
            `;
        });
    } else {
        signalsHtml = '<p>Connect XAI API for real-time trading signals based on social momentum and market analysis.</p>';
    }
    document.getElementById('tradingSignals').innerHTML = signalsHtml;
    
    // Risk Assessment - Enhanced formatting with tables/icons
    let riskHtml = data.risk_assessment || '';
    if (riskHtml && riskHtml.includes('**')) {
        // Parse structured risk data and format with icons/tables
        riskHtml = formatRiskAssessment(riskHtml);
    } else {
        riskHtml = `
            <div class="content-heading">⚠️ Risk Assessment</div>
            <p>Connect XAI API for comprehensive risk analysis including market cap assessment, liquidity analysis, and volatility metrics.</p>
        `;
    }
    document.getElementById('riskAssessment').innerHTML = riskHtml;
    
    // Market Predictions - Enhanced formatting with tables/icons
    let predictionHtml = data.market_predictions || '';
    if (predictionHtml && predictionHtml.includes('**')) {
        // Parse structured prediction data and format with icons/tables
        predictionHtml = formatMarketPredictions(predictionHtml);
    } else {
        predictionHtml = `
            <div class="content-heading">🔮 Market Predictions</div>
            <p>Connect XAI API for AI-powered market predictions based on social momentum, technical analysis, and fundamental metrics.</p>
        `;
    }
    document.getElementById('marketPredictions').innerHTML = predictionHtml;
}

function formatRiskAssessment(riskData) {
    // Parse structured risk data and format with icons/tables
    let html = '<div class="risk-table">';
    
    const lines = riskData.split('\n');
    lines.forEach(line => {
        if (line.includes('**Risk Level:')) {
            const riskLevel = line.match(/Risk Level:\s*(\w+)/i);
            if (riskLevel) {
                const level = riskLevel[1].toUpperCase();
                const icon = level === 'HIGH' ? '🔴' : level === 'MEDIUM' ? '🟡' : '🟢';
                const className = level === 'HIGH' ? 'risk-high' : level === 'MEDIUM' ? 'risk-medium' : 'risk-low';
                html += `
                    <div class="risk-item">
                        <div class="risk-icon">${icon}</div>
                        <div class="risk-text">
                            <strong>Overall Risk</strong>
                            <div class="risk-level ${className}">${level}</div>
                        </div>
                    </div>
                `;
            }
        } else if (line.includes('**') && line.includes('Risk:')) {
            const type = line.match(/\*\*([^*]+Risk):\*\*/);
            const value = line.split('**')[2]?.trim();
            if (type && value) {
                const icon = type[1].includes('Liquidity') ? '💧' : type[1].includes('Volatility') ? '📊' : '💰';
                html += `
                    <div class="risk-item">
                        <div class="risk-icon">${icon}</div>
                        <div class="risk-text">
                            <strong>${type[1]}</strong>
                            <div>${value}</div>
                        </div>
                    </div>
                `;
            }
        }
    });
    
    html += '</div>';
    return html;
}

function formatMarketPredictions(predictionData) {
    // Parse structured prediction data and format with icons/tables
    let html = '<div class="prediction-list">';
    
    const lines = predictionData.split('\n');
    lines.forEach(line => {
        if (line.includes('**') && line.includes('Outlook:')) {
            const outlook = line.match(/Outlook:\s*(\w+)/i);
            if (outlook) {
                const trend = outlook[1].toUpperCase();
                const icon = trend === 'BULLISH' ? '🚀' : trend === 'BEARISH' ? '📉' : '➡️';
                html += `
                    <div class="prediction-item">
                        <div class="prediction-icon">${icon}</div>
                        <div class="prediction-text">
                            <strong>7-Day Outlook: ${trend}</strong>
                        </div>
                    </div>
                `;
            }
        } else if (line.includes('**') && (line.includes('Catalysts') || line.includes('Targets'))) {
            const type = line.match(/\*\*([^*]+):\*\*/);
            if (type) {
                const icon = type[1].includes('Catalysts') ? '⚡' : '🎯';
                const content = line.split('**')[2]?.trim() || 'Data pending...';
                html += `
                    <div class="prediction-item">
                        <div class="prediction-icon">${icon}</div>
                        <div class="prediction-text">
                            <strong>${type[1]}</strong>
                            <div>${content}</div>
                        </div>
                    </div>
                `;
            }
        }
    });
    
    html += '</div>';
    return html;
}

// Chat Functions
function enableChatWithContext() {
    const chatInput = document.getElementById('chatInput');
    const chatSend = document.getElementById('chatSend');
    
    chatInput.disabled = false;
    chatInput.placeholder = `Ask about ${currentAnalysis?.token_symbol || 'this token'}...`;
    chatSend.disabled = false;
}

async function sendChatMessage() {
    const chatInput = document.getElementById('chatInput');
    const message = chatInput.value.trim();
    
    if (!message || !currentTokenAddress) return;

    addChatMessage(message, 'user');
    chatInput.value = '';

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                token_address: currentTokenAddress,
                message: message,
                history: chatHistory
            })
        });

        const data = await response.json();
        
        if (data.response) {
            addChatMessage(data.response, 'assistant');
        } else {
            addChatMessage('Sorry, I encountered an error. Please try again.', 'assistant');
        }
        
    } catch (error) {
        console.error('Chat error:', error);
        addChatMessage('Connection error. Please try again.', 'assistant');
    }
}

function addChatMessage(message, sender) {
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${sender}`;
    messageDiv.textContent = message;
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    chatHistory.push({
        role: sender === 'user' ? 'user' : 'assistant',
        content: message
    });
    
    if (chatHistory.length > 10) {
        chatHistory = chatHistory.slice(-10);
    }
}

// Utility Functions
function showLoading() {
    document.getElementById('loadingSpinner').style.display = 'block';
    document.querySelector('.analyze-btn').disabled = true;
}

function hideLoading() {
    document.getElementById('loadingSpinner').style.display = 'none';
    document.querySelector('.analyze-btn').disabled = false;
}

function updateLoadingText(message) {
    document.getElementById('loadingText').textContent = message;
}

function showError(message) {
    const errorEl = document.getElementById('errorMessage');
    errorEl.textContent = message;
    errorEl.style.display = 'block';
}

function hideError() {
    document.getElementById('errorMessage').style.display = 'none';
}

function hideResults() {
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('chartContainer').style.display = 'none';
    document.getElementById('tokenProfile').style.display = 'none';
}
