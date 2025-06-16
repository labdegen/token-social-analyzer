  let currentCharts = {};
        let currentMode = 'express';

        function setMode(mode) {
            currentMode = mode;
            document.querySelectorAll('.mode-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            document.querySelector(`[data-mode="${mode}"]`).classList.add('active');
        }

        function formatAddress(address) {
            if (!address) return '';
            return address.slice(0, 8) + '...' + address.slice(-6);
        }

        function formatNumber(num) {
            // FIXED: Handle null, undefined, and ensure proper number formatting
            if (num === null || num === undefined || isNaN(num)) return '0';
            
            const numValue = parseFloat(num);
            if (numValue >= 1e9) return (numValue / 1e9).toFixed(2) + 'B';
            if (numValue >= 1e6) return (numValue / 1e6).toFixed(2) + 'M';
            if (numValue >= 1e3) return (numValue / 1e3).toFixed(1) + 'K';
            return numValue.toFixed(2);
        }

        function formatCurrency(num) {
            // FIXED: Ensure proper currency formatting
            if (num === null || num === undefined || isNaN(num)) return '$0';
            return '$' + formatNumber(num);
        }

        function formatTokenAge(ageDays) {
            // FIXED: Remove decimals from token age display
            if (ageDays === null || ageDays === undefined || isNaN(ageDays)) return '0';
            return Math.floor(parseFloat(ageDays)).toString();
        }

        async function analyzeToken() {
            const tokenAddress = document.getElementById('tokenAddress').value.trim();
            if (!tokenAddress) {
                showError('Please enter a token address');
                return;
            }

            // Reset UI
            document.getElementById('loading').classList.add('active');
            document.getElementById('error').classList.remove('active');
            document.getElementById('results').classList.remove('active');
            document.querySelector('.analyze-btn').disabled = true;

            try {
                const response = await fetch('/rugcheck', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        address: tokenAddress,
                        mode: currentMode
                    })
                });

                const data = await response.json();

                if (!data.success) {
                    throw new Error(data.error || 'Revolutionary analysis failed');
                }

                displayResults(data);

            } catch (error) {
                showError(error.message);
            } finally {
                document.getElementById('loading').classList.remove('active');
                document.querySelector('.analyze-btn').disabled = false;
            }
        }

        function showError(message) {
            const errorEl = document.getElementById('error');
            errorEl.innerHTML = `<strong>Revolutionary Analysis Error:</strong> ${message}`;
            errorEl.classList.add('active');
        }

        function displayResults(data) {
            console.log('🧠 Revolutionary Galaxy Brain Results:', data);
            
            // Show results section
            document.getElementById('results').classList.add('active');

            // Display token info
            displayTokenInfo(data.token_info);

            // 🧠 REVOLUTIONARY: Display Grok Analysis FIRST (most important)
            displayRevolutionaryGrokInsights(data.grok_analysis);

            // Display Galaxy Brain summary
            displayGalaxyBrainSummary(data);

            // Display risk vectors
            displayRiskVectors(data.risk_vectors || []);

            // Display advanced metrics
            displayAdvancedMetrics(data);

            // Display charts
            displayAdvancedCharts(data);

            // Display detailed analysis tabs
            displayDetailedAnalysis(data);
        }

        function displayRevolutionaryGrokInsights(grokAnalysis) {
            const grokContainer = document.getElementById('revolutionaryGrokAnalysis');
            
            if (!grokAnalysis || !grokAnalysis.available) {
                grokContainer.innerHTML = `
                    <div class="grok-header">
                        <div class="grok-brain-icon">🧠</div>
                        <div class="grok-title">Revolutionary Grok Intelligence</div>
                        <div class="grok-verdict caution_advised">API Key Required</div>
                    </div>
                    <div class="grok-content">
                        <div class="grok-section">
                            <h4>🔥 Connect Grok for Revolutionary Analysis</h4>
                            <p>Add your XAI API key to unlock live community intelligence, real-time safety analysis, and revolutionary meme coin pattern detection.</p>
                        </div>
                        <div class="grok-section">
                            <h4>🧠 Revolutionary Features Available</h4>
                            <p>• Live X/Twitter community sentiment analysis<br>
                               • Real-time scam and rug pull detection<br>
                               • Meme coin safety pattern recognition<br>
                               • Diamond hands vs paper hands analysis<br>
                               • Whale legitimacy verification</p>
                        </div>
                    </div>
                `;
                grokContainer.style.display = 'block';
                return;
            }

            const grokData = grokAnalysis.parsed_analysis || {};
            const verdict = grokData.verdict || 'CAUTION_ADVISED';
            const verdictClass = verdict.toLowerCase();

            // Get specific community intelligence
            const positiveSentiment = grokData.positive_community_sentiment || [];
            const communityRisks = grokData.possible_community_risks || [];
            const usernameMentions = grokData.username_mentions || 0;
            const quoteMentions = grokData.quote_mentions || 0;

            // 🧠 REVOLUTIONARY Grok display with SPECIFIC community intelligence
            grokContainer.innerHTML = `
                <div class="grok-header">
                    <div class="grok-brain-icon">🧠</div>
                    <div class="grok-title">Revolutionary Grok Intelligence</div>
                    <div class="grok-verdict ${verdictClass}">${verdict.replace(/_/g, ' ')}</div>
                </div>
                
                <div class="grok-content">
                    ${positiveSentiment.length > 0 ? `
                    <div class="grok-section">
                        <h4>✅ Positive Community Sentiment</h4>
                        ${positiveSentiment.map(sentiment => `
                            <div style="background: rgba(81, 207, 102, 0.15); border-left: 3px solid #51cf66; padding: 12px; margin: 8px 0; border-radius: 5px;">
                                ${sentiment}
                            </div>
                        `).join('')}
                    </div>
                    ` : `
                    <div class="grok-section">
                        <h4>✅ Positive Community Sentiment</h4>
                        <p style="color: #888;">No specific positive community mentions found for this token.</p>
                    </div>
                    `}
                    
                    ${communityRisks.length > 0 ? `
                    <div class="grok-section">
                        <h4>⚠️ Possible Community Risks</h4>
                        ${communityRisks.map(risk => `
                            <div style="background: rgba(255, 107, 107, 0.15); border-left: 3px solid #ff6b6b; padding: 12px; margin: 8px 0; border-radius: 5px;">
                                ${risk}
                            </div>
                        `).join('')}
                    </div>
                    ` : `
                    <div class="grok-section">
                        <h4>⚠️ Possible Community Risks</h4>
                        <p style="color: #888;">No specific community risk warnings found for this token.</p>
                    </div>
                    `}
                    
                    <div class="grok-section">
                        <h4>🐋 Whale Analysis</h4>
                        <p>${grokData.whale_analysis || 'No specific whale behavior analysis available'}</p>
                    </div>
                    
                    <div class="grok-section">
                        <h4>🧠 Revolutionary Insight</h4>
                        <p>${grokData.revolutionary_insight || 'Analysis incomplete'}</p>
                    </div>
                </div>
                
                <div style="margin-top: 20px; padding: 15px; background: rgba(108, 92, 231, 0.1); border-radius: 10px; border: 1px solid #6c5ce7;">
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin-bottom: 10px;">
                        <div style="text-align: center;">
                            <div style="font-size: 1.5rem; font-weight: bold; color: #6c5ce7;">${usernameMentions}</div>
                            <div style="font-size: 0.8rem; color: #aaa;">@Username Mentions</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 1.5rem; font-weight: bold; color: #6c5ce7;">${quoteMentions}</div>
                            <div style="font-size: 0.8rem; color: #aaa;">Specific Quotes</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 1.5rem; font-weight: bold; color: #6c5ce7;">${Math.round((grokData.confidence || 0.5) * 100)}%</div>
                            <div style="font-size: 0.8rem; color: #aaa;">Analysis Confidence</div>
                        </div>
                    </div>
                    <p style="color: #a29bfe; font-size: 0.9rem; margin: 0; text-align: center;">
                        <strong>🧠 Revolutionary Intelligence:</strong> ${grokData.analysis_type || 'Standard'} | 
                        <strong>Data Source:</strong> Live X/Twitter Search
                    </p>
                </div>
            `;
            
            grokContainer.style.display = 'block';
        }

        function displayTokenInfo(tokenInfo) {
            if (!tokenInfo) return;

            const tokenInfoEl = document.getElementById('tokenInfo');
            const logoEl = document.getElementById('tokenLogo');
            const nameEl = document.getElementById('tokenName');
            const symbolEl = document.getElementById('tokenSymbol');
            const addressEl = document.getElementById('tokenAddressDisplay');

            if (tokenInfo.logo) {
                logoEl.innerHTML = `<img src="${tokenInfo.logo}" alt="Token Logo" style="width: 100%; height: 100%; border-radius: 50%;">`;
            } else {
                logoEl.innerHTML = '🪙';
            }

            nameEl.textContent = tokenInfo.name || 'Unknown Token';
            symbolEl.textContent = `${tokenInfo.symbol || 'UNKNOWN'}`;
            addressEl.textContent = formatAddress(tokenInfo.address || tokenInfo.mint);

            tokenInfoEl.style.display = 'flex';
        }

        function displayGalaxyBrainSummary(data) {
            const score = data.galaxy_brain_score || 0;
            const severity = data.severity_level || 'UNKNOWN';
            const confidence = data.confidence || 0;

            // Update Galaxy Brain score
            const scoreEl = document.getElementById('galaxyScore');
            const summaryEl = document.getElementById('galaxySummary');
            scoreEl.textContent = score;
            
            // Set severity styling
            const severityClass = severity.toLowerCase().replace(/_/g, '_');
            scoreEl.className = `galaxy-score ${severityClass}`;
            summaryEl.className = `galaxy-brain-summary ${scoreEl.className}`;

            // Update severity level
            document.getElementById('severityLevel').textContent = severity.replace(/_/g, ' ');

            // Update confidence
            const confidenceFill = document.getElementById('confidenceFill');
            const confidenceText = document.getElementById('confidenceText');
            confidenceFill.style.width = `${confidence * 100}%`;
            confidenceText.textContent = `Confidence: ${Math.round(confidence * 100)}%`;

            // Update AI analysis
            const aiAnalysisEl = document.getElementById('aiAnalysis');
            if (data.ai_analysis) {
                aiAnalysisEl.innerHTML = `
                    <h4>🧠 Revolutionary AI Analysis</h4>
                    <div>${data.ai_analysis}</div>
                `;
            }
        }

        function displayRiskVectors(riskVectors) {
            const grid = document.getElementById('riskVectorsGrid');
            grid.innerHTML = '';

            if (riskVectors.length === 0) {
                grid.innerHTML = `
                    <div class="risk-vector-card" style="grid-column: 1 / -1;">
                        <div class="vector-title">✅ No Major Risk Vectors Detected</div>
                        <div class="vector-impact">Revolutionary Galaxy Brain analysis found no critical risk patterns in this token.</div>
                    </div>
                `;
                return;
            }

            riskVectors.forEach(vector => {
                const card = document.createElement('div');
                card.className = `risk-vector-card ${vector.severity.toLowerCase()}`;
                card.innerHTML = `
                    <div class="vector-category">${vector.category}</div>
                    <div class="vector-title">${getSeverityIcon(vector.severity)} ${vector.risk_type}</div>
                    <div class="vector-impact">${vector.impact}</div>
                    <div class="vector-mitigation">
                        <strong>Mitigation:</strong> ${vector.mitigation}
                    </div>
                `;
                grid.appendChild(card);
            });
        }

        function getSeverityIcon(severity) {
            const icons = {
                'CRITICAL': '🚨',
                'HIGH': '⚠️',
                'MEDIUM': '🟡',
                'LOW': '🟢'
            };
            return icons[severity] || '⚪';
        }

        function displayAdvancedMetrics(data) {
            const grid = document.getElementById('advancedMetrics');
            grid.innerHTML = '';

            const tokenInfo = data.token_info || {};
            const holderAnalysis = data.holder_analysis || {};
            const liquidityAnalysis = data.liquidity_analysis || {};
            const transactionAnalysis = data.transaction_analysis || {};
            const bundleDetection = data.bundle_detection || {};
            const suspiciousActivity = data.suspicious_activity || {};

            const metrics = [
                {
                    label: 'Market Cap',
                    // FIXED: Properly handle market cap from correct field
                    value: formatCurrency(tokenInfo.market_cap || 0),
                    subtitle: 'Total Value'
                },
                {
                    label: 'Liquidity',
                    value: formatCurrency(tokenInfo.liquidity || 0),
                    subtitle: `${(liquidityAnalysis.liquidity_ratio || 0).toFixed(1)}% of MC`
                },
                {
                    label: 'Token Age',
                    // FIXED: Remove decimals from token age display
                    value: formatTokenAge(tokenInfo.age_days || 0),
                    subtitle: 'Days Old'
                },
                {
                    label: 'Top Holder',
                    value: `${(holderAnalysis.top_1_percent || 0).toFixed(1)}%`,
                    subtitle: 'Concentration Risk'
                },
                {
                    label: 'Wash Trading',
                    value: `${(suspiciousActivity.wash_trading_score || 0).toFixed(0)}%`,
                    subtitle: 'Manipulation Risk'
                },
                {
                    label: 'Bundle Risk',
                    value: `${(bundleDetection.bundled_percentage || 0).toFixed(1)}%`,
                    subtitle: `${bundleDetection.clusters_found || 0} Clusters`
                },
                {
                    label: 'Transaction Health',
                    value: `${(suspiciousActivity.transaction_health_score || 50).toFixed(0)}%`,
                    subtitle: 'Quality Score'
                },
                {
                    label: 'Authorities',
                    value: tokenInfo.is_mutable ? '🔴 Active' : '✅ Renounced',
                    subtitle: 'Mint/Freeze Status'
                }
            ];

            metrics.forEach(metric => {
                const card = document.createElement('div');
                card.className = 'metric-card';
                card.innerHTML = `
                    <div class="metric-label">${metric.label}</div>
                    <div class="metric-value">${metric.value}</div>
                    <div class="metric-subtitle">${metric.subtitle}</div>
                `;
                grid.appendChild(card);
            });
        }

        function displayAdvancedCharts(data) {
            // Destroy existing charts
            Object.values(currentCharts).forEach(chart => {
                if (chart && typeof chart.destroy === 'function') {
                    chart.destroy();
                }
            });
            currentCharts = {};

            // Wait a bit for DOM to settle
            setTimeout(() => {
                try {
                    // Revolutionary Risk Breakdown Chart
                    displayRiskBreakdownChart(data);
                    
                    // Holder Distribution Chart  
                    displayHolderChart(data.holder_analysis || {});
                    
                    // Transaction Health Chart
                    displayTransactionChart(data.transaction_analysis || {}, data.suspicious_activity || {});
                    
                    // Bundle Detection Chart
                    displayBundleChart(data.bundle_detection || {});
                } catch (error) {
                    console.error('Chart display error:', error);
                }
            }, 100);
        }

        function displayRiskBreakdownChart(data) {
            const canvas = document.getElementById('riskBreakdownChart');
            if (!canvas) return;
            
            const ctx = canvas.getContext('2d');
            
            const riskVectors = data.risk_vectors || [];
            const labels = riskVectors.map(v => v.risk_type);
            const scores = riskVectors.map(v => {
                const severityScores = { 'CRITICAL': 90, 'HIGH': 70, 'MEDIUM': 50, 'LOW': 30 };
                return severityScores[v.severity] || 25;
            });
            const colors = riskVectors.map(v => {
                const severityColors = { 
                    'CRITICAL': '#ff3838', 
                    'HIGH': '#ff6b6b', 
                    'MEDIUM': '#ffd93d', 
                    'LOW': '#51cf66' 
                };
                return severityColors[v.severity] || '#888';
            });

            if (labels.length === 0) {
                labels.push('No Major Risks');
                scores.push(10);
                colors.push('#51cf66');
            }

            currentCharts.riskBreakdown = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: labels,
                    datasets: [{
                        data: scores,
                        backgroundColor: colors,
                        borderWidth: 2,
                        borderColor: '#1a1a1a'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: {
                                color: '#fff',
                                padding: 15,
                                usePointStyle: true,
                                font: { size: 12 }
                            }
                        }
                    }
                }
            });
        }

        function displayHolderChart(holderData) {
            const canvas = document.getElementById('holderChart');
            if (!canvas) return;
            
            const ctx = canvas.getContext('2d');
            
            const top1 = holderData.top_1_percent || 0;
            const top5 = (holderData.top_5_percent || 0) - top1;
            const top10 = (holderData.top_10_percent || 0) - (holderData.top_5_percent || 0);
            const others = 100 - (holderData.top_10_percent || 0);

            currentCharts.holder = new Chart(ctx, {
                type: 'pie',
                data: {
                    labels: ['Top 1 Holder', 'Top 2-5 Holders', 'Top 6-10 Holders', 'Others'],
                    datasets: [{
                        data: [top1, top5, top10, others],
                        backgroundColor: ['#ff3838', '#ff6b6b', '#ffd93d', '#51cf66'],
                        borderWidth: 2,
                        borderColor: '#1a1a1a'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: {
                                color: '#fff',
                                padding: 15,
                                font: { size: 12 }
                            }
                        }
                    }
                }
            });
        }

        function displayTransactionChart(transactionData, suspiciousData) {
            const canvas = document.getElementById('transactionChart');
            if (!canvas) return;
            
            const ctx = canvas.getContext('2d');
            
            const healthScore = suspiciousData.transaction_health_score || 50;
            const washScore = suspiciousData.wash_trading_score || 0;
            const organicScore = 100 - washScore;

            currentCharts.transaction = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: ['Health', 'Organic', 'Wash Risk'],
                    datasets: [{
                        label: 'Score %',
                        data: [healthScore, organicScore, washScore],
                        backgroundColor: [
                            healthScore > 70 ? '#51cf66' : healthScore > 40 ? '#ffd93d' : '#ff6b6b',
                            organicScore > 70 ? '#51cf66' : organicScore > 40 ? '#ffd93d' : '#ff6b6b',
                            washScore < 30 ? '#51cf66' : washScore < 60 ? '#ffd93d' : '#ff6b6b'
                        ],
                        borderWidth: 2,
                        borderColor: '#1a1a1a'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100,
                            ticks: { color: '#fff', font: { size: 11 } },
                            grid: { color: '#333' }
                        },
                        x: {
                            ticks: { color: '#fff', font: { size: 11 } },
                            grid: { color: '#333' }
                        }
                    },
                    plugins: {
                        legend: { display: false }
                    }
                }
            });
        }

        function displayBundleChart(bundleData) {
            const canvas = document.getElementById('bundleChart');
            if (!canvas) return;
            
            const ctx = canvas.getContext('2d');
            
            const clustersFound = bundleData.clusters_found || 0;
            const highRiskClusters = bundleData.high_risk_clusters || 0;
            const lowRiskClusters = clustersFound - highRiskClusters;
            const bundledPercentage = bundleData.bundled_percentage || 0;
            const organicPercentage = 100 - bundledPercentage;

            currentCharts.bundle = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: ['Organic Holdings', 'Bundled Holdings', 'High Risk Clusters', 'Low Risk Clusters'],
                    datasets: [{
                        data: [organicPercentage, bundledPercentage, highRiskClusters * 5, lowRiskClusters * 3],
                        backgroundColor: ['#51cf66', '#ffd93d', '#ff3838', '#ff6b6b'],
                        borderWidth: 2,
                        borderColor: '#1a1a1a'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: {
                                color: '#fff',
                                padding: 15,
                                font: { size: 11 }
                            }
                        }
                    }
                }
            });
        }

        function displayDetailedAnalysis(data) {
            // Holders tab
            displayHoldersTab(data.holder_analysis || {});
            
            // Transactions tab
            displayTransactionsTab(data.transaction_analysis || {}, data.suspicious_activity || {});
            
            // Bundles tab
            displayBundlesTab(data.bundle_detection || {});
            
            // Liquidity tab
            displayLiquidityTab(data.liquidity_analysis || {});
        }

        function displayHoldersTab(holderData) {
            const holdersList = document.getElementById('holdersList');
            holdersList.innerHTML = '';

            const holders = holderData.top_holders || [];
            
            if (holders.length === 0) {
                holdersList.innerHTML = '<p style="color: #888; text-align: center;">No holder data available</p>';
                return;
            }

            holders.forEach(holder => {
                const row = document.createElement('div');
                row.className = 'holder-row';
                
                let riskColor = '#51cf66';
                if (holder.percentage > 15) riskColor = '#ff3838';
                else if (holder.percentage > 8) riskColor = '#ff6b6b';
                else if (holder.percentage > 3) riskColor = '#ffd93d';

                row.innerHTML = `
                    <span class="holder-address">
                        <strong>#${holder.rank}</strong> ${formatAddress(holder.address)}
                    </span>
                    <span class="holder-percent" style="color: ${riskColor}">
                        ${holder.percentage.toFixed(2)}%
                    </span>
                `;
                holdersList.appendChild(row);
            });
        }

        function displayTransactionsTab(transactionData, suspiciousData) {
            const container = document.getElementById('transactionAnalysis');
            
            container.innerHTML = `
                <div style="display: grid; gap: 20px;">
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                        <div class="metric-card">
                            <div class="metric-label">Transaction Health Score</div>
                            <div class="metric-value">${(suspiciousData.transaction_health_score || 50).toFixed(0)}%</div>
                            <div class="metric-subtitle">Overall transaction quality</div>
                        </div>
                        
                        <div class="metric-card">
                            <div class="metric-label">Wash Trading Risk</div>
                            <div class="metric-value">${(suspiciousData.wash_trading_score || 0).toFixed(1)}%</div>
                            <div class="metric-subtitle">Artificial volume inflation</div>
                        </div>
                        
                        <div class="metric-card">
                            <div class="metric-label">Insider Activity</div>
                            <div class="metric-value">${(suspiciousData.insider_activity_score || 0).toFixed(1)}%</div>
                            <div class="metric-subtitle">Coordinated trading patterns</div>
                        </div>
                    </div>
                    
                    ${suspiciousData.farming_indicators && suspiciousData.farming_indicators.length > 0 ? `
                    <div style="background: rgba(255, 107, 107, 0.1); padding: 20px; border-radius: 10px; border: 1px solid #ff6b6b;">
                        <h4 style="color: #ff6b6b; margin-bottom: 15px;">🚨 Farming Indicators Detected</h4>
                        ${suspiciousData.farming_indicators.map(indicator => `<p>• ${indicator}</p>`).join('')}
                    </div>
                    ` : ''}
                    
                    ${suspiciousData.suspicious_patterns && suspiciousData.suspicious_patterns.length > 0 ? `
                    <div style="background: rgba(255, 217, 61, 0.1); padding: 20px; border-radius: 10px; border: 1px solid #ffd93d;">
                        <h4 style="color: #ffd93d; margin-bottom: 15px;">⚠️ Suspicious Patterns</h4>
                        ${suspiciousData.suspicious_patterns.map(pattern => `<p>• ${pattern}</p>`).join('')}
                    </div>
                    ` : ''}
                </div>
            `;
        }

        function displayBundlesTab(bundleData) {
            const container = document.getElementById('bundleDetection');
            
            const clusters = bundleData.clusters || [];
            
            container.innerHTML = `
                <div style="display: grid; gap: 20px;">
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                        <div class="metric-card">
                            <div class="metric-label">Clusters Found</div>
                            <div class="metric-value">${bundleData.clusters_found || 0}</div>
                            <div class="metric-subtitle">Bundle groups detected</div>
                        </div>
                        
                        <div class="metric-card">
                            <div class="metric-label">High Risk Clusters</div>
                            <div class="metric-value">${bundleData.high_risk_clusters || 0}</div>
                            <div class="metric-subtitle">Dangerous patterns</div>
                        </div>
                        
                        <div class="metric-card">
                            <div class="metric-label">Bundled Percentage</div>
                            <div class="metric-value">${(bundleData.bundled_percentage || 0).toFixed(1)}%</div>
                            <div class="metric-subtitle">Of total supply</div>
                        </div>
                        
                        <div class="metric-card">
                            <div class="metric-label">Risk Level</div>
                            <div class="metric-value">${bundleData.risk_level || 'UNKNOWN'}</div>
                            <div class="metric-subtitle">Bundle threat assessment</div>
                        </div>
                    </div>
                    
                    ${clusters.length > 0 ? `
                    <div>
                        <h4 style="margin-bottom: 15px;">🕷️ Detected Clusters</h4>
                        ${clusters.map(cluster => `
                        <div class="cluster-item">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                <strong>${cluster.cluster_id}</strong>
                                <span style="color: ${cluster.risk_score > 70 ? '#ff3838' : cluster.risk_score > 40 ? '#ffd93d' : '#51cf66'}">
                                    Risk: ${cluster.risk_score}%
                                </span>
                            </div>
                            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 10px; font-size: 0.9rem; color: #ccc;">
                                <div>Wallets: ${cluster.wallet_count}</div>
                                <div>Holdings: ${cluster.total_percentage.toFixed(1)}%</div>
                                <div>Pattern: ${cluster.creation_pattern}</div>
                            </div>
                        </div>
                        `).join('')}
                    </div>
                    ` : '<p style="color: #888; text-align: center;">No clusters detected - this is good!</p>'}
                </div>
            `;
        }

        function displayLiquidityTab(liquidityData) {
            const container = document.getElementById('liquidityAnalysis');
            
            container.innerHTML = `
                <div style="display: grid; gap: 20px;">
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                        <div class="metric-card">
                            <div class="metric-label">Liquidity USD</div>
                            <div class="metric-value">${formatCurrency(liquidityData.liquidity_usd || 0)}</div>
                            <div class="metric-subtitle">Available liquidity</div>
                        </div>
                        
                        <div class="metric-card">
                            <div class="metric-label">Liquidity Ratio</div>
                            <div class="metric-value">${(liquidityData.liquidity_ratio || 0).toFixed(1)}%</div>
                            <div class="metric-subtitle">Of market cap</div>
                        </div>
                        
                        <div class="metric-card">
                            <div class="metric-label">Volume/Liquidity</div>
                            <div class="metric-value">${(liquidityData.volume_to_liquidity || 0).toFixed(1)}x</div>
                            <div class="metric-subtitle">Trading intensity</div>
                        </div>
                        
                        <div class="metric-card">
                            <div class="metric-label">Risk Level</div>
                            <div class="metric-value">${liquidityData.liquidity_risk || 'UNKNOWN'}</div>
                            <div class="metric-subtitle">Liquidity assessment</div>
                        </div>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;">
                        <div style="background: rgba(255, 217, 61, 0.1); padding: 20px; border-radius: 10px; border: 1px solid #ffd93d;">
                            <h4 style="color: #ffd93d; margin-bottom: 15px;">📊 Slippage Estimates</h4>
                            <div style="display: grid; gap: 10px;">
                                <div style="display: flex; justify-content: space-between;">
                                    <span>DEX:</span>
                                    <span style="font-weight: bold;">${liquidityData.dex || 'Unknown'}</span>
                                </div>
                                <div style="display: flex; justify-content: space-between;">
                                    <span>$1K Trade:</span>
                                    <span style="font-weight: bold;">${(liquidityData.slippage_1k || 0).toFixed(1)}%</span>
                                </div>
                                <div style="display: flex; justify-content: space-between;">
                                    <span>$10K Trade:</span>
                                    <span style="font-weight: bold;">${(liquidityData.slippage_10k || 0).toFixed(1)}%</span>
                                </div>
                            </div>
                        </div>
                        
                        <div style="background: rgba(108, 92, 231, 0.1); padding: 20px; border-radius: 10px; border: 1px solid #6c5ce7;">
                            <h4 style="color: #6c5ce7; margin-bottom: 15px;">🔒 Liquidity Lock Status</h4>
                            <div style="display: grid; gap: 10px;">
                                <div style="display: flex; justify-content: space-between;">
                                    <span>Locked:</span>
                                    <span style="font-weight: bold; color: ${liquidityData.is_locked ? '#51cf66' : '#ff6b6b'}">
                                        ${liquidityData.is_locked ? 'Yes' : 'No'}
                                    </span>
                                </div>
                                <div style="display: flex; justify-content: space-between;">
                                    <span>Duration:</span>
                                    <span style="font-weight: bold;">${liquidityData.lock_duration || 'N/A'}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }

        function showTab(tabName) {
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
            document.querySelectorAll('.tab-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            
            // Show selected tab
            document.getElementById(`${tabName}-tab`).classList.add('active');

            event.target.classList.add('active');
        }

        // Auto-analyze on load if address is present
        window.onload = function() {
            const urlParams = new URLSearchParams(window.location.search);
            const address = urlParams.get('address');
            if (address) {
                document.getElementById('tokenAddress').value = address;
                analyzeToken();
            }
        };

        // Enter key support
        document.getElementById('tokenAddress').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                analyzeToken();
            }
        });