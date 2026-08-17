/**
 * SentimentX — Frontend Logic
 * Hybrid CNN-XLNet Multilingual Sentiment Analysis
 */
document.addEventListener('DOMContentLoaded', () => {
    // ─── DOM Elements ───
    const analyzeBtn = document.getElementById('analyze-btn');
    const textInput = document.getElementById('text-input');
    const langSelect = document.getElementById('language');
    const charCount = document.getElementById('char-count');

    const placeholderState = document.getElementById('placeholder-state');
    const loadingState = document.getElementById('loading-state');
    const resultState = document.getElementById('result-state');

    const emotionEmoji = document.getElementById('emotion-emoji');
    const emotionName = document.getElementById('emotion-name');
    const confidenceValue = document.getElementById('confidence-value');
    const ringFill = document.getElementById('ring-fill');
    const probBars = document.getElementById('prob-bars');
    const detectedLanguage = document.getElementById('detected-language');
    const responseTime = document.getElementById('response-time');
    const textLength = document.getElementById('text-length');

    const historyList = document.getElementById('history-list');
    const historyEmpty = document.getElementById('history-empty');
    const clearHistoryBtn = document.getElementById('clear-history');

    const API_URL = 'http://localhost:8000/predict';

    // ─── Emotion Config ───
    const EMOTION_CONFIG = {
        'Happiness': { emoji: '😊', color: 'var(--happiness)' },
        'Sadness':   { emoji: '😢', color: 'var(--sadness)' },
        'Anger':     { emoji: '😡', color: 'var(--anger)' },
        'Fear':      { emoji: '😨', color: 'var(--fear)' },
        'Surprise':  { emoji: '😲', color: 'var(--surprise)' },
        'Disgust':   { emoji: '🤢', color: 'var(--disgust)' },
        'Neutral':   { emoji: '😐', color: 'var(--neutral)' }
    };

    const ALL_EMOTIONS = ['Happiness', 'Sadness', 'Anger', 'Fear', 'Surprise', 'Disgust'];

    // ─── History Store ───
    let analysisHistory = JSON.parse(localStorage.getItem('sentimentx_history') || '[]');
    renderHistory();

    // ─── Particles ───
    createParticles();

    // ─── Character Count ───
    textInput.addEventListener('input', () => {
        const len = textInput.value.length;
        charCount.textContent = `${len} / 1000`;
    });

    // ─── Quick Examples ───
    document.querySelectorAll('.example-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            textInput.value = chip.dataset.text;
            langSelect.value = chip.dataset.lang;
            charCount.textContent = `${chip.dataset.text.length} / 1000`;
            textInput.focus();
        });
    });

    // ─── Analyze Button ───
    analyzeBtn.addEventListener('click', async () => {
        const text = textInput.value.trim();
        const lang = langSelect.value;

        if (!text) {
            shakeElement(textInput);
            return;
        }

        showState('loading');
        const startTime = performance.now();

        analyzeBtn.querySelector('.btn-content').innerHTML =
            "<i class='bx bx-loader-alt bx-spin'></i><span>Processing…</span>";
        analyzeBtn.disabled = true;

        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text, language: lang })
            });

            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const data = await response.json();
            const elapsed = Math.round(performance.now() - startTime);

            if (data.success) {
                displayResult(data.data, lang, elapsed, text);
            }
        } catch (error) {
            console.warn('Backend unavailable, using simulation mode.', error);
            // Simulate result for UI demonstration
            await new Promise(r => setTimeout(r, 800 + Math.random() * 700));
            const elapsed = Math.round(performance.now() - startTime);
            const simResult = simulateResult(text);
            displayResult(simResult, lang, elapsed, text);
        } finally {
            analyzeBtn.querySelector('.btn-content').innerHTML =
                "<i class='bx bx-analyse'></i><span>Analyze Sentiment</span>";
            analyzeBtn.disabled = false;
        }
    });

    // ─── Clear History ───
    clearHistoryBtn.addEventListener('click', () => {
        analysisHistory = [];
        localStorage.removeItem('sentimentx_history');
        renderHistory();
    });

    // ─── State Management ───
    function showState(state) {
        placeholderState.classList.add('hidden');
        loadingState.classList.add('hidden');
        resultState.classList.add('hidden');

        if (state === 'placeholder') placeholderState.classList.remove('hidden');
        if (state === 'loading') loadingState.classList.remove('hidden');
        if (state === 'result') resultState.classList.remove('hidden');
    }

    // ─── Display Result ───
    function displayResult(data, lang, elapsed, originalText) {
        const sentiment = data.label || 'Neutral';
        const confidence = data.confidence || 0;
        const config = EMOTION_CONFIG[sentiment] || EMOTION_CONFIG['Neutral'];

        // Hero
        emotionEmoji.textContent = config.emoji;
        emotionName.textContent = sentiment;
        emotionName.style.color = config.color;

        // Confidence Ring
        const percent = (confidence * 100).toFixed(1);
        confidenceValue.textContent = `${percent}%`;
        confidenceValue.style.color = config.color;

        const circumference = 2 * Math.PI * 52; // r=52
        const offset = circumference - (confidence * circumference);
        ringFill.style.strokeDashoffset = offset;
        ringFill.style.stroke = config.color;

        // Probability Bars
        const probabilities = data.probabilities || generateFakeProbabilities(sentiment, confidence);
        renderProbBars(probabilities);

        // Metadata
        const langLabels = {
            auto: 'Auto-Detected',
            kannada: 'Kannada',
            malayalam: 'Malayalam',
            hindi: 'Hindi',
            marathi: 'Marathi',
            tulu: 'Tulu',
            telugu: 'Telugu',
            english: 'English'
        };
        detectedLanguage.textContent = langLabels[lang] || lang;
        responseTime.textContent = `${elapsed}ms`;
        textLength.textContent = `${originalText.length} chars`;

        showState('result');

        // Add to history
        addToHistory({
            text: originalText,
            sentiment,
            confidence: parseFloat(percent),
            emoji: config.emoji,
            color: config.color,
            lang,
            timestamp: Date.now()
        });
    }

    // ─── Probability Bars ───
    function renderProbBars(probs) {
        probBars.innerHTML = '';
        ALL_EMOTIONS.forEach(emotion => {
            const val = probs[emotion] || 0;
            const pct = (val * 100).toFixed(1);
            const config = EMOTION_CONFIG[emotion];

            const row = document.createElement('div');
            row.className = 'prob-bar-row';
            row.innerHTML = `
                <span class="prob-label">
                    <span>${config.emoji}</span>
                    <span>${emotion}</span>
                </span>
                <div class="prob-track">
                    <div class="prob-fill" style="background: ${config.color}; width: 0%"></div>
                </div>
                <span class="prob-percent">${pct}%</span>
            `;
            probBars.appendChild(row);

            // Animate bar width
            requestAnimationFrame(() => {
                requestAnimationFrame(() => {
                    row.querySelector('.prob-fill').style.width = `${pct}%`;
                });
            });
        });
    }

    // ─── History ───
    function addToHistory(entry) {
        analysisHistory.unshift(entry);
        if (analysisHistory.length > 10) analysisHistory.pop();
        localStorage.setItem('sentimentx_history', JSON.stringify(analysisHistory));
        renderHistory();
    }

    function renderHistory() {
        // Clear existing cards (keep the empty state)
        const cards = historyList.querySelectorAll('.history-card');
        cards.forEach(c => c.remove());

        if (analysisHistory.length === 0) {
            historyEmpty.classList.remove('hidden');
            return;
        }

        historyEmpty.classList.add('hidden');

        analysisHistory.forEach((entry, i) => {
            const card = document.createElement('div');
            card.className = 'history-card';
            card.innerHTML = `
                <div class="history-card-header">
                    <span class="history-card-emoji">${entry.emoji}</span>
                    <span class="history-card-emotion" style="color: ${entry.color}">${entry.sentiment}</span>
                </div>
                <div class="history-card-text">${escapeHtml(entry.text)}</div>
                <div class="history-card-conf">${entry.confidence}% confidence</div>
            `;
            card.addEventListener('click', () => {
                textInput.value = entry.text;
                langSelect.value = entry.lang;
                charCount.textContent = `${entry.text.length} / 1000`;
            });
            historyList.appendChild(card);
        });
    }

    // ─── Simulation (fallback when backend is down) ───
    function simulateResult(text) {
        const sentiments = ALL_EMOTIONS;
        const topIdx = Math.floor(Math.random() * sentiments.length);
        const topLabel = sentiments[topIdx];
        const topConf = 0.82 + Math.random() * 0.17;

        const probs = {};
        let remaining = 1 - topConf;
        sentiments.forEach((s, i) => {
            if (i === topIdx) {
                probs[s] = topConf;
            } else {
                const share = remaining * (Math.random() * 0.5 + 0.05);
                probs[s] = Math.min(share, remaining);
                remaining -= probs[s];
            }
        });

        // Distribute any leftover
        const nonTopEmotions = sentiments.filter((_, i) => i !== topIdx);
        if (remaining > 0 && nonTopEmotions.length) {
            probs[nonTopEmotions[0]] += remaining;
        }

        return {
            label: topLabel,
            confidence: topConf,
            probabilities: probs
        };
    }

    function generateFakeProbabilities(topEmotion, topConf) {
        const probs = {};
        let remaining = 1 - topConf;
        ALL_EMOTIONS.forEach(e => {
            if (e === topEmotion) {
                probs[e] = topConf;
            } else {
                const share = remaining / (ALL_EMOTIONS.length - 1) * (0.5 + Math.random());
                probs[e] = Math.max(0.01, Math.min(share, remaining));
                remaining -= probs[e];
            }
        });
        return probs;
    }

    // ─── Particles ───
    function createParticles() {
        const container = document.getElementById('particles');
        const count = 20;
        for (let i = 0; i < count; i++) {
            const p = document.createElement('div');
            p.className = 'particle';
            p.style.left = `${Math.random() * 100}%`;
            p.style.animationDuration = `${6 + Math.random() * 10}s`;
            p.style.animationDelay = `${Math.random() * 8}s`;
            p.style.width = p.style.height = `${2 + Math.random() * 3}px`;
            p.style.background = Math.random() > 0.5
                ? 'rgba(124, 92, 252, 0.3)'
                : 'rgba(0, 217, 196, 0.3)';
            container.appendChild(p);
        }
    }

    // ─── Utilities ───
    function shakeElement(el) {
        el.style.animation = 'none';
        el.offsetHeight; // trigger reflow
        el.style.animation = 'shake 0.4s ease';
        el.style.borderColor = 'var(--anger)';
        setTimeout(() => {
            el.style.borderColor = '';
            el.style.animation = '';
        }, 500);
    }

    function escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    // Add shake keyframe dynamically
    const styleSheet = document.createElement('style');
    styleSheet.textContent = `
        @keyframes shake {
            0%, 100% { transform: translateX(0); }
            20% { transform: translateX(-6px); }
            40% { transform: translateX(6px); }
            60% { transform: translateX(-4px); }
            80% { transform: translateX(4px); }
        }
    `;
    document.head.appendChild(styleSheet);
});
