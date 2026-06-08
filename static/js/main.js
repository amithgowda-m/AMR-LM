document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const textarea = document.getElementById('dna-sequence');
    const charCount = document.getElementById('char-count');
    const analyzeBtn = document.getElementById('analyze-btn');
    const btnText = analyzeBtn.querySelector('.btn-text');
    const loader = analyzeBtn.querySelector('.loader');
    const loadSampleBtn = document.getElementById('load-sample-btn');
    
    const resultSection = document.getElementById('result-section');
    const errorBanner = document.getElementById('error-banner');
    const errorMessage = document.getElementById('error-message');
    
    // Result DOM Elements
    const predictionTitle = document.getElementById('prediction-title');
    const predictionBadge = document.getElementById('prediction-badge');
    const confidenceValue = document.getElementById('confidence-value');
    const confidenceFill = document.getElementById('confidence-fill');
    const probAmr = document.getElementById('prob-amr');
    const probNonAmr = document.getElementById('prob-non-amr');

    // Sidebar DOM Elements
    const appLayout = document.querySelector('.app-layout');
    const sidebar = document.getElementById('history-sidebar');
    const toggleSidebarBtn = document.getElementById('toggle-sidebar-btn');
    const closeSidebarBtn = document.getElementById('close-sidebar-btn');
    const historyList = document.getElementById('history-list');

    // Sample AMR Sequence (mecA gene fragment commonly found in MRSA)
    const sampleSequence = "ATGAAGATACAAGCGCTTTGCCGCTATTTCGACAAAATGAAAACACTGAATTACGTTAAGTCTCAAAACAGAAAATCGTCCCGTCTCAAAGAAACAGGTAA";

    // Initialize History
    fetchHistory();

    // Sidebar Toggles
    toggleSidebarBtn.addEventListener('click', () => {
        sidebar.classList.add('open');
        appLayout.classList.add('sidebar-open');
    });

    closeSidebarBtn.addEventListener('click', () => {
        sidebar.classList.remove('open');
        appLayout.classList.remove('sidebar-open');
    });

    // Update character count
    textarea.addEventListener('input', () => {
        // Remove whitespace for counting
        const cleanSeq = textarea.value.replace(/\s/g, '');
        charCount.textContent = cleanSeq.length.toLocaleString();
        
        // Auto-hide errors/results when typing new sequence
        if (cleanSeq.length === 0) {
            hideResults();
        }
    });

    // Load sample sequence
    loadSampleBtn.addEventListener('click', () => {
        textarea.value = sampleSequence;
        // Trigger input event to update count
        textarea.dispatchEvent(new Event('input'));
        
        // Optional: Add a subtle flash effect to textarea
        textarea.style.backgroundColor = 'rgba(37, 99, 235, 0.05)';
        setTimeout(() => {
            textarea.style.backgroundColor = '';
        }, 300);
    });

    // Form submission
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const sequence = textarea.value.trim();
        if (!sequence) return;

        // UI Loading State
        setLoadingState(true);
        hideResults();

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ sequence })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Failed to analyze sequence');
            }

            displayResults(data);
            
            // Add to history
            addHistoryItemToDOM({
                id: data.id,
                sequence_preview: sequence.length > 100 ? sequence.substring(0, 100) + '...' : sequence,
                prediction: data.prediction,
                confidence: data.confidence,
                timestamp: new Date().toISOString()
            }, true);

        } catch (error) {
            showError(error.message);
        } finally {
            setLoadingState(false);
        }
    });

    function setLoadingState(isLoading) {
        if (isLoading) {
            analyzeBtn.disabled = true;
            btnText.classList.add('hidden');
            loader.classList.remove('hidden');
            textarea.disabled = true;
        } else {
            analyzeBtn.disabled = false;
            btnText.classList.remove('hidden');
            loader.classList.add('hidden');
            textarea.disabled = false;
        }
    }

    function hideResults() {
        resultSection.classList.add('hidden');
        errorBanner.classList.add('hidden');
        
        // Reset progress bar to allow re-animation
        confidenceFill.style.width = '0%';
    }

    function showError(message) {
        errorMessage.textContent = message;
        errorBanner.classList.remove('hidden');
    }

    function displayResults(data) {
        // Update Title & Badge
        const isAmr = data.prediction === "AMR Gene";
        
        predictionTitle.textContent = "Analysis Complete";
        predictionBadge.textContent = data.prediction;
        
        // Setup Badge styling
        predictionBadge.className = 'badge'; // reset
        predictionBadge.classList.add(isAmr ? 'amr' : 'non-amr');
        
        // Update Confidence
        confidenceValue.textContent = `${data.confidence}%`;
        
        // Animate Progress Bar
        setTimeout(() => {
            confidenceFill.style.width = `${data.confidence}%`;
            // Change color based on prediction
            if (isAmr) {
                confidenceFill.style.background = '#dc2626'; // --danger
            } else {
                confidenceFill.style.background = '#16a34a'; // --success
            }
        }, 100);

        // Update Probabilities
        probAmr.textContent = data.probabilities.amr.toFixed(4);
        probNonAmr.textContent = data.probabilities.non_amr.toFixed(4);

        // Show section
        resultSection.classList.remove('hidden');
        
        // Scroll into view if needed
        resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    // --- History Functions ---
    async function fetchHistory() {
        try {
            const res = await fetch('/api/history');
            if (res.ok) {
                const data = await res.json();
                renderHistory(data.history || []);
            }
        } catch (e) {
            console.error('Error fetching history:', e);
        }
    }

    function renderHistory(historyItems) {
        if (historyItems.length === 0) return;
        historyList.innerHTML = ''; // clear empty message
        historyItems.forEach(item => addHistoryItemToDOM(item, false));
    }

    function addHistoryItemToDOM(item, prepend = false) {
        // Remove empty state message if present
        const emptyMsg = historyList.querySelector('.history-empty');
        if (emptyMsg) {
            emptyMsg.remove();
        }

        const isAmr = item.prediction === "AMR Gene";
        const badgeClass = isAmr ? 'amr' : 'non-amr';
        const dateStr = new Date(item.timestamp).toLocaleString();

        const div = document.createElement('div');
        div.className = 'history-item';
        div.innerHTML = `
            <div class="history-item-header">
                <span class="badge ${badgeClass}">${item.prediction}</span>
                <span class="history-item-time" title="${dateStr}">${timeAgo(item.timestamp)}</span>
            </div>
            <div class="history-item-seq" title="Click to load sequence">${item.sequence_preview}</div>
            <div class="history-item-conf">Confidence: ${item.confidence.toFixed(1)}%</div>
        `;

        // Click to load sequence
        div.addEventListener('click', () => {
            textarea.value = item.sequence_preview.replace('...', '');
            textarea.dispatchEvent(new Event('input'));
            hideResults();
            if (window.innerWidth <= 900) {
                sidebar.classList.remove('open');
                appLayout.classList.remove('sidebar-open');
            }
        });

        if (prepend) {
            historyList.prepend(div);
        } else {
            historyList.appendChild(div);
        }
    }

    function timeAgo(dateString) {
        const date = new Date(dateString);
        // Fallback for missing 'Z' or invalid dates
        if (isNaN(date)) return "Recently";
        
        const seconds = Math.floor((new Date() - date) / 1000);
        let interval = seconds / 31536000;
        if (interval > 1) return Math.floor(interval) + " years ago";
        interval = seconds / 2592000;
        if (interval > 1) return Math.floor(interval) + " months ago";
        interval = seconds / 86400;
        if (interval > 1) return Math.floor(interval) + " days ago";
        interval = seconds / 3600;
        if (interval > 1) return Math.floor(interval) + " hours ago";
        interval = seconds / 60;
        if (interval > 1) return Math.floor(interval) + " minutes ago";
        return Math.floor(seconds) + " seconds ago";
    }
});
