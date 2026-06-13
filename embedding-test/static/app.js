// Antispace Dashboard Client-Side Logic

let currentMode = 'ask'; // 'ask' or 'search'
let healthTimer = null;
let lastQueryText = '';
let searchResultsData = []; // Store raw results for citation modal reference

// Initialize UI and run first diagnostics check
document.addEventListener('DOMContentLoaded', () => {
    runDiagnostics();
    // Start polling diagnostics every 30 seconds
    healthTimer = setInterval(runDiagnostics, 30000);
});

// Update display parameters
function updateLimitValue(val) {
    document.getElementById('val-limit').textContent = val;
}

function updateThresholdValue(val) {
    document.getElementById('val-threshold').textContent = val;
}

// Switch Mode (RAG Q&A vs Semantic Search)
function switchMode(mode) {
    currentMode = mode;
    
    // Toggle active buttons
    document.getElementById('mode-ask-btn').classList.toggle('active', mode === 'ask');
    document.getElementById('mode-search-btn').classList.toggle('active', mode === 'search');
    
    // Toggle placeholder text
    const inputField = document.getElementById('query-input');
    if (mode === 'ask') {
        inputField.placeholder = "Uzay bilimleriyle ilgili sorunuzu buraya yazın... (örn: What is Stephan's Quintet?)";
    } else {
        inputField.placeholder = "Semantik arama için terimler yazın... (örn: James Webb Optical Performance)";
    }
    
    // Reset output views
    hideAllOutputCards();
    document.getElementById('welcome-card').classList.remove('hidden');
}

// Set Query from Sample Chip
function setQuery(text) {
    document.getElementById('query-input').value = text;
    document.getElementById('query-input').focus();
}

// Run Diagnostics (health check)
async function runDiagnostics() {
    const healthBadge = document.getElementById('health-badge');
    const healthText = document.getElementById('health-text');
    const dbVectors = document.getElementById('db-vectors');
    const dbLatency = document.getElementById('db-latency');

    try {
        const response = await fetch('/api/v1/health');
        if (!response.ok) throw new Error("Status: " + response.status);
        
        const data = await response.json();
        
        if (data.status === 'healthy') {
            if (data.llm_configured) {
                healthBadge.className = 'health-badge status-healthy';
                healthText.innerHTML = '<i class="fa-solid fa-circle-check"></i> Sistem Çevrimiçi';
            } else {
                healthBadge.className = 'health-badge status-warning';
                healthText.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> Çevrimdışı Mod (Key Eksik)';
            }
        } else {
            healthBadge.className = 'health-badge status-unhealthy';
            healthText.innerHTML = '<i class="fa-solid fa-circle-xmark"></i> Veritabanı Bulunamadı';
        }
        
        dbVectors.textContent = data.vector_count.toLocaleString();
        dbLatency.textContent = data.latency_seconds.toFixed(3) + ' sn';
        
    } catch (error) {
        console.error("Diagnostics error:", error);
        healthBadge.className = 'health-badge status-unhealthy';
        healthText.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> Bağlantı Hatası';
        dbVectors.textContent = 'Bağlanılamadı';
        dbLatency.textContent = '-';
    }
}

// Hide all outputs
function hideAllOutputCards() {
    document.getElementById('welcome-card').classList.add('hidden');
    document.getElementById('loading-card').classList.add('hidden');
    document.getElementById('error-card').classList.add('hidden');
    document.getElementById('answer-card').classList.add('hidden');
    document.getElementById('citations-card').classList.add('hidden');
    document.getElementById('search-results-card').classList.add('hidden');
}

// Handle Form Submission
async function handleQuery(event) {
    if (event) event.preventDefault();
    
    const queryInput = document.getElementById('query-input');
    const queryText = queryInput.value.trim();
    if (!queryText) return;
    
    lastQueryText = queryText;
    hideAllOutputCards();
    document.getElementById('loading-card').classList.remove('hidden');
    
    const limit = parseInt(document.getElementById('param-limit').value);
    const scoreThreshold = parseFloat(document.getElementById('param-threshold').value);
    
    const submitBtn = document.getElementById('submit-btn');
    submitBtn.disabled = true;
    
    try {
        if (currentMode === 'ask') {
            await executeAskQuery(queryText, limit, scoreThreshold);
        } else {
            await executeSearchQuery(queryText, limit, scoreThreshold);
        }
    } catch (err) {
        console.error("Query Execution Error:", err);
        showError(err.message || "Bilinmeyen sunucu hatası oluştu.");
    } finally {
        submitBtn.disabled = false;
        document.getElementById('loading-card').classList.add('hidden');
    }
}

// Retry last query
function retryLastQuery() {
    if (lastQueryText) {
        document.getElementById('query-input').value = lastQueryText;
        handleQuery();
    }
}

// Show Error Panel
function showError(msg) {
    hideAllOutputCards();
    document.getElementById('error-message').textContent = msg;
    document.getElementById('error-card').classList.remove('hidden');
}

// Execute RAG Ask Query
async function executeAskQuery(question, limit, scoreThreshold) {
    const response = await fetch('/api/v1/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, limit, score_threshold: scoreThreshold })
    });
    
    if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Sunucu hatası (Status Code: ${response.status})`);
    }
    
    const data = await response.json();
    searchResultsData = []; // Clear old cache
    
    // Display Answer
    document.getElementById('ai-answer-text').innerHTML = formatAnswer(data.answer);
    document.getElementById('ask-latency-badge').textContent = `Latency: ${data.latency_seconds.toFixed(3)}s`;
    
    // Render Citations
    const citationsContainer = document.getElementById('citations-container');
    citationsContainer.innerHTML = '';
    
    if (data.citations && data.citations.length > 0) {
        data.citations.forEach((cit, idx) => {
            // Mock or find textual context for rendering inside tile.
            // Since `/api/v1/ask` currently does not return chunk texts directly in `citations` response model 
            // (it only returns metadata: source, page_number, score), we will load search results in parallel 
            // or fetch them to enrich tiles with snippets! This is a super elegant premium touch.
            // We can fetch from `/api/v1/search` with the same question/limits to align text chunks!
            // Let's create an elegant fallback text if search snippet isn't resolved.
            
            const tile = document.createElement('div');
            tile.className = 'citation-tile glass-panel';
            tile.onclick = () => openCitationModal(cit.source, cit.page_number, cit.score, idx);
            
            tile.innerHTML = `
                <div class="citation-header">
                    <span class="citation-title-file">
                        <i class="fa-solid fa-file-pdf" style="color: var(--accent-pink);"></i> Kaynak #${idx+1}
                    </span>
                    <span class="score-label">Skor: ${cit.score.toFixed(4)}</span>
                </div>
                <div class="citation-body" id="cit-body-${idx}">
                    Yükleniyor...
                </div>
                <div class="citation-footer">
                    <span>${cit.source} (Sayfa ${cit.page_number})</span>
                    <span class="click-hint"><i class="fa-solid fa-expand"></i> İncele</span>
                </div>
            `;
            citationsContainer.appendChild(tile);
        });
        
        // Enrich tiles with actual texts from parallel search query
        fetchTextSnippetsForCitations(question, limit, scoreThreshold);
        
        document.getElementById('citations-card').classList.remove('hidden');
    } else {
        document.getElementById('citations-card').classList.add('hidden');
    }
    
    document.getElementById('answer-card').classList.remove('hidden');
}

// Execute Semantic Search Query
async function executeSearchQuery(query, limit, scoreThreshold) {
    const response = await fetch('/api/v1/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, limit, score_threshold: scoreThreshold })
    });
    
    if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Sunucu hatası (Status: ${response.status})`);
    }
    
    const data = await response.json();
    searchResultsData = data.results || [];
    
    const container = document.getElementById('search-results-container');
    container.innerHTML = '';
    
    // Update count in header
    const headerTitle = document.querySelector('#search-results-card h3');
    headerTitle.innerHTML = `<i class="fa-solid fa-list-check"></i> Semantik Eşleşmeler (${searchResultsData.length})`;
    
    document.getElementById('search-latency-badge').textContent = `Latency: ${data.latency_seconds.toFixed(3)}s`;
    
    if (searchResultsData.length > 0) {
        searchResultsData.forEach((res, idx) => {
            const item = document.createElement('div');
            item.className = 'search-result-item glass-panel';
            item.innerHTML = `
                <div class="search-result-header">
                    <span class="search-result-title">
                        <i class="fa-solid fa-file-alt" style="color: var(--accent-blue);"></i> ${res.source} - Sayfa ${res.page_number}
                    </span>
                    <span class="search-result-score">Skor: ${res.score.toFixed(4)}</span>
                </div>
                <div class="search-result-body">
                    ${escapeHtml(res.text)}
                </div>
            `;
            container.appendChild(item);
        });
    } else {
        container.innerHTML = `
            <div style="text-align: center; padding: 2rem; color: var(--text-muted);">
                <i class="fa-solid fa-ban" style="font-size: 2rem; margin-bottom: 0.5rem;"></i>
                <p>Belirtilen benzerlik eşiğinde hiçbir eşleşen belge bulunamadı.</p>
            </div>
        `;
    }
    
    document.getElementById('search-results-card').classList.remove('hidden');
}

// Parallel fetch search chunks to populate citation tiles
async function fetchTextSnippetsForCitations(query, limit, scoreThreshold) {
    try {
        const response = await fetch('/api/v1/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, limit, score_threshold: scoreThreshold })
        });
        if (response.ok) {
            const data = await response.json();
            searchResultsData = data.results || [];
            
            // Map text to citation tiles
            searchResultsData.forEach((res, idx) => {
                const el = document.getElementById(`cit-body-${idx}`);
                if (el) {
                    el.textContent = res.text;
                }
            });
            
            // Clean up any remaining loading indicators if we got fewer search results
            for (let i = searchResultsData.length; i < limit; i++) {
                const el = document.getElementById(`cit-body-${i}`);
                if (el) el.textContent = "Bağlam detay metni çözümlenemedi.";
            }
        }
    } catch (e) {
        console.warn("Failed to load snippet enrichments:", e);
    }
}

// Format LLM output text (escaping HTML and styling page references)
function formatAnswer(text) {
    if (!text) return "";
    let formatted = escapeHtml(text);
    
    // Parse references like (jwst_performance.pdf, Sayfa: 4)
    const regex = /\(([^)]+\.pdf),\s*(Sayfa|Page):\s*(\d+)\)/gi;
    formatted = formatted.replace(regex, (match, file, lang, page) => {
        return `<span class="badge" style="cursor: pointer; background: rgba(236, 72, 153, 0.12); color: var(--accent-pink); border-color: rgba(236, 72, 153, 0.2); font-size: 0.75rem;" onclick="findAndOpenCitation('${file}', ${page})">${file} [S: ${page}]</span>`;
    });
    
    // Also parse markdown-like bold text
    formatted = formatted.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    
    // Break lines
    formatted = formatted.replace(/\n/g, '<br>');
    return formatted;
}

// Find citation in cache and open modal
function findAndOpenCitation(filename, page) {
    const idx = searchResultsData.findIndex(r => r.source.toLowerCase() === filename.toLowerCase() && parseInt(r.page_number) === parseInt(page));
    if (idx !== -1) {
        openCitationModal(searchResultsData[idx].source, searchResultsData[idx].page_number, searchResultsData[idx].score, idx);
    } else {
        openCitationModal(filename, page, 0.0, -1);
    }
}

// Open modal dialog for a citation item
function openCitationModal(source, pageNumber, score, cacheIdx) {
    const modal = document.getElementById('citation-modal');
    document.getElementById('modal-source-title').innerHTML = `<i class="fa-solid fa-file-pdf"></i> ${source} - Sayfa ${pageNumber}`;
    
    const chunkTextEl = document.getElementById('modal-chunk-text');
    const scoreValEl = document.getElementById('modal-score');
    const charCountEl = document.getElementById('modal-char-count');
    
    let text = "Kaynak belge metni bulunamadı.";
    let charCount = 0;
    
    if (cacheIdx !== -1 && searchResultsData[cacheIdx]) {
        text = searchResultsData[cacheIdx].text;
        charCount = text.length;
    } else {
        // Fallback search if cached index not matching directly
        const matched = searchResultsData.find(r => r.source === source && parseInt(r.page_number) === parseInt(pageNumber));
        if (matched) {
            text = matched.text;
            charCount = text.length;
            score = matched.score;
        }
    }
    
    chunkTextEl.textContent = text;
    scoreValEl.textContent = score > 0 ? score.toFixed(4) : "Belirtilmemiş";
    charCountEl.textContent = charCount;
    
    modal.classList.remove('hidden');
}

// Close Modal
function closeModal(event) {
    const modal = document.getElementById('citation-modal');
    modal.classList.add('hidden');
}

// Helpers
function escapeHtml(text) {
    const div = document.createElement('div');
    div.innerText = text;
    return div.innerHTML;
}

// Submit User Feedback (Thumbs Up / Down)
async function submitFeedback(score) {
    const question = lastQueryText;
    const answerElement = document.getElementById('ai-answer-text');
    const answer = answerElement ? answerElement.innerText : "";
    
    if (!question) return;
    
    // Disable feedback buttons to prevent double submission
    const feedbackBtns = document.querySelectorAll('.feedback-btn');
    feedbackBtns.forEach(btn => btn.disabled = true);
    
    try {
        const response = await fetch('/api/v1/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, answer, score, feedback_text: "" })
        });
        
        if (response.ok) {
            // Show a temporary visual indication
            const feedbackArea = document.querySelector('.feedback-area');
            if (feedbackArea) {
                feedbackArea.innerHTML = `<span style="font-size: 0.75rem; color: var(--accent-green);"><i class="fa-solid fa-heart"></i> Geri bildiriminiz için teşekkürler!</span>`;
            }
        }
    } catch (err) {
        console.error("Feedback submission error:", err);
    }
}

// Trigger arXiv Daily Ingestion
async function triggerArxivIngest() {
    const ingestBtn = document.getElementById('ingest-btn');
    const ingestStatus = document.getElementById('ingest-status');
    
    if (!ingestBtn) return;
    
    ingestBtn.disabled = true;
    ingestStatus.style.display = 'block';
    ingestStatus.textContent = 'Canlı arXiv API bağlanıyor, makaleler indiriliyor...';
    
    try {
        const response = await fetch('/api/v1/ingest/daily', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ category: "astro-ph.CO+OR+cat:astro-ph.EP", max_results: 3 })
        });
        
        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.detail || `Sunucu hatası (${response.status})`);
        }
        
        const data = await response.json();
        ingestStatus.style.color = 'var(--accent-green)';
        ingestStatus.innerHTML = `<i class="fa-solid fa-circle-check"></i> ${data.papers_ingested} yeni makale başarıyla eklendi!`;
        
        // Refresh diagnostics to update total vector count
        setTimeout(() => {
            runDiagnostics();
        }, 1500);
        
    } catch (err) {
        console.error("Arxiv Ingestion error:", err);
        ingestStatus.style.color = '#ef4444';
        ingestStatus.innerHTML = `<i class="fa-solid fa-circle-xmark"></i> Hata: ${err.message}`;
    } finally {
        setTimeout(() => {
            ingestBtn.disabled = false;
            ingestStatus.style.display = 'none';
            ingestStatus.style.color = 'var(--accent-pink)';
        }, 6000);
    }
}
