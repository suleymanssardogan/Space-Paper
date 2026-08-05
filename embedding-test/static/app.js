// Antispace Dashboard Client-Side Logic

// API sunucusunun adresi. Frontend ve Backend aynı sunucudaysa boş bırakın (relative path).
// Eğer frontend'i Netlify/GitHub Pages, backend'i Render/Railway'e kuracaksanız Render adresinizi girin: 'https://sunucu-adresiniz.onrender.com'
const API_BASE = '';

let currentMode = 'ask'; // 'ask' or 'search'
let healthTimer = null;
let lastQueryText = '';
let searchResultsData = []; // Store raw results for citation modal reference

// Initialize UI and run first diagnostics check
document.addEventListener('DOMContentLoaded', () => {
    runDiagnostics();
    loadSources();
    // Start polling diagnostics every 30 seconds
    healthTimer = setInterval(runDiagnostics, 30000);
});

// Fetch and populate the PDF sources filter dropdown dynamically
async function loadSources() {
    const sourceSelect = document.getElementById('param-source');
    if (!sourceSelect) return;
    
    try {
        const response = await fetch(`${API_BASE}/api/v1/sources`);
        if (!response.ok) throw new Error("Status: " + response.status);
        const sources = await response.json();
        
        // Keep only the default "All Sources" option
        sourceSelect.innerHTML = '<option value="">All Sources (Unfiltered)</option>';

        sources.forEach(src => {
            const opt = document.createElement('option');
            opt.value = src;

            // Format display name nicely
            let displayName = src;
            if (src === 'jwst_performance.pdf') {
                displayName = 'jwst_performance.pdf (JWST Performance)';
            } else if (src === 'kepler_mission.pdf') {
                displayName = 'kepler_mission.pdf (Kepler Mission)';
            }
            
            opt.textContent = displayName;
            sourceSelect.appendChild(opt);
        });
    } catch (error) {
        console.error("Failed to load PDF sources:", error);
    }
}

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
        inputField.placeholder = "Ask a scientific research question... (e.g. What is Stephan's Quintet?)";
    } else {
        inputField.placeholder = "Enter search terms for semantic search... (e.g. James Webb optical performance)";
    }
    
    // Reset output views
    hideAllOutputCards();
    document.getElementById('welcome-card').classList.remove('hidden');
}

// Set Query from Sample Card and immediately execute
function setQuery(text) {
    document.getElementById('query-input').value = text;
    handleQuery();
}

// Run Diagnostics (health check)
async function runDiagnostics() {
    const healthBadge = document.getElementById('health-badge');
    const healthText = document.getElementById('health-text');
    const dbVectors = document.getElementById('db-vectors');
    const dbLatency = document.getElementById('db-latency');

    try {
        const response = await fetch(`${API_BASE}/api/v1/health`);
        if (!response.ok) throw new Error("Status: " + response.status);
        
        const data = await response.json();
        
        if (data.status === 'healthy') {
            if (data.llm_configured) {
                healthBadge.className = 'health-badge status-healthy';
                healthText.innerHTML = '<i class="fa-solid fa-circle-check"></i> System Online';
            } else {
                healthBadge.className = 'health-badge status-warning';
                healthText.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> Offline Mode (Missing Key)';
            }
        } else {
            healthBadge.className = 'health-badge status-unhealthy';
            healthText.innerHTML = '<i class="fa-solid fa-circle-xmark"></i> Database Not Found';
        }

        dbVectors.textContent = data.vector_count.toLocaleString();
        dbLatency.textContent = data.latency_seconds.toFixed(3) + 's';

    } catch (error) {
        console.error("Diagnostics error:", error);
        healthBadge.className = 'health-badge status-unhealthy';
        healthText.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> Connection Error';
        dbVectors.textContent = 'Unavailable';
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
    const sourceSelect = document.getElementById('param-source');
    const source = sourceSelect ? sourceSelect.value : '';
    
    const submitBtn = document.getElementById('submit-btn');
    submitBtn.disabled = true;
    
    try {
        if (currentMode === 'ask') {
            await executeAskQuery(queryText, limit, scoreThreshold, source);
        } else {
            await executeSearchQuery(queryText, limit, scoreThreshold, source);
        }
    } catch (err) {
        console.error("Query Execution Error:", err);
        showError(err.message || "An unknown server error occurred.");
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

// Get score CSS class helper
function getScoreClass(score) {
    if (score >= 0.8) return 'eval-good';
    if (score >= 0.5) return 'eval-warn';
    return 'eval-poor';
}

// Execute RAG Ask Query
async function executeAskQuery(question, limit, scoreThreshold, source) {
    const response = await fetch(`${API_BASE}/api/v1/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, limit, score_threshold: scoreThreshold, source: source || null })
    });
    
    if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error (Status Code: ${response.status})`);
    }
    
    const data = await response.json();
    searchResultsData = []; // Clear old cache
    
    // Display Answer
    document.getElementById('ai-answer-text').innerHTML = formatAnswer(data.answer);
    document.getElementById('ask-latency-badge').textContent = `Latency: ${data.latency_seconds.toFixed(3)}s`;
    
    // Update Prefilter Badge in Ask Mode
    const askPrefilterBadge = document.getElementById('ask-prefilter-badge');
    if (data.prefiltered_source) {
        askPrefilterBadge.textContent = `Active (${data.prefiltered_source})`;
        askPrefilterBadge.className = 'metric-val eval-good';
    } else {
        askPrefilterBadge.textContent = `Passive`;
        askPrefilterBadge.className = 'metric-val';
    }
    
    // Render Evaluation Scores
    const evalScoresArea = document.getElementById('eval-scores-area');
    const evalFaithfulnessVal = document.getElementById('eval-faithfulness-val');
    const evalRelevanceVal = document.getElementById('eval-relevance-val');
    const evalStatusBadge = document.getElementById('eval-status-badge');
    
    evalScoresArea.classList.remove('hidden');
    
    if (data.faithfulness !== null && data.faithfulness !== undefined) {
        evalFaithfulnessVal.textContent = data.faithfulness.toFixed(2);
        evalFaithfulnessVal.className = 'metric-val ' + getScoreClass(data.faithfulness);
        
        evalRelevanceVal.textContent = data.answer_relevance.toFixed(2);
        evalRelevanceVal.className = 'metric-val ' + getScoreClass(data.answer_relevance);
        
        evalStatusBadge.textContent = "Active";
        evalStatusBadge.className = "metric-val eval-good";
    } else {
        evalFaithfulnessVal.textContent = "N/A";
        evalFaithfulnessVal.className = "metric-val eval-poor";
        
        evalRelevanceVal.textContent = "N/A";
        evalRelevanceVal.className = "metric-val eval-poor";
        
        evalStatusBadge.textContent = "Offline / Passive";
        evalStatusBadge.className = "metric-val eval-poor";
    }
    
    // Render Citations
    const citationsContainer = document.getElementById('citations-container');
    citationsContainer.innerHTML = '';
    
    if (data.citations && data.citations.length > 0) {
        data.citations.forEach((cit, idx) => {
            const tile = document.createElement('div');
            tile.className = 'citation-tile';
            tile.onclick = () => openCitationModal(cit.source, cit.page_number, cit.score, idx);
            
            tile.innerHTML = `
                <div class="citation-header">
                    <span>
                        <i class="fa-solid fa-file-pdf" style="color: var(--accent-blue);"></i> ${cit.source}
                    </span>
                    <span class="score-label">Score: ${cit.score.toFixed(4)}</span>
                </div>
                <div class="citation-body" id="cit-body-${idx}">
                    Retrieving chunk snippet...
                </div>
                <div class="citation-footer">
                    <span>Page ${cit.page_number}</span>
                    <span class="click-hint"><i class="fa-solid fa-up-right-from-square"></i> Inspect</span>
                </div>
            `;
            citationsContainer.appendChild(tile);
        });
        
        // Enrich tiles with actual texts from parallel search query
        fetchTextSnippetsForCitations(question, limit, scoreThreshold, source);
        
        document.getElementById('citations-card').classList.remove('hidden');
    } else {
        document.getElementById('citations-card').classList.add('hidden');
    }
    
    document.getElementById('answer-card').classList.remove('hidden');
}

// Execute Semantic Search Query
async function executeSearchQuery(query, limit, scoreThreshold, source) {
    const response = await fetch(`${API_BASE}/api/v1/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, limit, score_threshold: scoreThreshold, source: source || null })
    });
    
    if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error (Status: ${response.status})`);
    }
    
    const data = await response.json();
    searchResultsData = data.results || [];
    
    const container = document.getElementById('search-results-container');
    container.innerHTML = '';
    
    // Update count in header
    const headerTitle = document.querySelector('#search-results-card h3');
    headerTitle.innerHTML = `<i class="fa-solid fa-list-check"></i> Semantic Vector Matches (${searchResultsData.length})`;

    document.getElementById('search-latency-badge').textContent = `Latency: ${data.latency_seconds.toFixed(3)}s`;

    // Update Prefilter Badge in Search Mode
    const searchPrefilterBadge = document.getElementById('search-prefilter-badge');
    if (data.prefiltered_source) {
        searchPrefilterBadge.innerHTML = `<i class="fa-solid fa-filter"></i> Filter: Active (${data.prefiltered_source})`;
        searchPrefilterBadge.className = 'prefilter-badge active';
    } else {
        searchPrefilterBadge.innerHTML = `<i class="fa-solid fa-filter"></i> Filter: Passive`;
        searchPrefilterBadge.className = 'prefilter-badge inactive';
    }

    if (searchResultsData.length > 0) {
        searchResultsData.forEach((res, idx) => {
            const item = document.createElement('div');
            item.className = 'search-result-item glass-panel';
            item.innerHTML = `
                <div class="search-result-header">
                    <span class="search-result-title">
                        <i class="fa-solid fa-file-alt" style="color: var(--accent-blue);"></i> ${res.source} - Page ${res.page_number}
                    </span>
                    <span class="search-result-score">Score: ${res.score.toFixed(4)}</span>
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
                <p>No matching documents found at the specified similarity threshold.</p>
            </div>
        `;
    }
    
    document.getElementById('search-results-card').classList.remove('hidden');
}

// Parallel fetch search chunks to populate citation tiles
async function fetchTextSnippetsForCitations(query, limit, scoreThreshold, source) {
    try {
        const response = await fetch(`${API_BASE}/api/v1/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, limit, score_threshold: scoreThreshold, source: source || null })
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
                if (el) el.textContent = "Context snippet could not be resolved.";
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
    
    // Parse references like (jwst_performance.pdf, Sayfa: 4) or (jwst_performance.pdf, Page: 4)
    const regex = /\(([^)]+\.pdf),\s*(Sayfa|Page):\s*(\d+)\)/gi;
    formatted = formatted.replace(regex, (match, file, lang, page) => {
        return `<span class="badge citation-inline-pill" onclick="findAndOpenCitation('${file}', ${page})">[${file}, P. ${page}]</span>`;
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
    document.getElementById('modal-source-title').innerHTML = `<i class="fa-solid fa-file-pdf"></i> ${source} - Page ${pageNumber}`;
    
    const chunkTextEl = document.getElementById('modal-chunk-text');
    const scoreValEl = document.getElementById('modal-score');
    const charCountEl = document.getElementById('modal-char-count');
    
    let text = "Source document text not found.";
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
    scoreValEl.textContent = score > 0 ? score.toFixed(4) : "N/A";
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
        const response = await fetch(`${API_BASE}/api/v1/feedback`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, answer, score, feedback_text: "" })
        });
        
        if (response.ok) {
            // Show a temporary visual indication
            const feedbackArea = document.querySelector('.feedback-area');
            if (feedbackArea) {
                feedbackArea.innerHTML = `<span style="font-size: 0.75rem; color: var(--status-success-text);"><i class="fa-solid fa-heart"></i> Thanks for your feedback!</span>`;
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
    ingestStatus.textContent = 'Connecting to arXiv API, downloading papers...';

    try {
        const response = await fetch(`${API_BASE}/api/v1/ingest/daily`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ category: "astro-ph.CO+OR+cat:astro-ph.EP", max_results: 3 })
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.detail || `Server error (${response.status})`);
        }

        const data = await response.json();
        ingestStatus.style.color = 'var(--status-success-text)';
        ingestStatus.innerHTML = `<i class="fa-solid fa-circle-check"></i> ${data.papers_ingested} new papers added successfully!`;

        // Refresh diagnostics and sources to update total vector count and dropdown
        setTimeout(() => {
            runDiagnostics();
            loadSources();
        }, 1500);

    } catch (err) {
        console.error("Arxiv Ingestion error:", err);
        ingestStatus.style.color = 'var(--status-error-text)';
        ingestStatus.innerHTML = `<i class="fa-solid fa-circle-xmark"></i> Error: ${err.message}`;
    } finally {
        setTimeout(() => {
            ingestBtn.disabled = false;
            ingestStatus.style.display = 'none';
            ingestStatus.style.color = '';
        }, 6000);
    }
}

// Trigger PDF File Selection Dialog
function triggerPdfUpload() {
    const input = document.getElementById('pdf-upload-input');
    if (input) input.click();
}

// Handle Custom PDF File Upload
async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const toast = document.getElementById('upload-status-toast');
    const toastIcon = document.getElementById('upload-status-icon');
    const toastText = document.getElementById('upload-status-text');
    
    if (toast) {
        toast.classList.remove('hidden');
        toastIcon.innerHTML = `<i class="fa-solid fa-spinner fa-spin" style="color:var(--accent-blue);"></i>`;
        toastText.textContent = `Indexing '${file.name}'...`;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch(`${API_BASE}/api/v1/upload`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.detail || `Upload failed (Status: ${response.status})`);
        }
        
        const data = await response.json();
        if (toast) {
            toastIcon.innerHTML = `<i class="fa-solid fa-circle-check" style="color:var(--status-success-text);"></i>`;
            toastText.textContent = `${data.filename}: ${data.chunks_indexed} chunks indexed!`;
        }
        
        // Refresh sources dropdown and vector count stats
        loadSources();
        runDiagnostics();
        
    } catch (err) {
        console.error("PDF upload error:", err);
        if (toast) {
            toastIcon.innerHTML = `<i class="fa-solid fa-circle-xmark" style="color:#ef4444;"></i>`;
            toastText.textContent = `Upload error: ${err.message}`;
        }
    } finally {
        event.target.value = '';
        if (toast) {
            setTimeout(() => {
                toast.classList.add('hidden');
            }, 5000);
        }
    }
}
