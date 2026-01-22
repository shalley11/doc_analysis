/**
 * PDF Summarizer UI - Interactive API Testing
 */

const API_BASE = 'http://localhost:8080';
let currentBatchId = null;
let websocket = null;
let selectedFiles = [];

// =============================================================================
// Initialization
// =============================================================================

document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initUpload();
    initQA();
    initCopyButton();

    // Check for existing batch in URL or localStorage
    const urlParams = new URLSearchParams(window.location.search);
    const batchId = urlParams.get('batch') || localStorage.getItem('lastBatchId');
    if (batchId) {
        loadBatch(batchId);
    }
});

// =============================================================================
// Tab Navigation
// =============================================================================

function initTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabPanels = document.querySelectorAll('.tab-panel');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.dataset.tab;

            // Update buttons
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            // Update panels
            tabPanels.forEach(p => p.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');

            // Load data for specific tabs
            if (tabId === 'chunks' && currentBatchId) {
                loadChunks();
            }
        });
    });
}

function switchToTab(tabId) {
    document.querySelector(`[data-tab="${tabId}"]`).click();
}

// =============================================================================
// File Upload
// =============================================================================

function initUpload() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');

    // Drag and drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        handleFiles(e.dataTransfer.files);
    });

    // File input
    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
    });

    // Upload button
    uploadBtn.addEventListener('click', uploadFiles);
}

function handleFiles(files) {
    selectedFiles = Array.from(files).filter(f => f.type === 'application/pdf');
    renderSelectedFiles();
    document.getElementById('uploadBtn').disabled = selectedFiles.length === 0;
}

function renderSelectedFiles() {
    const container = document.getElementById('selectedFiles');
    if (selectedFiles.length === 0) {
        container.innerHTML = '';
        return;
    }

    container.innerHTML = selectedFiles.map((file, i) => `
        <div class="file-item">
            <div class="file-name">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                    <polyline points="14 2 14 8 20 8"></polyline>
                </svg>
                <span>${file.name}</span>
            </div>
            <span class="file-size">${formatFileSize(file.size)}</span>
            <span class="remove-file" onclick="removeFile(${i})">x</span>
        </div>
    `).join('');
}

function removeFile(index) {
    selectedFiles.splice(index, 1);
    renderSelectedFiles();
    document.getElementById('uploadBtn').disabled = selectedFiles.length === 0;
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

async function uploadFiles() {
    if (selectedFiles.length === 0) return;

    const useVision = document.getElementById('useVision').checked;
    const useSemantic = document.getElementById('useSemanticChunking').checked;

    const formData = new FormData();
    selectedFiles.forEach(file => {
        formData.append('files', file);
    });
    formData.append('use_vision', useVision ? 'yes' : 'no');
    formData.append('use_semantic_chunking', useSemantic ? 'yes' : 'no');
    formData.append('preview_only', 'no');

    showLoading('Uploading PDFs...');

    try {
        const response = await fetch(`${API_BASE}/api/v2/pdf/analyze`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            currentBatchId = data.batch_id;
            localStorage.setItem('lastBatchId', currentBatchId);
            updateBatchInfo();

            showToast('Upload successful! Processing started.', 'success');

            // Switch to status tab and connect WebSocket
            switchToTab('status');
            connectWebSocket(currentBatchId);

            // Clear selected files
            selectedFiles = [];
            renderSelectedFiles();
            document.getElementById('uploadBtn').disabled = true;
        } else {
            showToast(`Upload failed: ${data.error || 'Unknown error'}`, 'error');
        }
    } catch (error) {
        showToast(`Upload failed: ${error.message}`, 'error');
    } finally {
        hideLoading();
    }
}

// =============================================================================
// WebSocket Status Updates
// =============================================================================

function connectWebSocket(batchId) {
    if (websocket) {
        websocket.close();
    }

    const wsUrl = `ws://localhost:8080/ws/status/${batchId}`;
    console.log('Connecting to WebSocket:', wsUrl);

    websocket = new WebSocket(wsUrl);

    websocket.onopen = () => {
        console.log('WebSocket connected');
        showStatusContent();
    };

    websocket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log('WebSocket message:', data);

        if (data.type === 'status_update') {
            updateStatusDisplay(data.data);
        } else if (data.type === 'ping') {
            // Respond to keepalive
            websocket.send('pong');
        }
    };

    websocket.onclose = () => {
        console.log('WebSocket closed');
    };

    websocket.onerror = (error) => {
        console.error('WebSocket error:', error);
        // Fallback to polling
        startPolling(batchId);
    };
}

function startPolling(batchId) {
    const pollInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE}/api/v2/status/${batchId}`);
            const data = await response.json();

            updateStatusDisplay(data);

            if (data.state === 'completed' || data.state === 'failed') {
                clearInterval(pollInterval);
            }
        } catch (error) {
            console.error('Polling error:', error);
        }
    }, 2000);
}

function showStatusContent() {
    document.getElementById('noActiveJob').classList.add('hidden');
    document.getElementById('statusContent').classList.remove('hidden');
}

function updateStatusDisplay(status) {
    // Update batch ID
    document.getElementById('statusBatchId').textContent = `Batch: ${status.batch_id?.substring(0, 8) || '--'}...`;

    // Update state badge
    const stateBadge = document.getElementById('statusState');
    stateBadge.textContent = status.state || status.current_stage || '--';
    stateBadge.className = `status-badge ${status.state || status.current_stage || ''}`;

    // Update progress
    const progress = status.progress_percent || 0;
    document.getElementById('progressFill').style.width = `${progress}%`;
    document.getElementById('progressText').textContent = `${progress}%`;

    // Update message
    const message = status.message || `Processing... ${status.processed_pages || 0}/${status.total_pages || 0} pages`;
    document.getElementById('statusMessage').textContent = message;

    // Update timeline
    updateTimeline(status);

    // Update PDF details
    updatePdfDetails(status);

    // Handle completion
    if (status.state === 'completed') {
        showToast('Processing completed! Ready for Q&A.', 'success');
        enableQA();
        loadChunks();
    } else if (status.state === 'failed') {
        showToast(`Processing failed: ${status.error || 'Unknown error'}`, 'error');
    }
}

function updateTimeline(status) {
    const stages = ['initializing', 'extracting', 'vision', 'chunking', 'embedding', 'indexing', 'completed'];
    const currentStage = status.current_stage || '';
    const state = status.state || '';

    let reachedCurrent = false;

    stages.forEach(stage => {
        const item = document.querySelector(`.timeline-item[data-stage="${stage}"]`);
        if (!item) return;

        item.classList.remove('completed', 'running', 'failed');

        if (state === 'completed') {
            item.classList.add('completed');
        } else if (state === 'failed' && stage === currentStage) {
            item.classList.add('failed');
            reachedCurrent = true;
        } else if (stage === currentStage) {
            item.classList.add('running');
            reachedCurrent = true;
        } else if (!reachedCurrent) {
            item.classList.add('completed');
        }
    });
}

function updatePdfDetails(status) {
    const container = document.getElementById('pdfDetails');
    const pdfs = status.pdfs || {};

    if (Object.keys(pdfs).length === 0) {
        container.innerHTML = '';
        return;
    }

    container.innerHTML = Object.entries(pdfs).map(([name, pdf]) => `
        <div class="pdf-detail-item">
            <h4>${name}</h4>
            <div class="detail-stats">
                <span>Pages: ${pdf.processed_pages || 0}/${pdf.total_pages || 0}</span>
                <span>Stage: ${pdf.current_stage || '--'}</span>
                <span>Progress: ${pdf.progress_percent || 0}%</span>
            </div>
        </div>
    `).join('');
}

// =============================================================================
// Load Existing Batch
// =============================================================================

async function loadBatch(batchId) {
    currentBatchId = batchId;
    localStorage.setItem('lastBatchId', batchId);
    updateBatchInfo();

    try {
        // Check status
        const response = await fetch(`${API_BASE}/api/v2/status/${batchId}`);
        if (!response.ok) {
            // Try basic status
            const basicResponse = await fetch(`${API_BASE}/api/v1/pdf/status/${batchId}`);
            if (!basicResponse.ok) {
                showToast('Batch not found', 'error');
                return;
            }
            const basicData = await basicResponse.json();

            if (basicData.status === 'completed') {
                enableQA();
                loadChunks();
            }
            return;
        }

        const data = await response.json();
        showStatusContent();
        updateStatusDisplay(data);

        if (data.state === 'completed') {
            enableQA();
            loadChunks();
        } else if (data.state !== 'failed') {
            // Still processing, connect WebSocket
            connectWebSocket(batchId);
        }
    } catch (error) {
        console.error('Error loading batch:', error);
    }
}

// =============================================================================
// Chunks Viewer
// =============================================================================

async function loadChunks() {
    if (!currentBatchId) return;

    try {
        const response = await fetch(`${API_BASE}/api/v2/pdf/chunks/${currentBatchId}`);
        const data = await response.json();

        if (response.ok && data.chunks) {
            displayChunks(data.chunks);
        }
    } catch (error) {
        console.error('Error loading chunks:', error);
    }
}

function displayChunks(chunks) {
    document.getElementById('noChunks').classList.add('hidden');
    document.getElementById('chunksContent').classList.remove('hidden');

    // Update stats
    document.getElementById('totalChunks').textContent = `${chunks.length} chunks`;

    const types = {};
    const pdfs = new Set();
    chunks.forEach(c => {
        types[c.content_type] = (types[c.content_type] || 0) + 1;
        pdfs.add(c.pdf_name);
    });

    document.getElementById('chunkTypes').textContent =
        Object.entries(types).map(([t, n]) => `${n} ${t}`).join(', ');

    // Populate PDF filter
    const pdfFilter = document.getElementById('chunkPdfFilter');
    pdfFilter.innerHTML = '<option value="">All PDFs</option>' +
        Array.from(pdfs).map(p => `<option value="${p}">${p}</option>`).join('');

    // Store chunks for filtering
    window.allChunks = chunks;

    // Add filter listeners
    document.getElementById('chunkTypeFilter').onchange = filterChunks;
    document.getElementById('chunkPdfFilter').onchange = filterChunks;

    renderChunks(chunks);
}

function filterChunks() {
    const typeFilter = document.getElementById('chunkTypeFilter').value;
    const pdfFilter = document.getElementById('chunkPdfFilter').value;

    let filtered = window.allChunks || [];

    if (typeFilter) {
        filtered = filtered.filter(c => c.content_type === typeFilter);
    }
    if (pdfFilter) {
        filtered = filtered.filter(c => c.pdf_name === pdfFilter);
    }

    renderChunks(filtered);
}

function renderChunks(chunks) {
    const container = document.getElementById('chunksList');

    if (chunks.length === 0) {
        container.innerHTML = '<p class="text-center">No chunks match the filters</p>';
        return;
    }

    // Limit display for performance
    const displayChunks = chunks.slice(0, 50);

    container.innerHTML = displayChunks.map((chunk, i) => `
        <div class="chunk-card">
            <div class="chunk-header">
                <span class="chunk-id">#${i + 1} | ${chunk.chunk_id?.substring(0, 12) || '--'}...</span>
                <span class="chunk-type ${chunk.content_type}">${chunk.content_type}</span>
            </div>
            <div class="chunk-meta">
                <span>PDF: ${chunk.pdf_name}</span>
                <span>Page: ${(chunk.page_no || 0) + 1}</span>
                <span>Position: ${chunk.position || 0}</span>
            </div>
            <div class="chunk-content">${escapeHtml(chunk.text || '')}</div>
        </div>
    `).join('');

    if (chunks.length > 50) {
        container.innerHTML += `<p class="text-center mt-2">Showing 50 of ${chunks.length} chunks</p>`;
    }
}

// =============================================================================
// Q&A
// =============================================================================

function initQA() {
    document.getElementById('askBtn').addEventListener('click', askQuestion);
    document.getElementById('summarizeBtn').addEventListener('click', generateSummary);

    // Show/hide topic count based on scope
    document.getElementById('summaryScope').addEventListener('change', (e) => {
        const topicOption = document.getElementById('topicCountOption');
        topicOption.style.display = e.target.value === 'topic' ? 'flex' : 'none';
    });

    // Enter key to submit
    document.getElementById('questionInput').addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            askQuestion();
        }
    });
}

function enableQA() {
    document.getElementById('noQA').classList.add('hidden');
    document.getElementById('qaContent').classList.remove('hidden');
}

async function askQuestion() {
    const question = document.getElementById('questionInput').value.trim();
    if (!question || !currentBatchId) return;

    const topK = parseInt(document.getElementById('topK').value) || 5;
    const temperature = parseFloat(document.getElementById('temperature').value) || 0.7;

    showLoading('Asking question...');

    try {
        const response = await fetch(`${API_BASE}/api/v2/qa/ask/${currentBatchId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question,
                top_k: topK,
                temperature,
                include_sources: true
            })
        });

        const data = await response.json();

        if (response.ok) {
            displayQAResult(question, data);
            document.getElementById('questionInput').value = '';
        } else {
            showToast(`Error: ${data.error || 'Unknown error'}`, 'error');
        }
    } catch (error) {
        showToast(`Error: ${error.message}`, 'error');
    } finally {
        hideLoading();
    }
}

async function generateSummary() {
    if (!currentBatchId) return;

    const scope = document.getElementById('summaryScope').value;
    const summaryFormat = document.getElementById('summaryFormat').value;
    const numTopics = parseInt(document.getElementById('numTopics').value) || 5;
    const temperature = parseFloat(document.getElementById('temperature').value) || 0.7;

    const scopeLabels = {
        'all': 'All Documents',
        'document': 'Per Document',
        'topic': 'By Topic'
    };

    showLoading(`Generating ${scopeLabels[scope]} summary...`);

    try {
        const response = await fetch(`${API_BASE}/api/v2/qa/summarize-advanced/${currentBatchId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                scope: scope,
                summary_format: summaryFormat,
                num_topics: numTopics,
                max_chunks: 100,
                temperature
            })
        });

        const data = await response.json();

        if (response.ok) {
            displaySummaryResult(data);
        } else {
            showToast(`Error: ${data.error || 'Unknown error'}`, 'error');
        }
    } catch (error) {
        showToast(`Error: ${error.message}`, 'error');
    } finally {
        hideLoading();
    }
}

function displaySummaryResult(data) {
    const container = document.getElementById('qaResults');
    let resultHtml = '';

    if (data.scope === 'topic' && data.sections) {
        // Topic-wise summary
        resultHtml = `
            <div class="qa-result">
                <div class="qa-question">Topic-wise Summary (${data.total_topics} topics identified)</div>
                <div class="qa-answer">
                    ${data.sections.map(section => `
                        <div class="section-card">
                            <div class="section-header">${escapeHtml(section.title)}</div>
                            <div class="section-body">
                                ${escapeHtml(section.summary)}
                                <div class="section-meta">
                                    ${section.chunk_count} chunks | Sources: ${section.sources?.map(s => `${s.pdf_name} p.${s.page_no}`).join(', ') || 'N/A'}
                                </div>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    } else if (data.scope === 'document' && data.documents) {
        // Document-wise summary
        resultHtml = `
            <div class="qa-result">
                <div class="qa-question">Document-wise Summary (${data.total_documents} documents)</div>
                <div class="qa-answer">
                    ${data.documents.map(doc => `
                        <div class="document-card">
                            <div class="document-header">${escapeHtml(doc.pdf_name)}</div>
                            <div class="document-body">
                                ${escapeHtml(doc.summary)}
                                <div class="document-meta">
                                    <span>${doc.page_count} pages</span>
                                    <span>${doc.chunk_count} chunks</span>
                                </div>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    } else {
        // Combined summary (all)
        const docsIncluded = data.documents_included?.join(', ') || 'All documents';
        resultHtml = `
            <div class="qa-result">
                <div class="qa-question">Combined Summary (${data.total_documents || 1} documents, ${data.total_chunks} chunks)</div>
                <div class="qa-answer">${escapeHtml(data.summary)}</div>
                <div class="qa-sources">
                    <h4>Documents included:</h4>
                    <span class="source-item">${docsIncluded}</span>
                </div>
            </div>
        `;
    }

    container.insertAdjacentHTML('afterbegin', resultHtml);
}

function displayQAResult(question, data) {
    const container = document.getElementById('qaResults');

    const sources = (data.sources || []).map(s => `
        <span class="source-item">
            ${s.pdf_name} p.${(s.page_no || 0) + 1}
            <span class="score">${(s.score || 0).toFixed(2)}</span>
        </span>
    `).join('');

    const resultHtml = `
        <div class="qa-result">
            <div class="qa-question">Q: ${escapeHtml(question)}</div>
            <div class="qa-answer">${escapeHtml(data.answer || 'No answer')}</div>
            ${sources ? `<div class="qa-sources"><h4>Sources:</h4>${sources}</div>` : ''}
        </div>
    `;

    container.insertAdjacentHTML('afterbegin', resultHtml);
}

// =============================================================================
// Utilities
// =============================================================================

function updateBatchInfo() {
    if (currentBatchId) {
        document.getElementById('batchInfo').classList.remove('hidden');
        document.getElementById('currentBatchId').textContent = currentBatchId;
    } else {
        document.getElementById('batchInfo').classList.add('hidden');
    }
}

function initCopyButton() {
    document.getElementById('copyBatchId').addEventListener('click', () => {
        navigator.clipboard.writeText(currentBatchId).then(() => {
            showToast('Batch ID copied!', 'success');
        });
    });
}

function showLoading(text = 'Processing...') {
    document.getElementById('loadingText').textContent = text;
    document.getElementById('loadingOverlay').classList.remove('hidden');
}

function hideLoading() {
    document.getElementById('loadingOverlay').classList.add('hidden');
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
        toast.remove();
    }, 4000);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
