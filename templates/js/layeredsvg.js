// V13 - AI Text-Prompted Layered Vectorization
// Three-phase: Keyword Detection → Segmentation → Vectorization

// Lightbox for layer previews
const lightbox = document.getElementById('image-lightbox');
const lightboxImg = document.getElementById('lightbox-img');
lightbox.addEventListener('click', () => { lightbox.style.display = 'none'; });
function openLightbox(src) {
    lightboxImg.src = src;
    lightbox.style.display = 'flex';
}

let currentRunId = null;
let currentFile = null;
let processingInterval = null;
let selectedLayers = new Set();
let allLayers = [];
let currentKeywords = [];   // [{keyword, confidence, checked}]
let currentSvgPath = null;
let reviewNewKeywords = [];  // keywords added from the review screen

// DOM elements
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const browseBtn = document.getElementById('browse-btn');
const removeBtn = document.getElementById('remove-btn');
const previewContainer = document.getElementById('preview-container');
const previewImage = document.getElementById('preview-image');
const processSection = document.getElementById('process-section');
const detectBtn = document.getElementById('detect-btn');
const progressSection = document.getElementById('progress-section');
const progressTitle = document.getElementById('progress-title');
const progressFill = document.getElementById('progress-fill');
const progressPercentage = document.getElementById('progress-percentage');
const progressMessage = document.getElementById('progress-message');
const keywordEditorSection = document.getElementById('keyword-editor-section');
const keywordList = document.getElementById('keyword-list');
const keywordCountEl = document.getElementById('keyword-count');
const keywordCheckedCountEl = document.getElementById('keyword-checked-count');
const kwSelectAllBtn = document.getElementById('kw-select-all-btn');
const kwDeselectAllBtn = document.getElementById('kw-deselect-all-btn');
const addKeywordBtn = document.getElementById('add-keyword-btn');
const segmentBtn = document.getElementById('segment-btn');
const layerReviewSection = document.getElementById('layer-review-section');
const layerGallery = document.getElementById('layer-gallery');
const selectedCountEl = document.getElementById('selected-count');
const totalCountEl = document.getElementById('total-count');
const selectAllBtn = document.getElementById('select-all-btn');
const deselectAllBtn = document.getElementById('deselect-all-btn');
const mergePreviewContainer = document.getElementById('merge-preview-container');
const mergePreviewImg = document.getElementById('merge-preview-img');
const resegmentBtn = document.getElementById('resegment-btn');
const redetectBtn = document.getElementById('redetect-btn');
const qualitySelect = document.getElementById('quality-select');
const vectorizeBtn = document.getElementById('vectorize-btn');
const resultSection = document.getElementById('result-section');
const resultSummary = document.getElementById('result-summary');
const svgPreview = document.getElementById('svg-preview');
const viewDetailsBtn = document.getElementById('view-details-btn');
const downloadSvgBtn = document.getElementById('download-svg-btn');
const historyList = document.getElementById('history-list');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    loadHistory();
});

function setupEventListeners() {
    browseBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);
    dropZone.addEventListener('dragover', handleDragOver);
    dropZone.addEventListener('dragleave', handleDragLeave);
    dropZone.addEventListener('drop', handleDrop);
    removeBtn.addEventListener('click', resetUpload);
    detectBtn.addEventListener('click', startKeywordDetection);
    addKeywordBtn.addEventListener('click', addEmptyKeyword);
    segmentBtn.addEventListener('click', startSegmentation);
    kwSelectAllBtn.addEventListener('click', () => {
        currentKeywords.forEach(kw => kw.checked = true);
        renderKeywordList();
    });
    kwDeselectAllBtn.addEventListener('click', () => {
        currentKeywords.forEach(kw => kw.checked = false);
        renderKeywordList();
    });
    selectAllBtn.addEventListener('click', () => {
        allLayers.forEach(l => selectedLayers.add(l.layer_id));
        updateLayerCards();
    });
    deselectAllBtn.addEventListener('click', () => {
        selectedLayers.clear();
        updateLayerCards();
    });
    resegmentBtn.addEventListener('click', resegmentFromReview);
    document.getElementById('add-keyword-review-btn').addEventListener('click', addKeywordFromReview);
    document.getElementById('new-keyword-input').addEventListener('keydown', (e) => {
        if (e.key === 'Enter') addKeywordFromReview();
    });
    redetectBtn.addEventListener('click', startRedetect);
    vectorizeBtn.addEventListener('click', startVectorization);
    viewDetailsBtn.addEventListener('click', () => {
        if (currentRunId) window.location.href = `/view/${currentRunId}`;
    });
    downloadSvgBtn.addEventListener('click', () => {
        if (currentSvgPath) window.open(`/results/${currentSvgPath}`, '_blank');
    });
}

// =========================================================================
// Drag & Drop + File Selection
// =========================================================================

function handleDragOver(e) {
    e.preventDefault();
    dropZone.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
}

function handleFileSelect(e) {
    if (e.target.files.length > 0) handleFile(e.target.files[0]);
}

function handleFile(file) {
    if (!file.type.match('image/(png|jpeg|jpg|webp|bmp)')) {
        alert('Please upload a PNG, JPG, WebP, or BMP image.');
        return;
    }
    if (file.size > 50 * 1024 * 1024) {
        alert('File size must be less than 50MB.');
        return;
    }
    currentFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        document.querySelector('.drop-zone-content').style.display = 'none';
        previewContainer.style.display = 'block';
        processSection.style.display = 'block';
    };
    reader.readAsDataURL(file);
}

function resetUpload() {
    currentFile = null;
    currentRunId = null;
    fileInput.value = '';
    previewImage.src = '';
    document.querySelector('.drop-zone-content').style.display = 'flex';
    previewContainer.style.display = 'none';
    processSection.style.display = 'none';
    progressSection.style.display = 'none';
    keywordEditorSection.style.display = 'none';
    layerReviewSection.style.display = 'none';
    resultSection.style.display = 'none';
    currentKeywords = [];
    allLayers = [];
    selectedLayers.clear();
}

// =========================================================================
// Phase 1: Keyword Detection
// =========================================================================

async function startKeywordDetection() {
    if (!currentFile) {
        alert('Please select a file first.');
        return;
    }

    detectBtn.disabled = true;
    detectBtn.textContent = 'Uploading...';

    try {
        // Upload file if no run_id yet
        if (!currentRunId) {
            const formData = new FormData();
            formData.append('file', currentFile);

            const uploadResponse = await fetch('/layeredsvg/upload', { method: 'POST', body: formData });
            if (!uploadResponse.ok) throw new Error('Upload failed');

            const uploadData = await uploadResponse.json();
            currentRunId = uploadData.job_id;
            pollUrl = uploadData.poll_url;
        }

        // Hide upload controls, show progress
        processSection.style.display = 'none';
        keywordEditorSection.style.display = 'none';
        layerReviewSection.style.display = 'none';
        progressSection.style.display = 'block';
        progressTitle.textContent = 'Detecting Objects...';
        resetProgress();

        // Start keyword detection
        // const response = await fetch(pollUrl, { method: 'GET' });
        // if (!response.ok) throw new Error('Keyword detection failed to start');

        // startStatusPolling('keywords');
        showKeywordEditor(["the object"]);
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred: ' + error.message);
        detectBtn.disabled = false;
        detectBtn.textContent = 'Detect Objects';
    }
}

async function startRedetect() {
    // Go back to keyword detection from layer review
    layerReviewSection.style.display = 'none';
    progressSection.style.display = 'block';
    progressTitle.textContent = 'Re-detecting Objects...';
    resetProgress();

    try {
        // const response = await fetch(`/detect_keywords/${currentRunId}`, { method: 'POST' });
        // if (!response.ok) throw new Error('Keyword detection failed to start');
        startStatusPolling('keywords');
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred: ' + error.message);
        layerReviewSection.style.display = 'block';
    }
}

// =========================================================================
// Screen 2: Keyword Editor
// =========================================================================

function showKeywordEditor(keywords) {
    progressSection.style.display = 'none';
    layerReviewSection.style.display = 'none';

    // Build keyword data (with default confidence 0.2, all checked)
    currentKeywords = keywords.map(kw => ({
        keyword: kw,
        confidence: 0.2,
        checked: true
    }));

    renderKeywordList();
    keywordEditorSection.style.display = 'block';

    // Re-enable detect button for next use
    detectBtn.disabled = false;
    detectBtn.textContent = 'Detect Objects';
}

function addKeywordFromReview() {
    const input = document.getElementById('new-keyword-input');
    const confInput = document.getElementById('new-keyword-conf');
    const keyword = input.value.trim();
    if (!keyword) return;
    const confidence = parseFloat(confInput.value) || 0.2;
    reviewNewKeywords.push({ keyword, confidence });
    // Show feedback
    const container = document.querySelector('.add-keyword-review');
    const tag = document.createElement('span');
    tag.style.cssText = 'display:inline-block; background:#e0f0e0; padding:2px 8px; border-radius:12px; font-size:0.85rem; margin-left:4px;';
    tag.textContent = `${keyword} (${confidence})`;
    container.appendChild(tag);
    input.value = '';
}

function resegmentFromReview() {
    // Collect edited keywords/confidence from layer cards, push to currentKeywords, then segment
    const editedKeywords = [];
    allLayers.forEach(layer => {
        if (layer.is_remaining) return;
        if (!selectedLayers.has(layer.layer_id)) return;
        editedKeywords.push({
            keyword: layer._editedKeyword || layer.keyword,
            confidence: layer._editedConfidence !== undefined ? layer._editedConfidence : (layer.confidence || 0.2),
            checked: true
        });
    });
    // Include newly added keywords from review screen
    reviewNewKeywords.forEach(nk => {
        editedKeywords.push({ keyword: nk.keyword, confidence: nk.confidence, checked: true });
    });
    reviewNewKeywords = [];
    if (editedKeywords.length === 0) {
        alert('No keywords to re-segment.');
        return;
    }
    currentKeywords = editedKeywords;
    // Go directly to segmentation
    startSegmentation();
}

function renderKeywordList() {
    keywordList.innerHTML = '';
    currentKeywords.forEach((kw, idx) => {
        const row = createKeywordRow(kw.keyword, kw.confidence, kw.checked, idx);
        keywordList.appendChild(row);
    });
    updateKeywordCounts();
}

function updateKeywordCounts() {
    const checkedCount = currentKeywords.filter(kw => kw.checked).length;
    keywordCountEl.textContent = currentKeywords.length;
    keywordCheckedCountEl.textContent = checkedCount;
    segmentBtn.textContent = `Segment Selected (${checkedCount})`;
    segmentBtn.disabled = checkedCount === 0;
}

function createKeywordRow(keyword, confidence, checked, index) {
    const row = document.createElement('div');
    row.className = 'keyword-row' + (checked ? '' : ' unchecked');
    row.dataset.index = index;

    // Checkbox
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.className = 'keyword-checkbox';
    cb.checked = checked;
    cb.addEventListener('change', (e) => {
        currentKeywords[index].checked = e.target.checked;
        row.classList.toggle('unchecked', !e.target.checked);
        updateKeywordCounts();
    });

    // Keyword text input (editable whether checked or not)
    const input = document.createElement('input');
    input.type = 'text';
    input.value = keyword;
    input.placeholder = 'Object keyword...';
    input.addEventListener('input', (e) => {
        currentKeywords[index].keyword = e.target.value;
    });

    // Confidence label
    const label = document.createElement('label');
    label.textContent = 'Conf:';

    // Confidence slider
    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = '0.05';
    slider.max = '0.9';
    slider.step = '0.05';
    slider.value = confidence;
    const confValue = document.createElement('span');
    confValue.className = 'confidence-value';
    confValue.textContent = confidence.toFixed(2);
    slider.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        currentKeywords[index].confidence = val;
        confValue.textContent = val.toFixed(2);
    });

    // Delete button
    const delBtn = document.createElement('button');
    delBtn.className = 'btn-remove-keyword';
    delBtn.textContent = 'X';
    delBtn.title = 'Remove keyword';
    delBtn.addEventListener('click', () => {
        currentKeywords.splice(index, 1);
        renderKeywordList();
    });

    row.appendChild(cb);
    row.appendChild(input);
    row.appendChild(label);
    row.appendChild(slider);
    row.appendChild(confValue);
    row.appendChild(delBtn);

    return row;
}

function addEmptyKeyword() {
    currentKeywords.push({ keyword: '', confidence: 0.2, checked: true });
    renderKeywordList();
    // Focus the new input
    const inputs = keywordList.querySelectorAll('input[type="text"]');
    if (inputs.length > 0) inputs[inputs.length - 1].focus();
}

// =========================================================================
// Phase 2: Segmentation
// =========================================================================

async function startSegmentation() {
    // Only send checked keywords with non-empty text
    const validKeywords = currentKeywords
        .filter(kw => kw.checked && kw.keyword.trim() !== '')
        .map(kw => ({ keyword: kw.keyword.trim(), confidence: kw.confidence }));
    if (validKeywords.length === 0) {
        alert('Please select at least one keyword.');
        return;
    }

    segmentBtn.disabled = true;
    segmentBtn.textContent = 'Starting...';

    try {
        keywordEditorSection.style.display = 'none';
        progressSection.style.display = 'block';
        progressTitle.textContent = `Segmenting ${validKeywords.length} Keywords...`;
        resetProgress();

        const response = await fetch(`/layeredsvg/segment/${currentRunId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ keywords: validKeywords })
        });

        if (!response.ok) throw new Error('Segmentation failed to start');

        startStatusPolling('segment');

    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred: ' + error.message);
        segmentBtn.disabled = false;
        segmentBtn.textContent = 'Segment All';
        keywordEditorSection.style.display = 'block';
    }
}

// =========================================================================
// Status Polling
// =========================================================================

function resetProgress() {
    progressFill.style.width = '0%';
    progressPercentage.textContent = '0%';
    progressMessage.textContent = 'Starting...';
}

function startStatusPolling(phase) {
    checkStatus(phase);
    processingInterval = setInterval(() => checkStatus(phase), 2000);
}

async function checkStatus(phase) {
    if (!currentRunId) return;

    try {
        const response = await fetch(`/job/${currentRunId}/status`);
        if (!response.ok) return;

        const status = await response.json();
        updateProgress(status);

        if (phase === 'keywords') {
            // clearInterval(processingInterval);
            showKeywordEditor(["the object"]);
        } else if (phase === 'segment' && status.status === 'layers_ready') {
            clearInterval(processingInterval);
            showLayerReview();
        } else if (phase === 'vectorize' && status.status === 'completed') {
            clearInterval(processingInterval);
            showResults(status);
        } else if (status.status === 'error') {
            clearInterval(processingInterval);
            alert('Processing failed: ' + status.message);
            // Go back to appropriate screen
            progressSection.style.display = 'none';
            if (phase === 'keywords') {
                processSection.style.display = 'block';
                detectBtn.disabled = false;
                detectBtn.textContent = 'Detect Objects';
            } else if (phase === 'segment') {
                segmentBtn.disabled = false;
                segmentBtn.textContent = 'Segment All';
                keywordEditorSection.style.display = 'block';
            } else if (phase === 'vectorize') {
                layerReviewSection.style.display = 'block';
                vectorizeBtn.disabled = false;
                vectorizeBtn.textContent = `Vectorize Selected Layers (${selectedLayers.size})`;
            }
        }
    } catch (error) {
        console.error('Status check error:', error);
    }
}

function updateProgress(status) {
    const progress = status.progress || 0;
    progressFill.style.width = progress + '%';
    progressPercentage.textContent = progress + '%';
    progressMessage.textContent = status.message || 'Processing...';
}

// =========================================================================
// Screen 3: Layer Review Gallery
// =========================================================================

async function showLayerReview() {
    progressSection.style.display = 'none';
    keywordEditorSection.style.display = 'none';

    // Re-enable segment button for next use
    segmentBtn.disabled = false;
    segmentBtn.textContent = 'Segment All';

    try {
        const response = await fetch(`/layeredsvg/layers/${currentRunId}`);
        if (!response.ok) throw new Error('Failed to load layers');

        const data = await response.json();
        allLayers = data.layers;

        // Select all layers by default
        selectedLayers.clear();
        allLayers.forEach(l => selectedLayers.add(l.layer_id));

        // Show merge preview if available (cache-bust for re-segment)
        const cacheBust = Date.now();
        const mergeLayer = allLayers.find(l => l.merge_preview_url);
        if (data.merge_preview_url) {
            mergePreviewImg.src = `/layer_asset/${currentRunId}/${data.merge_preview_url}?t=${cacheBust}`;
            mergePreviewContainer.style.display = 'block';
        } else {
            mergePreviewContainer.style.display = 'none';
        }

        // Clear added-keyword tags from review screen
        reviewNewKeywords = [];
        const addKwContainer = document.querySelector('.add-keyword-review');
        if (addKwContainer) {
            addKwContainer.querySelectorAll('span').forEach(s => s.remove());
        }

        // Build gallery
        layerGallery.innerHTML = '';
        allLayers.forEach(layer => {
            const card = createLayerCard(layer, cacheBust);
            layerGallery.appendChild(card);
        });

        totalCountEl.textContent = allLayers.length;
        updateLayerCards();

        layerReviewSection.style.display = 'block';

    } catch (error) {
        console.error('Error loading layers:', error);
        alert('Failed to load layer previews');
    }
}

function createLayerCard(layer, cacheBust) {
    const card = document.createElement('div');
    card.className = 'layer-card selected';
    card.dataset.layerId = layer.layer_id;

    // Header: checkbox + badge
    const header = document.createElement('div');
    header.className = 'layer-card-header';

    const title = document.createElement('h4');
    if (layer.is_remaining) {
        title.textContent = 'Remaining ';
        const badge = document.createElement('span');
        badge.className = 'badge-remaining';
        badge.textContent = 'auto';
        title.appendChild(badge);
    } else {
        title.textContent = (layer.keyword || `Layer ${layer.layer_id}`) + ' ';
    }

    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.className = 'layer-checkbox';
    checkbox.checked = true;
    if (layer.is_remaining) {
        checkbox.disabled = true;
    } else {
        checkbox.addEventListener('change', (e) => {
            e.stopPropagation();
            toggleLayer(layer.layer_id, checkbox.checked);
        });
    }

    header.appendChild(title);
    header.appendChild(checkbox);

    // Preview image
    const img = document.createElement('img');
    if (layer.preview_url) {
        img.src = `/layer_asset/${currentRunId}/${layer.preview_url}?t=${cacheBust}`;
    }
    img.alt = layer.keyword || `Layer ${layer.layer_id}`;
    img.loading = 'lazy';
    img.style.cursor = 'pointer';
    img.addEventListener('click', (e) => {
        e.stopPropagation();
        openLightbox(img.src);
    });

    card.appendChild(header);
    card.appendChild(img);

    // Editable keyword + confidence (not for "remaining" layer)
    if (!layer.is_remaining) {
        const editDiv = document.createElement('div');
        editDiv.className = 'layer-card-edit';

        const kwInput = document.createElement('input');
        kwInput.type = 'text';
        kwInput.value = layer.keyword || '';
        kwInput.placeholder = 'Keyword...';
        kwInput.addEventListener('input', (e) => {
            layer._editedKeyword = e.target.value;
        });

        const confRow = document.createElement('div');
        confRow.className = 'conf-row';
        const confLabel = document.createElement('span');
        confLabel.textContent = 'Conf:';
        const confSlider = document.createElement('input');
        confSlider.type = 'range';
        confSlider.min = '0.05';
        confSlider.max = '0.9';
        confSlider.step = '0.05';
        confSlider.value = layer.confidence || 0.2;
        const confVal = document.createElement('span');
        confVal.className = 'conf-val';
        confVal.textContent = (layer.confidence || 0.2).toFixed(2);
        confSlider.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value);
            layer._editedConfidence = val;
            confVal.textContent = val.toFixed(2);
        });

        confRow.appendChild(confLabel);
        confRow.appendChild(confSlider);
        confRow.appendChild(confVal);

        editDiv.appendChild(kwInput);
        editDiv.appendChild(confRow);
        card.appendChild(editDiv);
    }

    // Meta info
    const meta = document.createElement('div');
    meta.className = 'layer-meta';
    const areaPct = layer.area_pct !== undefined ? layer.area_pct.toFixed(1) : '?';
    meta.innerHTML = `<span>Area: ${areaPct}%</span>`;
    card.appendChild(meta);

    // Click card to toggle checkbox (but not on inputs)
    card.addEventListener('click', (e) => {
        if (e.target.type === 'checkbox' || e.target.type === 'text' || e.target.type === 'range') return;
        const cb = card.querySelector('.layer-checkbox');
        cb.checked = !cb.checked;
        toggleLayer(layer.layer_id, cb.checked);
    });

    return card;
}

function toggleLayer(layerId, selected) {
    if (selected) {
        selectedLayers.add(layerId);
    } else {
        selectedLayers.delete(layerId);
    }
    updateLayerCards();
}

function updateLayerCards() {
    const cards = layerGallery.querySelectorAll('.layer-card');
    cards.forEach(card => {
        const lid = parseInt(card.dataset.layerId);
        const isSelected = selectedLayers.has(lid);
        const checkbox = card.querySelector('.layer-checkbox');
        checkbox.checked = isSelected;
        card.className = 'layer-card ' + (isSelected ? 'selected' : 'deselected');
    });

    const count = selectedLayers.size;
    selectedCountEl.textContent = count;
    vectorizeBtn.textContent = `Vectorize Selected Layers (${count})`;
    vectorizeBtn.disabled = count === 0;
}

// =========================================================================
// Phase 3: Vectorize Confirmed Layers
// =========================================================================

async function startVectorization() {
    if (selectedLayers.size === 0) {
        alert('Please select at least one layer.');
        return;
    }

    vectorizeBtn.disabled = true;
    vectorizeBtn.textContent = 'Starting...';

    try {
        const response = await fetch(`/vectorize/${currentRunId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                selected_layers: Array.from(selectedLayers),
                quality: qualitySelect.value
            })
        });

        if (!response.ok) throw new Error('Vectorization failed to start');

        layerReviewSection.style.display = 'none';
        progressSection.style.display = 'block';
        progressTitle.textContent = `Vectorizing ${selectedLayers.size} Layers (SDS + DiffVG)`;
        resetProgress();

        startStatusPolling('vectorize');

    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred: ' + error.message);
        vectorizeBtn.disabled = false;
        vectorizeBtn.textContent = `Vectorize Selected Layers (${selectedLayers.size})`;
    }
}

// =========================================================================
// Screen 4: Results
// =========================================================================

function showResults(status) {
    progressSection.style.display = 'none';
    resultSection.style.display = 'block';

    const nLayers = status.n_layers || 'multiple';
    resultSummary.innerHTML = `
        <p><strong>${nLayers} editable layers</strong> generated successfully!</p>
        <p>Each layer can be independently moved, scaled, or edited in vector editors like Inkscape, Illustrator, or Figma.</p>
    `;

    currentSvgPath = status.result_svg;

    if (status.result_png) {
        const imgElement = document.createElement('img');
        imgElement.src = `/results/${status.result_png}`;
        imgElement.style.maxWidth = '100%';
        imgElement.style.maxHeight = '500px';
        imgElement.alt = 'Vectorization Result';
        svgPreview.innerHTML = '';
        svgPreview.appendChild(imgElement);
    } else if (status.result_svg) {
        const svgObject = document.createElement('object');
        svgObject.data = `/results/${status.result_svg}`;
        svgObject.type = 'image/svg+xml';
        svgObject.style.maxWidth = '100%';
        svgObject.style.maxHeight = '500px';
        svgPreview.innerHTML = '';
        svgPreview.appendChild(svgObject);
    }

    loadHistory();
}

// =========================================================================
// History
// =========================================================================

async function loadHistory() {
    try {
        const response = await fetch('/results');
        if (!response.ok) return;

        const data = await response.json();
        const results = data.results || [];

        if (results.length === 0) {
            historyList.innerHTML = '<p class="loading">No conversions yet</p>';
            return;
        }

        historyList.innerHTML = '';
        results.forEach(result => {
            const div = document.createElement('div');
            div.className = 'history-item';

            const h4 = document.createElement('h4');
            h4.textContent = result.run_id;

            const p = document.createElement('p');
            if (result.has_svg) {
                p.textContent = 'Completed';
                p.className = 'completed';
            } else if (result.has_meta) {
                p.textContent = 'Layers ready';
                p.className = 'processing';
            } else {
                p.textContent = 'In progress';
                p.className = 'processing';
            }

            div.appendChild(h4);
            div.appendChild(p);

            if (result.has_svg || result.has_meta) {
                div.addEventListener('click', () => {
                    window.location.href = `/view/${result.run_id}`;
                });
                div.style.cursor = 'pointer';
            }

            historyList.appendChild(div);
        });
    } catch (error) {
        console.error('History load error:', error);
        historyList.innerHTML = '<p class="loading">Error loading history</p>';
    }
}
