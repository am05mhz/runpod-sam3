// Global variables - V11 Version (app10.py)
// Combines App3 SVG quality with App8 layer editability
let currentRunId = null;
let currentFile = null;
let processingInterval = null;

// DOM elements
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const browseBtn = document.getElementById('browse-btn');
const removeBtn = document.getElementById('remove-btn');
const previewContainer = document.getElementById('preview-container');
const previewImage = document.getElementById('preview-image');
const processSection = document.getElementById('process-section');
const processBtn = document.getElementById('process-btn');
const progressSection = document.getElementById('progress-section');
const progressFill = document.getElementById('progress-fill');
const progressPercentage = document.getElementById('progress-percentage');
const progressMessage = document.getElementById('progress-message');
const timeEstimate = document.getElementById('time-estimate');
const resultSection = document.getElementById('result-section');
const resultSummary = document.getElementById('result-summary');
const svgPreview = document.getElementById('svg-preview');
const viewDetailsBtn = document.getElementById('view-details-btn');
const downloadSvgBtn = document.getElementById('download-svg-btn');
const historyList = document.getElementById('history-list');
const qualitySelect = document.getElementById('quality-select');
const qualityInfo = document.getElementById('quality-info');
const maxLayersInput = document.getElementById('max-layers');
const depthClustersSelect = document.getElementById('depth-clusters');
const backgroundMethodSelect = document.getElementById('background-method');

// Quality preset descriptions
const qualityDescriptions = {
    fast: 'Fast - 256 paths/layer, quick layer decomposition (~5 min)',
    balanced: 'Balanced - 512 paths/layer, better detail (~10 min)',
    'balanced+': 'Balanced+ - 512 paths/layer, better detail (~10 min)',
    high: 'High Quality - 768 paths/layer, fine details (~20 min)',
    best: 'Best Quality - 1024 paths/layer, maximum detail (~40 min)'
};

// Current result SVG path for download
let currentSvgPath = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    loadHistory();
});

// Setup event listeners
function setupEventListeners() {
    // Browse button
    browseBtn.addEventListener('click', () => fileInput.click());

    // File input change
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    dropZone.addEventListener('dragover', handleDragOver);
    dropZone.addEventListener('dragleave', handleDragLeave);
    dropZone.addEventListener('drop', handleDrop);

    // Remove button
    removeBtn.addEventListener('click', resetUpload);

    // Process button
    processBtn.addEventListener('click', startProcessing);

    // View details button
    viewDetailsBtn.addEventListener('click', () => {
        if (currentRunId) {
            window.location.href = `/layeredsvg/view/${currentRunId}`;
        }
    });

    // Download SVG button
    downloadSvgBtn.addEventListener('click', () => {
        if (currentSvgPath) {
            window.open(`/layeredsvg/results/${currentSvgPath}`, '_blank');
        }
    });

    // Quality selector change
    qualitySelect.addEventListener('change', (e) => {
        const quality = e.target.value;
        qualityInfo.textContent = qualityDescriptions[quality];
    });
}

// Handle drag over
function handleDragOver(e) {
    e.preventDefault();
    dropZone.classList.add('drag-over');
}

// Handle drag leave
function handleDragLeave(e) {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
}

// Handle drop
function handleDrop(e) {
    e.preventDefault();
    dropZone.classList.remove('drag-over');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

// Handle file select
function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

// Handle file
function handleFile(file) {
    // Validate file type
    if (!file.type.match('image/(png|jpeg|jpg)')) {
        alert('Please upload a PNG or JPG image.');
        return;
    }

    // Validate file size (16MB)
    if (file.size > 16 * 1024 * 1024) {
        alert('File size must be less than 16MB.');
        return;
    }

    currentFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        document.querySelector('.drop-zone-content').style.display = 'none';
        previewContainer.style.display = 'block';
        processSection.style.display = 'block';
    };
    reader.readAsDataURL(file);
}

// Reset upload
function resetUpload() {
    currentFile = null;
    fileInput.value = '';
    previewImage.src = '';
    document.querySelector('.drop-zone-content').style.display = 'flex';
    previewContainer.style.display = 'none';
    processSection.style.display = 'none';
    progressSection.style.display = 'none';
    resultSection.style.display = 'none';
}

// Start processing
async function startProcessing() {
    if (!currentFile) {
        alert('Please select a file first.');
        return;
    }

    // Upload file
    processBtn.disabled = true;
    processBtn.textContent = 'Uploading...';

    try {
        const formData = new FormData();
        formData.append('file', currentFile);
        formData.append('quality', qualitySelect.value);

        // Layer decomposition parameters
        formData.append('max_layers', maxLayersInput.value);
        formData.append('n_depth_clusters', depthClustersSelect.value);
        formData.append('background_method', backgroundMethodSelect.value);

        // Legacy depth params (kept for backward compatibility, now uses Depth Anything)
        formData.append('moge_version', 'v2');
        formData.append('moge_resolution', 'High');

        const uploadResponse = await fetch('/layeredsvg/upload', {
            method: 'POST',
            body: formData
        });

        if (!uploadResponse.ok) {
            throw new Error('Upload failed');
        }

        const uploadData = await uploadResponse.json();
        currentRunId = uploadData.job_id;

        // Hide upload section, show progress
        processSection.style.display = 'none';
        progressSection.style.display = 'block';

        // Start polling for status
        startStatusPolling();

    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred: ' + error.message);
        processBtn.disabled = false;
        processBtn.textContent = 'Start Layered Vectorization';
    }
}

// Start status polling
function startStatusPolling() {
    // Initial call
    checkStatus();

    // Poll every 2 seconds
    processingInterval = setInterval(checkStatus, 2000);
}

// Check status
async function checkStatus() {
    if (!currentRunId) return;

    try {
        const response = await fetch(`/job/${currentRunId}/status`);
        if (!response.ok) return;

        const status = await response.json();

        // Update progress
        updateProgress(status);

        // Check if completed or error
        if (status.status === 'completed') {
            clearInterval(processingInterval);
            showResults(status);
        } else if (status.status === 'error') {
            clearInterval(processingInterval);
            alert('Processing failed: ' + status.message);
            resetUpload();
        }

    } catch (error) {
        console.error('Status check error:', error);
    }
}

// Update progress
function updateProgress(status) {
    const progress = status.progress || 0;
    progressFill.style.width = progress + '%';
    progressPercentage.textContent = progress + '%';
    progressMessage.textContent = status.message || 'Processing...';

    // Show time estimate if available
    if (status.time_remaining && status.time_remaining > 0) {
        const minutes = Math.ceil(status.time_remaining / 60);
        if (minutes > 60) {
            const hours = Math.floor(minutes / 60);
            const remainingMins = minutes % 60;
            timeEstimate.textContent = `Estimated time remaining: ${hours}h ${remainingMins}m`;
        } else {
            timeEstimate.textContent = `Estimated time remaining: ${minutes} min`;
        }
    } else {
        timeEstimate.textContent = '';
    }
}

// Show results
function showResults(status) {
    progressSection.style.display = 'none';
    resultSection.style.display = 'block';

    // Show summary
    const nLayers = status.n_layers || 'multiple';
    resultSummary.innerHTML = `
        <p><strong>${nLayers} editable layers</strong> generated successfully!</p>
        <p>Each layer can be independently moved, scaled, or edited in vector editors like Inkscape, Illustrator, or Figma.</p>
    `;

    // Store SVG path for download
    currentSvgPath = status.result_svg;

    // Prefer PNG preview, fallback to SVG
    if (status.result_png) {
        const pngUrl = `${status.png_url}`;
        const imgElement = document.createElement('img');
        imgElement.src = pngUrl;
        imgElement.style.maxWidth = '100%';
        imgElement.style.maxHeight = '500px';
        imgElement.alt = 'Vectorization Result';
        svgPreview.innerHTML = '';
        svgPreview.appendChild(imgElement);
    } else if (status.result_svg) {
        const svgUrl = `${status.svg_url}`;
        const svgObject = document.createElement('object');
        svgObject.data = svgUrl;
        svgObject.type = 'image/svg+xml';
        svgObject.style.maxWidth = '100%';
        svgObject.style.maxHeight = '500px';
        svgPreview.innerHTML = '';
        svgPreview.appendChild(svgObject);
    }

    // Reload history
    loadHistory();
}

// Load history
async function loadHistory() {
    try {
        const response = await fetch('/layeredsvg/results');
        if (!response.ok) return;

        const results = await response.json();

        if (results.length === 0) {
            historyList.innerHTML = '<p class="loading">No conversions yet</p>';
            return;
        }

        historyList.innerHTML = '';
        results.forEach(result => {
            const item = createHistoryItem(result);
            historyList.appendChild(item);
        });

    } catch (error) {
        console.error('History load error:', error);
        historyList.innerHTML = '<p class="loading">Error loading history</p>';
    }
}

// Create history item
function createHistoryItem(result) {
    const div = document.createElement('div');
    div.className = 'history-item';

    const h4 = document.createElement('h4');
    h4.textContent = result.run_id;

    const p = document.createElement('p');
    if (result.has_result) {
        p.textContent = `${result.n_layers || '?'} layers`;
        p.className = 'completed';
    } else {
        p.textContent = 'Processing...';
        p.className = 'processing';
    }

    div.appendChild(h4);
    div.appendChild(p);

    if (result.has_result) {
        div.addEventListener('click', () => {
            window.location.href = `/layeredsvg/view/${result.run_id}`;
        });
        div.style.cursor = 'pointer';
    }

    return div;
}
