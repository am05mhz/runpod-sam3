// Global variables - SDXL Version (app3.py)
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
const resultSection = document.getElementById('result-section');
const svgPreview = document.getElementById('svg-preview');
const viewDetailsBtn = document.getElementById('view-details-btn');
const historyList = document.getElementById('history-list');
const qualitySelect = document.getElementById('quality-select');
const qualityInfo = document.getElementById('quality-info');

// Quality preset descriptions - SDXL version (1024x1024)
// Processing times are longer due to higher resolution
const qualityDescriptions = {
    fast: 'SDXL Fast - 256 paths, 80 colors (~5 min)',
    balanced: 'SDXL Balanced - 512 paths, 120 colors (~10 min)',
    high: 'SDXL High Quality - 768 paths, 150 colors (~20 min)',
    best: 'SDXL Best Quality - 1024 paths, 180 colors (~30 min)',
    ultra: 'SDXL Ultra-Detail - 2048 paths, 250 colors (~80 min)',
    extreme: 'SDXL Extreme-Detail - 2560 paths, 250 colors (~120 min, requires 20GB+ GPU)',
    max: 'SDXL Max-Detail - 4096 paths, 250 colors (~3 hours, requires 24GB+ GPU)'
};

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
            window.location.href = `/view/${currentRunId}`;
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

        const uploadResponse = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        if (!uploadResponse.ok) {
            throw new Error('Upload failed');
        }

        const uploadData = await uploadResponse.json();
        currentRunId = uploadData.run_id;

        // Hide upload section, show progress
        processSection.style.display = 'none';
        progressSection.style.display = 'block';

        // Start processing
        const processResponse = await fetch(`/process/${currentRunId}`, {
            method: 'POST'
        });

        if (!processResponse.ok) {
            throw new Error('Processing failed');
        }

        // Start polling for status
        startStatusPolling();

    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred: ' + error.message);
        processBtn.disabled = false;
        processBtn.textContent = 'Start Vectorization';
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
        const response = await fetch(`/status/${currentRunId}`);
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

    // Build message with time estimate if available
    let message = status.message || 'Processing...';

    if (status.time_remaining && status.time_remaining > 0) {
        const minutes = Math.ceil(status.time_remaining / 60);
        if (minutes > 60) {
            const hours = Math.floor(minutes / 60);
            const remainingMins = minutes % 60;
            message += ` (Est. ${hours}h ${remainingMins}m remaining)`;
        } else {
            message += ` (Est. ${minutes} min remaining)`;
        }
    }

    progressMessage.textContent = message;
}

// Show results
function showResults(status) {
    progressSection.style.display = 'none';
    resultSection.style.display = 'block';

    // Prefer PNG preview, fallback to full-size SVG, then regular SVG
    if (status.result_png) {
        const pngUrl = `/results/${status.result_png}`;
        const imgElement = document.createElement('img');
        imgElement.src = pngUrl;
        imgElement.style.maxWidth = '100%';
        imgElement.style.maxHeight = '500px';
        imgElement.alt = 'Vectorization Result';
        svgPreview.innerHTML = '';
        svgPreview.appendChild(imgElement);
    } else if (status.result_svg_fullsize) {
        // Fallback to full-size SVG if PNG not available
        const svgUrl = `/results/${status.result_svg_fullsize}`;
        const svgObject = document.createElement('object');
        svgObject.data = svgUrl;
        svgObject.type = 'image/svg+xml';
        svgObject.style.maxWidth = '100%';
        svgObject.style.maxHeight = '500px';
        svgPreview.innerHTML = '';
        svgPreview.appendChild(svgObject);
    } else if (status.result_svg) {
        // Final fallback to regular SVG
        const svgUrl = `/results/${status.result_svg}`;
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
        const response = await fetch('/results');
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
    p.textContent = result.has_result ? '✓ Completed' : '⏳ Processing...';

    div.appendChild(h4);
    div.appendChild(p);

    if (result.has_result) {
        div.addEventListener('click', () => {
            window.location.href = `/view/${result.run_id}`;
        });
    }

    return div;
}

// Format date/time
function formatDateTime(dateStr) {
    const year = dateStr.substring(0, 4);
    const month = dateStr.substring(4, 6);
    const day = dateStr.substring(6, 8);
    const hour = dateStr.substring(9, 11);
    const minute = dateStr.substring(11, 13);
    const second = dateStr.substring(13, 15);

    return `${year}-${month}-${day} ${hour}:${minute}:${second}`;
}
