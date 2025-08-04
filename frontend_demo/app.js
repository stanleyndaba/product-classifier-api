/**
 * Smart Product Classifier - Frontend Demo
 * JavaScript functionality for the web interface
 */

// Configuration
const CONFIG = {
    API_BASE_URL: window.location.origin,
    MAX_FILE_SIZE: 10 * 1024 * 1024, // 10MB
    ALLOWED_TYPES: ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 'image/tiff'],
    PRODUCT_CATEGORIES: [
        'electronics', 'clothing', 'furniture', 'books', 'toys',
        'sports', 'beauty', 'home_garden', 'automotive', 'health',
        'food', 'jewelry'
    ]
};

// DOM Elements
const elements = {
    uploadArea: document.getElementById('uploadArea'),
    fileInput: document.getElementById('fileInput'),
    imageUrl: document.getElementById('imageUrl'),
    classifyUrlBtn: document.getElementById('classifyUrlBtn'),
    resultsSection: document.getElementById('resultsSection'),
    loadingSection: document.getElementById('loadingSection'),
    errorSection: document.getElementById('errorSection'),
    previewImage: document.getElementById('previewImage'),
    imageInfo: document.getElementById('imageInfo'),
    predictionsList: document.getElementById('predictionsList'),
    processingInfo: document.getElementById('processingInfo'),
    errorMessage: document.getElementById('errorMessage')
};

// State
let currentFile = null;

// Initialize the application
function initializeApp() {
    setupEventListeners();
    checkAPIHealth();
}

// Setup event listeners
function setupEventListeners() {
    // File input
    elements.fileInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop
    elements.uploadArea.addEventListener('dragover', handleDragOver);
    elements.uploadArea.addEventListener('dragleave', handleDragLeave);
    elements.uploadArea.addEventListener('drop', handleDrop);
    elements.uploadArea.addEventListener('click', () => elements.fileInput.click());
    
    // URL classification
    elements.classifyUrlBtn.addEventListener('click', handleUrlClassification);
    elements.imageUrl.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleUrlClassification();
    });
}

// Check API health
async function checkAPIHealth() {
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/health`);
        const data = await response.json();
        console.log('API Health:', data);
    } catch (error) {
        console.warn('API health check failed:', error);
        showError('API is not available. Please ensure the backend is running.');
    }
}

// Handle file selection
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        processFile(file);
    }
}

// Handle drag over
function handleDragOver(event) {
    event.preventDefault();
    elements.uploadArea.classList.add('drag-over');
}

// Handle drag leave
function handleDragLeave(event) {
    event.preventDefault();
    elements.uploadArea.classList.remove('drag-over');
}

// Handle drop
function handleDrop(event) {
    event.preventDefault();
    elements.uploadArea.classList.remove('drag-over');
    
    const files = event.dataTransfer.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
}

// Process selected file
function processFile(file) {
    // Validate file
    if (!validateFile(file)) {
        return;
    }
    
    currentFile = file;
    showImagePreview(file);
    classifyImage(file);
}

// Validate file
function validateFile(file) {
    // Check file type
    if (!CONFIG.ALLOWED_TYPES.includes(file.type)) {
        showError('Please select a valid image file (JPEG, PNG, BMP, or TIFF).');
        return false;
    }
    
    // Check file size
    if (file.size > CONFIG.MAX_FILE_SIZE) {
        showError('File size is too large. Please select an image under 10MB.');
        return false;
    }
    
    return true;
}

// Show image preview
function showImagePreview(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        elements.previewImage.src = e.target.result;
        elements.imageInfo.innerHTML = `
            <div class="info-item">
                <span class="label">Name:</span>
                <span class="value">${file.name}</span>
            </div>
            <div class="info-item">
                <span class="label">Size:</span>
                <span class="value">${formatFileSize(file.size)}</span>
            </div>
            <div class="info-item">
                <span class="label">Type:</span>
                <span class="value">${file.type}</span>
            </div>
        `;
        showSection('results');
    };
    reader.readAsDataURL(file);
}

// Handle URL classification
async function handleUrlClassification() {
    const url = elements.imageUrl.value.trim();
    if (!url) {
        showError('Please enter a valid image URL.');
        return;
    }
    
    if (!isValidUrl(url)) {
        showError('Please enter a valid URL.');
        return;
    }
    
    showSection('loading');
    
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/predict/url`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image_url: url,
                confidence_threshold: 0.1,
                top_k: 5
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        // Show image preview
        elements.previewImage.src = url;
        elements.imageInfo.innerHTML = `
            <div class="info-item">
                <span class="label">Source:</span>
                <span class="value">URL</span>
            </div>
            <div class="info-item">
                <span class="label">URL:</span>
                <span class="value">${url}</span>
            </div>
        `;
        
        displayResults(data);
        
    } catch (error) {
        console.error('URL classification error:', error);
        showError(`Failed to classify image from URL: ${error.message}`);
    }
}

// Classify image
async function classifyImage(file) {
    showSection('loading');
    
    try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('confidence_threshold', '0.1');
        formData.append('top_k', '5');
        
        const response = await fetch(`${CONFIG.API_BASE_URL}/predict`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        displayResults(data);
        
    } catch (error) {
        console.error('Classification error:', error);
        showError(`Failed to classify image: ${error.message}`);
    }
}

// Display classification results
function displayResults(data) {
    if (!data.success || !data.predictions || data.predictions.length === 0) {
        showError('No predictions were returned from the API.');
        return;
    }
    
    // Clear previous results
    elements.predictionsList.innerHTML = '';
    
    // Display predictions
    data.predictions.forEach((prediction, index) => {
        const predictionElement = createPredictionElement(prediction, index);
        elements.predictionsList.appendChild(predictionElement);
    });
    
    // Display processing info
    elements.processingInfo.innerHTML = `
        <div class="processing-stats">
            <div class="stat">
                <span class="label">Processing Time:</span>
                <span class="value">${data.processing_time_ms.toFixed(1)}ms</span>
            </div>
            <div class="stat">
                <span class="label">Model Version:</span>
                <span class="value">${data.model_version || '1.0.0'}</span>
            </div>
            <div class="stat">
                <span class="label">Predictions:</span>
                <span class="value">${data.predictions.length}</span>
            </div>
        </div>
    `;
    
    showSection('results');
}

// Create prediction element
function createPredictionElement(prediction, index) {
    const element = document.createElement('div');
    element.className = 'prediction-item';
    
    const confidence = Math.round(prediction.confidence * 100);
    const isTopPrediction = index === 0;
    
    element.innerHTML = `
        <div class="prediction-rank ${isTopPrediction ? 'top-prediction' : ''}">
            ${index + 1}
        </div>
        <div class="prediction-content">
            <div class="prediction-header">
                <span class="category-name">${formatCategoryName(prediction.class_name)}</span>
                <span class="confidence-score">${confidence}%</span>
            </div>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: ${confidence}%"></div>
            </div>
        </div>
        ${isTopPrediction ? '<div class="top-badge"><i class="fas fa-crown"></i></div>' : ''}
    `;
    
    return element;
}

// Format category name
function formatCategoryName(category) {
    return category
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

// Show specific section
function showSection(section) {
    // Hide all sections
    elements.resultsSection.style.display = 'none';
    elements.loadingSection.style.display = 'none';
    elements.errorSection.style.display = 'none';
    
    // Show requested section
    switch (section) {
        case 'results':
            elements.resultsSection.style.display = 'block';
            break;
        case 'loading':
            elements.loadingSection.style.display = 'block';
            break;
        case 'error':
            elements.errorSection.style.display = 'block';
            break;
    }
}

// Show error message
function showError(message) {
    elements.errorMessage.textContent = message;
    showSection('error');
}

// Reset application
function resetApp() {
    currentFile = null;
    elements.fileInput.value = '';
    elements.imageUrl.value = '';
    elements.uploadArea.classList.remove('drag-over');
    
    // Hide all sections
    elements.resultsSection.style.display = 'none';
    elements.loadingSection.style.display = 'none';
    elements.errorSection.style.display = 'none';
}

// Utility functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function isValidUrl(string) {
    try {
        new URL(string);
        return true;
    } catch (_) {
        return false;
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', initializeApp);

// Handle window resize for responsive layout
window.addEventListener('resize', () => {
    // Add any responsive adjustments here
});

// Service worker registration for PWA functionality (optional)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/sw.js')
            .then((registration) => {
                console.log('SW registered: ', registration);
            })
            .catch((registrationError) => {
                console.log('SW registration failed: ', registrationError);
            });
    });
}