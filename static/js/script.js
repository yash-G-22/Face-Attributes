// DOM Elements
const uploadBox = document.getElementById('uploadBox');
const imageInput = document.getElementById('imageInput');
const imagePreview = document.getElementById('imagePreview');
const previewImg = document.getElementById('previewImg');
const predictBtn = document.getElementById('predictBtn');
const clearBtn = document.getElementById('clearBtn');
const loading = document.getElementById('loading');
const results = document.getElementById('results');
const errorMsg = document.getElementById('errorMsg');
const successMsg = document.getElementById('successMsg');

// State
let selectedFile = null;
const API_URL = 'https://face-attributes.onrender.com/api/predict';

/**
 * Initialize event listeners
 */
function initializeEventListeners() {
    uploadBox.addEventListener('click', () => imageInput.click());
    uploadBox.addEventListener('dragover', handleDragOver);
    uploadBox.addEventListener('dragleave', handleDragLeave);
    uploadBox.addEventListener('drop', handleDrop);
    imageInput.addEventListener('change', handleImageSelect);
    predictBtn.addEventListener('click', handlePredict);
    clearBtn.addEventListener('click', handleClear);
}

/**
 * Handle drag over event
 */
function handleDragOver(e) {
    e.preventDefault();
    uploadBox.classList.add('dragover');
}

/**
 * Handle drag leave event
 */
function handleDragLeave() {
    uploadBox.classList.remove('dragover');
}

/**
 * Handle drop event
 */
function handleDrop(e) {
    e.preventDefault();
    uploadBox.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        imageInput.files = files;
        handleImageSelect();
    }
}

/**
 * Handle image selection
 */
function handleImageSelect() {
    const file = imageInput.files[0];
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith('image/')) {
        showError('Please select a valid image file');
        return;
    }

    // Validate file size (10MB)
    if (file.size > 10 * 1024 * 1024) {
        showError('File size must be less than 10MB');
        return;
    }

    selectedFile = file;
    const reader = new FileReader();
    
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        imagePreview.style.display = 'block';
        predictBtn.disabled = false;
        clearError();
    };

    reader.onerror = () => {
        showError('Error reading file');
    };

    reader.readAsDataURL(file);
}

/**
 * Handle predict button click
 */
async function handlePredict() {
    if (!selectedFile) {
        showError('Please select an image first');
        return;
    }

    const formData = new FormData();
    formData.append('image', selectedFile);

    loading.style.display = 'block';
    results.style.display = 'none';
    clearError();
    predictBtn.disabled = true;

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        console.log('API Response:', data);

        if (!response.ok || data.error) {
            showError(data.error || 'Prediction failed');
            return;
        }

        // Check if faces array exists and has results
        if (!data.faces || data.faces.length === 0) {
            showError('No faces detected in image');
            return;
        }

        // Display results for all detected faces
        displayAllResults(data.faces);
        showSuccess(`Prediction completed! Found ${data.faces.length} face(s).`);

    } catch (error) {
        console.error('Error:', error);
        showError('Network error: ' + error.message);
    } finally {
        loading.style.display = 'none';
        predictBtn.disabled = false;
    }
}

/**
 * Display prediction results for all faces
 */
function displayAllResults(faces) {
    // Clear previous results
    results.innerHTML = '';

    // Create a result card for each face
    faces.forEach((face, index) => {
        const faceCard = document.createElement('div');
        faceCard.className = 'face-card';
        
        const faceTitle = document.createElement('div');
        faceTitle.className = 'face-title';
        faceTitle.textContent = `Face ${index + 1}`;
        
        const resultsContainer = document.createElement('div');
        resultsContainer.className = 'face-results';

        // Gender result
        const genderItem = document.createElement('div');
        genderItem.className = 'result-item';
        genderItem.innerHTML = `
            <div class="result-label">👤 Gender</div>
            <div class="result-value">${face.gender.label}</div>
        `;
        resultsContainer.appendChild(genderItem);

        // Ethnicity result
        const ethnicityItem = document.createElement('div');
        ethnicityItem.className = 'result-item';
        ethnicityItem.innerHTML = `
            <div class="result-label">🌍 Ethnicity</div>
            <div class="result-value">${face.ethnicity.label}</div>
        `;
        resultsContainer.appendChild(ethnicityItem);

        // Age result
        const ageItem = document.createElement('div');
        ageItem.className = 'result-item';
        ageItem.innerHTML = `
            <div class="result-label">🎂 Age</div>
            <div class="result-value">${face.age.value} years</div>
        `;
        resultsContainer.appendChild(ageItem);

        faceCard.appendChild(faceTitle);
        faceCard.appendChild(resultsContainer);
        results.appendChild(faceCard);
    });

    // Show results section
    results.style.display = 'block';
    scrollToResults();
}

/**
 * Scroll to results section
 */
function scrollToResults() {
    results.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/**
 * Handle clear button click
 */
function handleClear() {
    imageInput.value = '';
    selectedFile = null;
    imagePreview.style.display = 'none';
    results.style.display = 'none';
    results.innerHTML = '';
    predictBtn.disabled = true;
    clearError();
}

/**
 * Show error message
 */
function showError(message) {
    errorMsg.textContent = message;
    errorMsg.style.display = 'block';
    successMsg.style.display = 'none';
    console.error('Error:', message);
}

/**
 * Show success message
 */
function showSuccess(message) {
    successMsg.textContent = message;
    successMsg.style.display = 'block';
    errorMsg.style.display = 'none';
}

/**
 * Clear all messages
 */
function clearError() {
    errorMsg.style.display = 'none';
    successMsg.style.display = 'none';
}

/**
 * Check API health on page load
 */
async function checkAPIHealth() {
    try {
        const response = await fetch('http://localhost:5000/api/health');
        if (!response.ok) {
            console.warn('API health check failed');
        }
    } catch (error) {
        console.warn('Cannot reach API server');
        showError('Warning: Backend server is not running. Please start the server with: python app.py');
    }
}

/**
 * Initialize app on DOM ready
 */
document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
    checkAPIHealth();
});
