// DOM Elements
const textInput = document.getElementById('textInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const clearBtn = document.getElementById('clearBtn');
const charCount = document.getElementById('charCount');
const resultsSection = document.getElementById('resultsSection');
const loadingSection = document.getElementById('loadingSection');
const errorSection = document.getElementById('errorSection');
const predictionBadge = document.getElementById('predictionBadge');
const predictionDescription = document.getElementById('predictionDescription');
const processedText = document.getElementById('processedText');
const probabilitiesCard = document.getElementById('probabilitiesCard');
const probabilityBars = document.getElementById('probabilityBars');
const errorMessage = document.getElementById('errorMessage');

// Event listeners
document.addEventListener('DOMContentLoaded', function() {
    // Character counter
    textInput.addEventListener('input', updateCharCounter);
    
    // Button events
    analyzeBtn.addEventListener('click', analyzeText);
    clearBtn.addEventListener('click', clearText);
    
    // Enter key to analyze (Ctrl+Enter)
    textInput.addEventListener('keydown', function(e) {
        if (e.ctrlKey && e.key === 'Enter') {
            e.preventDefault();
            analyzeText();
        }
    });
    
    // Initial setup
    updateCharCounter();
});

function updateCharCounter() {
    const length = textInput.value.length;
    charCount.textContent = length;
    
    // Change color based on character count
    const counter = charCount.parentElement;
    if (length > 900) {
        counter.style.color = '#e53e3e';
    } else if (length > 700) {
        counter.style.color = '#d69e2e';
    } else {
        counter.style.color = '#718096';
    }
    
    // Enable/disable analyze button
    analyzeBtn.disabled = length === 0;
}

function clearText() {
    textInput.value = '';
    updateCharCounter();
    hideAllSections();
    textInput.focus();
}

function hideAllSections() {
    resultsSection.style.display = 'none';
    loadingSection.style.display = 'none';
    errorSection.style.display = 'none';
}

function showLoading() {
    hideAllSections();
    loadingSection.style.display = 'block';
    analyzeBtn.disabled = true;
}

function hideLoading() {
    loadingSection.style.display = 'none';
    analyzeBtn.disabled = false;
}

function showError(message) {
    hideAllSections();
    errorMessage.textContent = message;
    errorSection.style.display = 'block';
}

function showResults(data) {
    hideAllSections();
    
    // Update prediction badge and description
    const prediction = data.prediction;
    const badgeClass = prediction.toLowerCase().replace(' ', '-');
    
    predictionBadge.textContent = prediction;
    predictionBadge.className = `prediction-badge ${badgeClass}`;
    
    // Use interpretation if available, otherwise use default descriptions
    let description = data.interpretation || '';
    if (!description) {
        const descriptions = {
            'Hate Speech': 'This text has been classified as containing hate speech. It may include content that attacks or discriminates against individuals or groups.',
            'Offensive Language': 'This text contains offensive language that may be inappropriate or harmful, but does not necessarily target specific groups.',
            'Neither': 'This text appears to be neutral and does not contain hate speech or offensive language.'
        };
        description = descriptions[prediction] || 'Classification complete.';
    }
    
    // Add confidence information to description
    if (data.confidence) {
        description += ` (Confidence: ${data.confidence})`;
    }
    
    predictionDescription.innerHTML = description;
    
    // Show processed text with better formatting
    if (data.cleaned_text) {
        processedText.innerHTML = `"${data.cleaned_text}"`;
    } else {
        processedText.textContent = 'No processed text available';
    }
    
    // Show probabilities if available
    if (data.probabilities) {
        displayProbabilities(data.probabilities);
        probabilitiesCard.style.display = 'block';
    } else {
        probabilitiesCard.style.display = 'none';
    }
    
    // Show results section with animation
    resultsSection.style.display = 'block';
    resultsSection.classList.add('fade-in');
    
    // Scroll to results
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 300);
}

function displayProbabilities(probabilities) {
    probabilityBars.innerHTML = '';
    
    // Sort probabilities by value (highest first)
    const sortedProbs = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
    
    sortedProbs.forEach(([label, prob]) => {
        const percentage = Math.round(prob * 100);
        const barClass = label.toLowerCase().replace(' ', '-');
        
        const barElement = document.createElement('div');
        barElement.className = 'probability-bar';
        barElement.innerHTML = `
            <div class="probability-label">${label}</div>
            <div class="probability-progress">
                <div class="probability-fill ${barClass}" style="width: ${percentage}%"></div>
            </div>
            <div class="probability-value">${percentage}%</div>
        `;
        
        probabilityBars.appendChild(barElement);
    });
}

async function analyzeText() {
    const text = textInput.value.trim();
    
    if (!text) {
        showError('Please enter some text to analyze.');
        return;
    }
    
    if (text.length > 1000) {
        showError('Text is too long. Please limit to 1000 characters.');
        return;
    }
    
    showLoading();
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'An error occurred during analysis');
        }
        
        hideLoading();
        showResults(data);
        
    } catch (error) {
        hideLoading();
        showError(`Analysis failed: ${error.message}`);
        console.error('Error:', error);
    }
}

// Sample texts for testing (optional - can be removed)
function loadSampleText(type) {
    const samples = {
        hate: "I hate all people from that country, they should all be eliminated",
        offensive: "You're such an idiot, shut up and go away",
        neutral: "I really enjoy spending time with my family and friends"
    };
    
    if (samples[type]) {
        textInput.value = samples[type];
        updateCharCounter();
    }
}

// Add keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Escape key to clear
    if (e.key === 'Escape') {
        clearText();
    }
});

// Add smooth scrolling behavior
document.documentElement.style.scrollBehavior = 'smooth';
