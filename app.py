from flask import Flask, request, jsonify, render_template # used for making the web application
from flask_cors import CORS # used for requesting data from another domain
import pickle # used for saving and loading models
import re # used for text cleaning
import string # for string manipulation
import pandas as pd # used for data cleaning and manipulation
import os # used for accessing files

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables to store model and vectorizer
model = None
vectorizer = None

def preprocess_text(text):
    """Clean and preprocess text data - same as in training script"""
    if pd.isna(text) or not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs, mentions, and hashtags
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove punctuation and special characters
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def load_model():
    """Load the trained model and vectorizer"""
    global model, vectorizer
    
    try:
        with open('hate_speech_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        print("Model and vectorizer loaded successfully!")
        return True
    except FileNotFoundError:
        print("Model files not found. Please train the model first by running train_model.py")
        return False
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

def predict_hate_speech(text):
    """Predict if text contains hate speech"""
    if not model or not vectorizer:
        return None, None
    
    # Preprocess the text
    cleaned_text = preprocess_text(text)
    
    if not cleaned_text:
        return "normal", 0.5
    
    # Vectorize the text
    vectorized_text = vectorizer.transform([cleaned_text])
    
    # Make prediction
    prediction = model.predict(vectorized_text)[0]
    probabilities = model.predict_proba(vectorized_text)[0]
    
    # Get the result and confidence
    result = "hate/offensive" if prediction == 1 else "normal"
    confidence = max(probabilities)
    
    return result, confidence

@app.route('/')
def home():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for hate speech prediction"""
    try:
        # Get the text from request
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'error': 'No text provided',
                'status': 'error'
            }), 400
        
        text = data['text'].strip()
        
        if not text:
            return jsonify({
                'error': 'Empty text provided',
                'status': 'error'
            }), 400
        
        # Make prediction
        prediction, confidence = predict_hate_speech(text)
        
        if prediction is None:
            return jsonify({
                'error': 'Model not loaded',
                'status': 'error'
            }), 500
        
        # Return the result
        return jsonify({
            'text': text,
            'prediction': prediction,
            'confidence': float(confidence),
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    model_loaded = model is not None and vectorizer is not None
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded
    })

@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """API endpoint for batch prediction"""
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({
                'error': 'No texts provided',
                'status': 'error'
            }), 400
        
        texts = data['texts']
        
        if not isinstance(texts, list):
            return jsonify({
                'error': 'Texts must be a list',
                'status': 'error'
            }), 400
        
        results = []
        
        for text in texts:
            if isinstance(text, str) and text.strip():
                prediction, confidence = predict_hate_speech(text.strip())
                results.append({
                    'text': text.strip(),
                    'prediction': prediction,
                    'confidence': float(confidence) if confidence else None
                })
            else:
                results.append({
                    'text': str(text),
                    'prediction': 'error',
                    'confidence': None
                })
        
        return jsonify({
            'results': results,
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

if __name__ == '__main__':
    print("Starting Hate Speech Detection API...")
    
    # Load the model
    if load_model():
        print("Model loaded successfully!")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model. Please run train_model.py first to train the model.")
        print("Starting server anyway (model endpoints will return errors)...")
        app.run(debug=True, host='0.0.0.0', port=5000)