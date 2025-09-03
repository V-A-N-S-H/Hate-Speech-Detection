from flask import Flask, render_template, request, jsonify
import pickle
import re

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

# Global variable for the complete model pipeline
model_pipeline = None

def load_model():
    """Load the final trained model pipeline"""
    global model_pipeline
    
    try:
        # Load final model (complete pipeline)
        with open('hate_speech_model_final.pkl', 'rb') as f:
            model_pipeline = pickle.load(f)
        
        print("Final model pipeline loaded successfully!")
        return True
    except FileNotFoundError as e:
        print(f"Error loading final model: {e}")
        print("Please run train_final_model.py first to generate the final model.")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def clean_text_for_display(text):
    """Clean text and return the processed version for display"""
    text = str(text).lower()
    
    # Remove URLs and social media artifacts
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags completely
    text = re.sub(r'rt\s+', '', text)  # Remove retweet indicators
    
    # Remove HTML and special characters
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    
    # Handle punctuation more carefully
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove all punctuation
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
    text = re.sub(r'\b\w*\d+\w*\b', '', text)  # Remove words with numbers
    
    # Keep only meaningful words (don't remove stopwords completely)
    words = text.split()
    # Only remove very common articles and prepositions
    basic_stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being'}
    words = [word for word in words if word not in basic_stopwords and len(word) > 1]
    
    text = ' '.join(words).strip()
    return text

def predict_hate_speech(text):
    """Predict if text contains hate speech using the final model"""
    if not model_pipeline:
        return {"error": "Model not loaded"}
    
    try:
        # Get the processed text for display
        cleaned_text = clean_text_for_display(text)
        
        # The pipeline handles all preprocessing internally
        prediction = model_pipeline.predict([text])[0]
        
        # Get prediction probabilities
        try:
            probabilities = model_pipeline.predict_proba([text])[0]
            classes = model_pipeline.classes_
            prob_dict = {classes[i]: float(probabilities[i]) for i in range(len(classes))}
            
            # Add confidence level
            max_prob = max(probabilities)
            if max_prob > 0.8:
                confidence = "Very High"
            elif max_prob > 0.6:
                confidence = "High"
            elif max_prob > 0.4:
                confidence = "Medium"
            else:
                confidence = "Low"
            
            # Add interpretation
            if prediction == "Neither":
                interpretation = "✅ This text appears to be neutral/positive"
            elif prediction == "Offensive Language":
                interpretation = "⚠️ This text may contain offensive language"
            else:  # Hate Speech
                interpretation = "🚫 This text may contain hate speech"
                
        except Exception as e:
            prob_dict = None
            confidence = "Unknown"
            interpretation = f"Error getting probabilities: {e}"
        
        return {
            "prediction": prediction,
            "cleaned_text": cleaned_text,  # Add cleaned text for display
            "probabilities": prob_dict,
            "confidence": confidence,
            "interpretation": interpretation,
            "max_probability": float(max_prob) if 'max_prob' in locals() else None
        }
    except Exception as e:
        return {"error": str(e)}

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for hate speech prediction"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({"error": "Please provide text to analyze"}), 400
        
        if len(text) > 1000:
            return jsonify({"error": "Text is too long. Please limit to 1000 characters."}), 400
        
        result = predict_hate_speech(text)
        
        if "error" in result:
            return jsonify(result), 500
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model_pipeline is not None,
        "model_type": "final_optimized_pipeline"
    })

@app.route('/test-examples')
def test_examples():
    """Test endpoint with various examples"""
    test_texts = [
        # Positive examples
        "I love this beautiful day",
        "Thank you for your help",
        "This is a great project", 
        "You are amazing",
        "What a wonderful morning",
        "Good morning everyone",
        "Nice to meet you",
        "Hope you have a great time",
        
        # Potentially offensive examples
        "I hate you so much",
        "This is so stupid",
        "You are an idiot",
        
        # Neutral examples
        "The weather is okay today",
        "I went to the store",
        "Meeting at 3pm"
    ]
    
    results = []
    for text in test_texts:
        result = predict_hate_speech(text)
        results.append({
            "text": text,
            "result": result
        })
    
    return jsonify(results)

if __name__ == '__main__':
    print("Starting Final Hate Speech Detection App...")
    
    # Load final model on startup
    if not load_model():
        print("Failed to load final model. Please run train_final_model.py first.")
        exit(1)
    
    print("Final model loaded successfully!")
    print("The app now uses an improved model with better accuracy on positive sentences.")
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=8000)
