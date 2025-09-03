import pickle
import re

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
    """Test prediction function that shows processed text"""
    try:
        # Load the final model
        with open('hate_speech_model_final.pkl', 'rb') as f:
            model_pipeline = pickle.load(f)
        
        # Get the processed text for display
        cleaned_text = clean_text_for_display(text)
        
        # The pipeline handles all preprocessing internally
        prediction = model_pipeline.predict([text])[0]
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
        
        return {
            "prediction": prediction,
            "cleaned_text": cleaned_text,
            "probabilities": prob_dict,
            "confidence": confidence,
            "max_probability": float(max_prob)
        }
    except Exception as e:
        return {"error": str(e)}

# Test with various examples
test_texts = [
    "I love this beautiful day! Thank you so much @friend #happiness https://example.com",
    "Have a great day everyone!",
    "This is a wonderful project, I'm so excited!",
    "Good morning, hope you have fun today"
]

print("Testing Final App - Processed Text Display")
print("=" * 60)

for text in test_texts:
    result = predict_hate_speech(text)
    
    if "error" not in result:
        print(f"Original: '{text}'")
        print(f"Processed: '{result['cleaned_text']}'")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']} ({result['max_probability']:.3f})")
        print("Probabilities:")
        for label, prob in result['probabilities'].items():
            print(f"  {label}: {prob:.3f}")
        print("-" * 50)
    else:
        print(f"Error with '{text}': {result['error']}")
        print("-" * 50)
