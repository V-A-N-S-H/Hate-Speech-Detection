import os
import re
import string
import nltk
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__, template_folder='templates', static_folder='static')

# -----------------------------------------------------------------------------
# 1. Initialization, NLP, & Model Loading
# -----------------------------------------------------------------------------
# Download NLTK resources quietly
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
stop_words_set = set(stopwords.words('english'))
stemmer = nltk.SnowballStemmer('english')

print("Loading ML model binaries...")
model = joblib.load('model.joblib')
vectorizer = joblib.load('vectorizer.joblib')
metadata = joblib.load('model_metadata.joblib')
print("ML binaries loaded successfully!")

def clean_text_for_inference(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\w*\d\w*", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = ' '.join(text.split())
    
    # Stem individual words and filter out stopwords
    words = [stemmer.stem(word) for word in text.split(' ') if word not in stop_words_set and len(word) > 0]
    return ' '.join(words)

# -----------------------------------------------------------------------------
# 2. Pre-computing Dataset Analytics (for Chart.js Frontend)
# -----------------------------------------------------------------------------
def precompute_dataset_analytics():
    print("Pre-computing corpus statistics...")
    df = pd.read_csv('Twitter.csv')
    label_map = {0: 'hate_speech', 1: 'offensive_language', 2: 'normal'}
    df['labels'] = df['class'].map(label_map)
    df = df.dropna(subset=['tweet'])
    
    total = len(df)
    hate_count = int((df['labels'] == 'hate_speech').sum())
    offensive_count = int((df['labels'] == 'offensive_language').sum())
    normal_count = int((df['labels'] == 'normal').sum())
    
    # Annotator correlation (excluding count)
    corr = df[['hate_speech', 'offensive_language', 'neither']].corr()
    consensus_matrix = corr.values.tolist()
    
    # Simulated Temporal trends
    np.random.seed(42)
    start_date = pd.to_datetime('2026-05-01')
    random_days = np.random.randint(0, 30, size=len(df))
    df['timestamp'] = start_date + pd.to_timedelta(random_days, unit='D')
    
    trend_counts = df.groupby(['timestamp', 'labels']).size().unstack(fill_value=0)
    timeline_dates = [d.strftime('%Y-%m-%d') for d in trend_counts.index]
    
    trends_data = {
        "dates": timeline_dates,
        "hate_speech": trend_counts['hate_speech'].tolist() if 'hate_speech' in trend_counts else [0]*30,
        "offensive_language": trend_counts['offensive_language'].tolist() if 'offensive_language' in trend_counts else [0]*30,
        "normal": trend_counts['normal'].tolist() if 'normal' in trend_counts else [0]*30
    }
    
    # High-frequency terms per category
    def get_top_words(label_name, num=15):
        sub_tweets = df[df['labels'] == label_name]['tweet'].tolist()
        text = " ".join(sub_tweets).lower()
        text = re.sub(r'rt\b|http\S+|@\w+|#\w+', '', text)
        words = re.findall(r'\b\w+\b', text)
        filtered = [w for w in words if w not in stop_words_set and len(w) > 2 and w not in {'amp', 'co', 'like', 'one', 'get', 'dont', 'go', 'think', 'u'}]
        from collections import Counter
        return Counter(filtered).most_common(num)
        
    vocab_data = {
        "hate_speech": [{"word": w, "count": c} for w, c in get_top_words('hate_speech')],
        "offensive_language": [{"word": w, "count": c} for w, c in get_top_words('offensive_language')],
        "normal": [{"word": w, "count": c} for w, c in get_top_words('normal')]
    }
    
    print("Pre-computation finished!")
    return {
        "total_tweets": total,
        "hate_pct": float(hate_count / total * 100),
        "offensive_pct": float(offensive_count / total * 100),
        "normal_pct": float(normal_count / total * 100),
        "hate_count": hate_count,
        "offensive_count": offensive_count,
        "normal_count": normal_count,
        "consensus_matrix": consensus_matrix,
        "trends": trends_data,
        "vocab": vocab_data,
        "model_accuracy": float(metadata['accuracy']),
        "classification_report": metadata['classification_report'],
        "confusion_matrix": metadata['confusion_matrix'],
        "classes": metadata['classes'],
        "top_features": metadata['top_features']
    }

cached_analytics = precompute_dataset_analytics()

# -----------------------------------------------------------------------------
# 3. Web & API Routes
# -----------------------------------------------------------------------------
@app.route('/')
def index():
    # Renders the single-page HTML client dashboard
    return render_template('index.html')

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    # Returns all cached dataset statistics and model coefficient records
    return jsonify(cached_analytics)

@app.route('/api/predict', methods=['POST'])
def predict():
    # Asynchronous REST endpoint for custom text diagnostics
    data = request.get_json(force=True)
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    input_text = data['text']
    if not input_text.strip():
        return jsonify({
            "predicted_class": "normal",
            "confidence": 1.0,
            "cleaned_text": "",
            "probabilities": {
                "hate_speech": 0.0,
                "offensive_language": 0.0,
                "normal": 1.0
            },
            "word_contributions": []
        })
        
    cleaned_text = clean_text_for_inference(input_text)
    vec = vectorizer.transform([cleaned_text])
    
    probs = model.predict_proba(vec)[0]
    pred_idx = np.argmax(probs)
    predicted_class = model.classes_[pred_idx]
    confidence = float(probs[pred_idx])
    
    prob_dict = {c_label: float(p) for c_label, p in zip(model.classes_, probs)}
    
    # Calculate SHAP/LIME attribution contributions
    class_idx = list(model.classes_).index(predicted_class)
    coefs = model.coef_[class_idx]
    vocab = vectorizer.vocabulary_
    tweet_tfidf = vec.toarray()[0]
    
    orig_words = input_text.split()
    word_contributions = []
    
    for w in orig_words:
        cleaned_word = w.lower().translate(str.maketrans("", "", string.punctuation))
        stemmed_word = stemmer.stem(cleaned_word)
        contribution = 0.0
        
        if stemmed_word in vocab:
            idx = vocab[stemmed_word]
            tfidf_score = tweet_tfidf[idx]
            coef_score = coefs[idx]
            contribution = float(tfidf_score * coef_score)
            
        word_contributions.append({
            "word": w,
            "contribution": contribution
        })
        
    return jsonify({
        "predicted_class": predicted_class,
        "confidence": confidence,
        "cleaned_text": cleaned_text,
        "probabilities": prob_dict,
        "word_contributions": word_contributions
    })

# -----------------------------------------------------------------------------
# 4. Main Runner
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # Launch on http://127.0.0.1:5000
    app.run(host="0.0.0.0", port=5000, debug=True)
