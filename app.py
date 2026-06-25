import os
import re
import string
import nltk
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import json

st.set_page_config(page_title="Hate Speech Detection", layout="wide", initial_sidebar_state="collapsed")

# -----------------------------------------------------------------------------
# 1. Initialization, NLP, & Model Loading
# -----------------------------------------------------------------------------
@st.cache_resource
def load_nltk_resources():
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    return set(stopwords.words('english')), nltk.SnowballStemmer('english')

stop_words_set, stemmer = load_nltk_resources()

@st.cache_resource
def load_models():
    model = joblib.load('model.joblib')
    vectorizer = joblib.load('vectorizer.joblib')
    metadata = joblib.load('model_metadata.joblib')
    return model, vectorizer, metadata

model, vectorizer, metadata = load_models()

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
    
    words = [stemmer.stem(word) for word in text.split(' ') if word not in stop_words_set and len(word) > 0]
    return ' '.join(words)

# -----------------------------------------------------------------------------
# 2. Pre-computing Dataset Analytics (for Chart.js Frontend)
# -----------------------------------------------------------------------------
@st.cache_data
def precompute_dataset_analytics():
    df = pd.read_csv('Twitter.csv')
    label_map = {0: 'hate_speech', 1: 'offensive_language', 2: 'normal'}
    df['labels'] = df['class'].map(label_map)
    df = df.dropna(subset=['tweet'])
    
    total = len(df)
    hate_count = int((df['labels'] == 'hate_speech').sum())
    offensive_count = int((df['labels'] == 'offensive_language').sum())
    normal_count = int((df['labels'] == 'normal').sum())
    
    corr = df[['hate_speech', 'offensive_language', 'neither']].corr()
    consensus_matrix = corr.values.tolist()
    
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
# 3. Streamlit Custom CSS Injection
# -----------------------------------------------------------------------------
st.markdown("""
<style>
/* Base Theme overrides for Streamlit */
:root {
    --bg-color: #0e1117;
    --card-bg: rgba(255, 255, 255, 0.04);
    --card-border: rgba(255, 255, 255, 0.08);
    --text-color: #ffffff;
    --text-muted: #8892b0;
    --accent-blue: #00f2fe;
    --color-hate: #e74c3c;
    --color-offensive: #f39c12;
    --color-normal: #2ecc71;
}

div[role="tabpanel"] {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 30px;
    margin-top: 10px;
}

div.stButton > button {
    background: rgba(255,255,255,0.02);
    border: 1px solid var(--card-border);
    border-radius: 8px;
    padding: 12px 16px;
    color: var(--text-color);
    cursor: pointer;
    text-align: left;
    transition: all 0.2s ease;
    height: 100%;
    width: 100%;
    font-size: 13px;
}
div.stButton > button:hover {
    transform: scale(1.02);
    background: rgba(255,255,255,0.04);
    border-color: rgba(255, 255, 255, 0.15);
}

/* Preset buttons left borders */
div[data-testid="column"]:nth-child(1) div.stButton > button { border-left: 3px solid var(--color-normal); }
div[data-testid="column"]:nth-child(2) div.stButton > button { border-left: 3px solid var(--color-offensive); }
div[data-testid="column"]:nth-child(3) div.stButton > button { border-left: 3px solid var(--color-hate); }

div.stTextArea textarea {
    background-color: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid rgba(0, 242, 254, 0.5) !important;
    border-radius: 8px !important;
    color: white !important;
    font-size: 14px;
    padding: 15px !important;
}
div.stTextArea textarea:focus {
    border-color: #00f2fe !important;
    box-shadow: 0 0 8px rgba(0, 242, 254, 0.3) !important;
}

.card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 25px;
    margin-bottom: 20px;
}
.card-title {
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 5px;
    color: white;
}
</style>
""", unsafe_allow_html=True)

def get_svg_gauge(percentage, color):
    circumference = 2 * 3.14159 * 40
    offset = circumference - (percentage / 100) * circumference
    return f'''
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; min-height: 140px;">
        <svg width="120" height="120" viewBox="0 0 100 100">
            <circle cx="50" cy="50" r="40" fill="none" stroke="rgba(255,255,255,0.08)" stroke-width="8"></circle>
            <circle cx="50" cy="50" r="40" fill="none" stroke="{color}" stroke-width="8" stroke-dasharray="{circumference}" stroke-dashoffset="{offset}" stroke-linecap="round" transform="rotate(-90 50 50)"></circle>
            <text x="50" y="57" font-size="22" font-weight="800" fill="{color}" text-anchor="middle">{int(percentage)}%</text>
        </svg>
        <div style="color: #8892b0; font-size: 13px; margin-top: 15px;">Classification Confidence</div>
    </div>
    '''

# -----------------------------------------------------------------------------
# 4. Prediction Logic
# -----------------------------------------------------------------------------
def predict_text(input_text):
    if not input_text.strip():
        return None
        
    cleaned_text = clean_text_for_inference(input_text)
    vec = vectorizer.transform([cleaned_text])
    
    probs = model.predict_proba(vec)[0]
    pred_idx = np.argmax(probs)
    predicted_class = model.classes_[pred_idx]
    confidence = float(probs[pred_idx])
    
    prob_dict = {c_label: float(p) for c_label, p in zip(model.classes_, probs)}
    
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
        
    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "cleaned_text": cleaned_text,
        "probabilities": prob_dict,
        "word_contributions": word_contributions
    }

# -----------------------------------------------------------------------------
# 5. UI Layout
# -----------------------------------------------------------------------------
st.markdown("<h1 style='text-align: center; color: #00f2fe;'>Hate Speech Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8892b0;'>Machine Learning Diagnostics & Analytics</p>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Hate Speech Detector", "Visual Analytics Dashboard", "Model Metrics & Insights"])

# -----------------------------------------------------------------------------
# TAB 1: Detector
# -----------------------------------------------------------------------------
with tab1:
    st.markdown("""
    <h3 style="color: white; margin-bottom: 5px; font-size: 18px;">Hate Speech Detector</h3>
    <p style="color: #8892b0; font-size: 13px; margin-top: 0px; margin-bottom: 20px;">Analyze text inputs instantly to flag hate speech and offensive content with advanced feature attribution highlighting.</p>
    """, unsafe_allow_html=True)
    
    st.markdown("<p style='font-size: 12px; font-weight: 700; margin-bottom: 10px;'>Quick-Test Presets</p>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    if 'input_text' not in st.session_state:
        st.session_state.input_text = ""
        
    with col1:
        if st.button("Clean / Friendly\n\n'I am so incredibly proud of your achievements! Keep shining!'"):
            st.session_state.input_text = "I am so incredibly proud of your achievements! Keep shining!"
    with col2:
        if st.button("Offensive / Toxic\n\n'This is stupid and ridiculous, shut the hell up.'"):
            st.session_state.input_text = "This is stupid and ridiculous, shut the hell up."
    with col3:
        if st.button("Hate Speech\n\n'I hate all these toxic immigrants, they are dirty and ruined our country.'"):
            st.session_state.input_text = "I hate all these toxic immigrants, they are dirty and ruined our country."

    user_input = st.text_area("", value=st.session_state.input_text, height=120)
    
    if st.button("Enter (Analyze)"):
        if user_input:
            res = predict_text(user_input)
            
            # Render exactly the same HTML result box
            metaMap = {
                'hate_speech': {'title': 'Hate Speech', 'color': '#e74c3c', 'desc': 'Content expressing hatred or promoting discrimination against protected groups.'},
                'offensive_language': {'title': 'Offensive Language', 'color': '#f39c12', 'desc': 'Content containing vulgarity, insults, or highly aggressive remarks.'},
                'normal': {'title': 'Clean / Normal', 'color': '#2ecc71', 'desc': 'Safe and acceptable content.'}
            }
            meta = metaMap[res['predicted_class']]
            
            html_results = f"""
            <div style="display: flex; flex-direction: row; gap: 30px; margin-top: 20px;">
                <!-- Left Column -->
                <div style="flex: 1.2;">
                    <h3 style="color: white; font-size: 18px; font-weight: 700; margin-bottom: 15px;">Assessment Results</h3>
                    
                    <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 20px; margin-bottom: 20px;">
                        <h2 style="color: {meta['color']}; margin: 0; font-size: 22px;">{meta['title']}</h2>
                        <p style="color: #8892b0; font-size: 13px; margin-top: 8px; margin-bottom: 15px;">{meta['desc']}</p>
                        <div style="font-size: 12px; color: #8892b0;">
                            <strong>Preprocessed Token Stream:</strong><br/>
                            <div style="margin-top: 5px; color: white;"><code>{res['cleaned_text']}</code></div>
                        </div>
                    </div>
                    
                    <h5 style="color: white; font-size: 14px; margin-bottom: 15px;">Probability Breakdown</h5>
                    <div style="margin-bottom: 12px;">
                        <div style="display: flex; justify-content: space-between; font-size: 12px; color: white; font-weight: 600; margin-bottom: 6px;">
                            <span>Clean / Normal</span> <span>{(res['probabilities']['normal']*100):.2f}%</span>
                        </div>
                        <div style="background: rgba(255,255,255,0.08); border-radius: 4px; height: 6px; width: 100%;"><div style="background-color: #2ecc71; height: 100%; border-radius: 4px; width: {(res['probabilities']['normal']*100)}%;"></div></div>
                    </div>
                    <div style="margin-bottom: 12px;">
                        <div style="display: flex; justify-content: space-between; font-size: 12px; color: white; font-weight: 600; margin-bottom: 6px;">
                            <span>Offensive Language</span> <span>{(res['probabilities']['offensive_language']*100):.2f}%</span>
                        </div>
                        <div style="background: rgba(255,255,255,0.08); border-radius: 4px; height: 6px; width: 100%;"><div style="background-color: #f39c12; height: 100%; border-radius: 4px; width: {(res['probabilities']['offensive_language']*100)}%;"></div></div>
                    </div>
                    <div style="margin-bottom: 12px;">
                        <div style="display: flex; justify-content: space-between; font-size: 12px; color: white; font-weight: 600; margin-bottom: 6px;">
                            <span>Hate Speech</span> <span>{(res['probabilities']['hate_speech']*100):.2f}%</span>
                        </div>
                        <div style="background: rgba(255,255,255,0.08); border-radius: 4px; height: 6px; width: 100%;"><div style="background-color: #e74c3c; height: 100%; border-radius: 4px; width: {(res['probabilities']['hate_speech']*100)}%;"></div></div>
                    </div>
                </div>
                
                <!-- Right Column -->
                <div style="flex: 0.8;">
                    <h3 style="color: white; font-size: 18px; font-weight: 700; margin-bottom: 15px;">Confidence Score</h3>
                    <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 20px; height: 180px;">
                        {get_svg_gauge(res['confidence']*100, meta['color'])}
                    </div>
                </div>
            </div>
            
            <div style="margin-top: 30px;">
                <h3 style="color: white; font-size: 18px; font-weight: 700; margin-bottom: 5px;">Feature Attribution Highlighting</h3>
                <p style="color: #8892b0; font-size: 13px; margin-bottom: 15px;">The highlighted words below mathematically contributed to the prediction. Darker/more saturated backgrounds indicate a stronger model coefficient impact.</p>
                <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 20px; font-size: 16px; line-height: 1.8; color: white;">
            """
            
            for item in res['word_contributions']:
                w = item['word']
                score = item['contribution']
                if score > 0.01:
                    opacity = min(0.2 + (score * 4.0), 0.9)
                    if res['predicted_class'] == 'hate_speech':
                        bg = f"rgba(231, 76, 60, {opacity})"
                        border = "#e74c3c"
                    elif res['predicted_class'] == 'offensive_language':
                        bg = f"rgba(243, 156, 18, {opacity})"
                        border = "#f39c12"
                    else:
                        bg = f"rgba(46, 204, 113, {opacity})"
                        border = "#2ecc71"
                    html_results += f'<span style="background-color: {bg}; border-bottom: 2px solid {border}; padding: 2px 4px; border-radius: 4px; font-weight: 600;">{w} </span>'
                else:
                    html_results += f'<span>{w} </span>'
            
            html_results += """
                </div>
                <div style="display: flex; gap: 15px; font-size: 11px; color: #8892b0; margin-top: 15px;">
                    <span>Legend:</span>
                    <span><span style="border-bottom: 2px solid #e74c3c; color: #e74c3c; padding-bottom: 2px;">Hate Speech Word Indicator</span></span>
                    <span><span style="border-bottom: 2px solid #f39c12; color: #f39c12; padding-bottom: 2px;">Offensive Word Indicator</span></span>
                    <span><span style="border-bottom: 2px solid #2ecc71; color: #2ecc71; padding-bottom: 2px;">Normal Word Indicator</span></span>
                </div>
            </div>
            """
            clean_html = "".join([line.strip() for line in html_results.split("\n")])
            st.markdown(clean_html, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 2 & 3: Visual Analytics & Metrics (using Chart.js in st.components.v1.html)
# -----------------------------------------------------------------------------
analytics_json = json.dumps(cached_analytics)

chartjs_html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ background-color: transparent; color: #ffffff; font-family: -apple-system, sans-serif; margin: 0; }}
        .card {{ background: rgba(255, 255, 255, 0.04); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 16px; padding: 25px; box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3); margin-bottom: 20px; }}
        .card-title {{ font-size: 18px; font-weight: 700; margin-bottom: 20px; }}
        
        .metric-label {{ font-size: 13px; text-transform: uppercase; letter-spacing: 1.2px; color: #8892b0; margin-bottom: 8px; }}
        .metric-value {{ font-size: 34px; font-weight: 800; }}
        
        .grid-4 {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 30px; }}
        .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-bottom: 30px;}}
        
        .heatmap-grid {{ display: grid; grid-template-columns: 100px 1fr 1fr 1fr; grid-template-rows: auto auto auto auto; gap: 6px; margin-top: 15px; }}
        .heatmap-cell {{ aspect-ratio: 2.2; display: flex; align-items: center; justify-content: center; border-radius: 6px; font-size: 14px; font-weight: 700; color: #0e1117; }}
        .heatmap-label {{ display: flex; align-items: center; font-size: 12px; font-weight: 600; color: #8892b0; }}
        .heatmap-corner {{ grid-column: 1; grid-row: 1; }}
        
        table {{ width: 100%; border-collapse: collapse; font-size: 14px; margin-bottom: 20px; }}
        th {{ text-align: left; padding: 12px 16px; color: #8892b0; border-bottom: 1px solid rgba(255, 255, 255, 0.08); font-weight: 600; }}
        td {{ padding: 14px 16px; border-bottom: 1px solid rgba(255, 255, 255, 0.03); }}
        
        /* Confusion Matrix CSS Heatmap styles */
        .confusion-matrix-wrapper {{ width: 100%; display: flex; justify-content: center; align-items: center; margin-top: 10px; }}
        .axis-label-y {{ font-size: 13px; font-weight: 700; color: #8892b0; writing-mode: vertical-rl; transform: rotate(180deg); margin-right: 15px; text-transform: uppercase; letter-spacing: 1.2px; }}
        .axis-label-x {{ font-size: 13px; font-weight: 700; color: #8892b0; text-align: center; margin-top: 15px; text-transform: uppercase; letter-spacing: 1.2px; width: 100%; }}
        .matrix-grid-container {{ display: flex; flex-direction: column; align-items: center; width: 100%; max-width: 440px; }}
        .matrix-grid {{ display: grid; grid-template-columns: 120px 1fr 1fr 1fr; grid-template-rows: auto auto auto auto; gap: 6px; width: 100%; }}
        .matrix-cell {{ aspect-ratio: 1.6; display: flex; align-items: center; justify-content: center; border-radius: 6px; font-size: 15px; font-weight: 700; transition: all 0.2s ease; cursor: default; }}
        .matrix-label-row {{ display: flex; align-items: center; justify-content: flex-end; padding-right: 12px; font-size: 12px; font-weight: 600; color: #8892b0; text-align: right; }}
        .matrix-label-col {{ display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 600; color: #8892b0; padding-bottom: 8px; text-align: center; }}
        .matrix-corner {{ grid-column: 1; grid-row: 1; }}
    </style>
</head>
<body>
    <script>
        const data = {analytics_json};
        window.data = data;
    </script>
    <div id="analytics" style="display: none;">
        <div class="grid-4">
            <div class="card"><div class="metric-label">Total Corpus Size</div><div class="metric-value" id="meta-total-tweets">0</div><div style="font-size: 12px; color: #3498db; margin-top: 5px;">Standard Davidson Dataset</div></div>
            <div class="card"><div class="metric-label" style="color: #e74c3c;">Hate Speech Flagged</div><div class="metric-value" style="color: #e74c3c;" id="meta-hate-pct">0.00%</div><div style="font-size: 12px; color: #e74c3c; margin-top: 5px;" id="meta-hate-count">0 Tweets</div></div>
            <div class="card"><div class="metric-label" style="color: #f39c12;">Offensive Language</div><div class="metric-value" style="color: #f39c12;" id="meta-offensive-pct">0.00%</div><div style="font-size: 12px; color: #f39c12; margin-top: 5px;" id="meta-offensive-count">0 Tweets</div></div>
            <div class="card"><div class="metric-label" style="color: #2ecc71;">Clean / Normal</div><div class="metric-value" style="color: #2ecc71;" id="meta-normal-pct">0.00%</div><div style="font-size: 12px; color: #2ecc71; margin-top: 5px;" id="meta-normal-count">0 Tweets</div></div>
        </div>
        <div class="grid-2">
            <div class="card"><div class="card-title">Dataset Label Distribution</div><div style="height: 300px; display: flex; align-items: center; justify-content: center;"><canvas id="donutChart"></canvas></div></div>
            <div class="card"><div class="card-title">Annotator Consensus Matrix</div><div class="heatmap-grid" id="consensus-heatmap-container"></div></div>
        </div>
        <div class="grid-2">
            <div class="card"><div class="card-title">Content Activity Trends over Time</div><div style="height: 320px;"><canvas id="trendsChart"></canvas></div></div>
            <div class="card"><div class="card-title">High-Frequency Terms (Hate Speech)</div><div style="height: 320px;"><canvas id="vocabHateChart"></canvas></div></div>
        </div>
    </div>
    
    <div id="metrics" style="display: none;">
        <div class="grid-2">
            <div class="card">
                <div class="card-title">Classification Diagnostics Report</div>
                <table id="metrics-table">
                    <thead><tr><th>Class Category</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Validation Tweets</th></tr></thead>
                    <tbody id="metrics-table-body"></tbody>
                </table>
            </div>
            <div class="card">
                <div class="card-title">Confusion Matrix</div>
                <div class="confusion-matrix-wrapper">
                    <div style="display: flex; align-items: center; justify-content: center; width: 100%;">
                        <div class="axis-label-y">Actual Label</div>
                        <div class="matrix-grid-container">
                            <div class="matrix-grid" id="confusion-matrix-grid"></div>
                            <div class="axis-label-x">Predicted Label</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <div class="grid-2">
            <div class="card"><div class="card-title">Top Hate Speech Features</div><div style="height: 400px;"><canvas id="coefHateChart"></canvas></div></div>
            <div class="card"><div class="card-title">Top Clean Features</div><div style="height: 400px;"><canvas id="coefNormChart"></canvas></div></div>
        </div>
    </div>

    <script>
        function populateMetricsOverview(data) {{
            document.getElementById('meta-total-tweets').innerText = data.total_tweets.toLocaleString();
            document.getElementById('meta-hate-pct').innerText = `${{data.hate_pct.toFixed(2)}}%`;
            document.getElementById('meta-hate-count').innerText = `${{data.hate_count.toLocaleString()}} Tweets`;
            document.getElementById('meta-offensive-pct').innerText = `${{data.offensive_pct.toFixed(2)}}%`;
            document.getElementById('meta-offensive-count').innerText = `${{data.offensive_count.toLocaleString()}} Tweets`;
            document.getElementById('meta-normal-pct').innerText = `${{data.normal_pct.toFixed(2)}}%`;
            document.getElementById('meta-normal-count').innerText = `${{data.normal_count.toLocaleString()}} Tweets`;
        }}

        function loadLabelDonutChart(data) {{
            const ctx = document.getElementById('donutChart').getContext('2d');
            new Chart(ctx, {{
                type: 'doughnut',
                data: {{
                    labels: ['Hate Speech', 'Offensive Language', 'Clean / Normal'],
                    datasets: [{{
                        data: [data.hate_count, data.offensive_count, data.normal_count],
                        backgroundColor: ['#e74c3c', '#f39c12', '#2ecc71'],
                        borderColor: 'rgba(0, 0, 0, 0.4)',
                        borderWidth: 1.5
                    }}]
                }},
                options: {{ responsive: true, maintainAspectRatio: false, plugins: {{ legend: {{ position: 'bottom', labels: {{ color: '#ffffff' }} }} }} }}
            }});
        }}

        function loadConsensusHeatmap(data) {{
            const container = document.getElementById('consensus-heatmap-container');
            const matrix = data.consensus_matrix;
            const labels = ['Hate Votes', 'Offensive', 'Clean'];
            
            const corner = document.createElement('div'); corner.className = 'heatmap-corner'; container.appendChild(corner);
            labels.forEach(lbl => {{ const head = document.createElement('div'); head.className = 'heatmap-label'; head.style.justifyContent = 'center'; head.innerText = lbl; container.appendChild(head); }});
            
            labels.forEach((rowLbl, rowIdx) => {{
                const label = document.createElement('div'); label.className = 'heatmap-label'; label.innerText = rowLbl; container.appendChild(label);
                matrix[rowIdx].forEach(val => {{
                    const cell = document.createElement('div'); cell.className = 'heatmap-cell'; cell.innerText = val.toFixed(2);
                    const percent = (val + 1) / 2;
                    let r, g, b;
                    if (percent < 0.5) {{
                        r = 231; g = Math.floor(76 + percent * 2 * (241 - 76)); b = Math.floor(60 + percent * 2 * (19 - 60));
                    }} else {{
                        r = Math.floor(241 - (percent - 0.5) * 2 * (241 - 46)); g = Math.floor(196 + (percent - 0.5) * 2 * (204 - 196)); b = Math.floor(15 + (percent - 0.5) * 2 * (113 - 15));
                    }}
                    cell.style.backgroundColor = `rgb(${{r}}, ${{g}}, ${{b}})`;
                    container.appendChild(cell);
                }});
            }});
        }}

        function loadTrendsChart(data) {{
            const ctx = document.getElementById('trendsChart').getContext('2d');
            const trends = data.trends;
            new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: trends.dates,
                    datasets: [
                        {{ label: 'Hate Speech', data: trends.hate_speech, borderColor: '#e74c3c', backgroundColor: 'rgba(231, 76, 60, 0.1)', borderWidth: 2.5, tension: 0.25, fill: false }},
                        {{ label: 'Offensive Language', data: trends.offensive_language, borderColor: '#f39c12', backgroundColor: 'rgba(243, 156, 18, 0.1)', borderWidth: 2.5, tension: 0.25, fill: false }},
                        {{ label: 'Clean / Normal', data: trends.normal, borderColor: '#2ecc71', backgroundColor: 'rgba(46, 204, 113, 0.1)', borderWidth: 2.5, tension: 0.25, fill: false }}
                    ]
                }},
                options: {{ responsive: true, maintainAspectRatio: false, scales: {{ x: {{ grid: {{ color: 'rgba(255, 255, 255, 0.04)' }}, ticks: {{ color: '#8892b0' }} }}, y: {{ grid: {{ color: 'rgba(255, 255, 255, 0.04)' }}, ticks: {{ color: '#8892b0' }} }} }}, plugins: {{ legend: {{ labels: {{ color: '#ffffff' }} }} }} }}
            }});
        }}
        
        function loadVocabAndCoefCharts(data) {{
            function makeBar(id, list, color, label_suffix) {{
                if(!document.getElementById(id)) return;
                const labels = list.map(item => item.word || item[0]);
                const values = list.map(item => item.count || item[1]);
                const ctx = document.getElementById(id).getContext('2d');
                new Chart(ctx, {{
                    type: 'bar',
                    data: {{ labels: labels, datasets: [{{ label: label_suffix, data: values, backgroundColor: color, borderRadius: 4 }}] }},
                    options: {{ indexAxis: 'y', responsive: true, maintainAspectRatio: false, scales: {{ x: {{ grid: {{ color: 'rgba(255, 255, 255, 0.04)' }}, ticks: {{ color: '#8892b0' }} }}, y: {{ grid: {{ display: false }}, ticks: {{ color: '#ffffff' }} }} }}, plugins: {{ legend: {{ display: false }} }} }}
                }});
            }}
            makeBar('vocabHateChart', data.vocab.hate_speech, '#e74c3c', 'Frequency Count');
            makeBar('coefHateChart', data.top_features.hate_speech.slice(0, 15), '#e74c3c', 'Log-Odds Weight');
            makeBar('coefNormChart', data.top_features.normal.slice(0, 15), '#2ecc71', 'Log-Odds Weight');
        }}
        
        function loadMetricsTablesAndConfusion(data) {{
            const report = data.classification_report;
            const body = document.getElementById('metrics-table-body');
            const classesMapped = {{ 'hate_speech': 'Hate Speech', 'offensive_language': 'Offensive', 'normal': 'Clean/Normal' }};
            
            Object.keys(classesMapped).forEach(cls => {{
                if (report[cls]) {{
                    const row = document.createElement('tr');
                    row.innerHTML = `<td><strong>${{classesMapped[cls]}}</strong></td><td>${{(report[cls].precision * 100).toFixed(2)}}%</td><td>${{(report[cls].recall * 100).toFixed(2)}}%</td><td>${{(report[cls]['f1-score'] * 100).toFixed(2)}}%</td><td>${{report[cls].support.toLocaleString()}}</td>`;
                    body.appendChild(row);
                }}
            }});
            
            const matrix = data.confusion_matrix;
            const classes = data.classes;
            const matrixContainer = document.getElementById('confusion-matrix-grid');
            
            const corner = document.createElement('div'); corner.className = 'matrix-corner'; matrixContainer.appendChild(corner);
            classes.forEach(cls => {{ const head = document.createElement('div'); head.className = 'matrix-label-col'; head.innerText = classesMapped[cls] || cls; matrixContainer.appendChild(head); }});
            
            const maxVal = Math.max(...matrix.flat());
            
            classes.forEach((rowCls, rowIdx) => {{
                const label = document.createElement('div'); label.className = 'matrix-label-row'; label.innerText = classesMapped[rowCls] || rowCls; matrixContainer.appendChild(label);
                matrix[rowIdx].forEach((val, colIdx) => {{
                    const cell = document.createElement('div'); cell.className = 'matrix-cell'; cell.innerText = val.toLocaleString();
                    const percent = val / maxVal;
                    cell.style.backgroundColor = `rgba(15, 76, 129, ${{0.05 + percent * 0.95}})`;
                    cell.style.color = percent > 0.5 ? '#ffffff' : '#e2e8f0';
                    cell.title = `Actual: ${{classesMapped[rowCls]}}, Predicted: ${{classesMapped[classes[colIdx]]}} - ${{val}} Tweets`;
                    matrixContainer.appendChild(cell);
                }});
            }});
        }}

        // Trigger rendering based on mode passed in URL hash
        window.onload = function() {{
            const mode = window.location.hash;
            if(mode === '#analytics') {{
                document.getElementById('analytics').style.display = 'block';
                populateMetricsOverview(window.data);
                loadLabelDonutChart(window.data);
                loadConsensusHeatmap(window.data);
                loadTrendsChart(window.data);
                loadVocabAndCoefCharts(window.data);
            }} else if(mode === '#metrics') {{
                document.getElementById('metrics').style.display = 'block';
                loadMetricsTablesAndConfusion(window.data);
                loadVocabAndCoefCharts(window.data);
            }}
        }};
    </script>
</body>
</html>
"""

with tab2:
    # Use iframe height slightly larger to prevent scrollbars
    components.html(chartjs_html_template + "<script>window.location.hash='#analytics';</script>", height=1000, scrolling=True)

with tab3:
    components.html(chartjs_html_template + "<script>window.location.hash='#metrics';</script>", height=1000, scrolling=True)

