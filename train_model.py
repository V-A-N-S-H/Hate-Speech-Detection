import pandas as pd # used for data cleaning
import numpy as np # for numerical operations
import re # used for text cleaning
import string # for string manipulation
import pickle # for saving model
from sklearn.model_selection import train_test_split # for splitting data into train and test
from sklearn.feature_extraction.text import TfidfVectorizer # for converting text to numerical features
from sklearn.linear_model import LogisticRegression # for training model
from sklearn.metrics import accuracy_score, classification_report # for finding accuracy and generating report

def preprocess_text(text): # for data cleaning and preprocessing
    if pd.isna(text):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove URLs, mentions, hashtags
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def load_and_prepare_data(): # for loading and preparing data
    df = pd.read_csv("Twitter.csv")
    df.columns = df.columns.str.strip()
    
    texts = df["tweet"].values
    labels = df["class"].values
    
    # Clean texts
    texts_cleaned = [preprocess_text(text) for text in texts]
    
    # Binary labels: 1 -> Hate/Offensive, 0 -> Normal
    labels_binary = np.where(labels == 2, 0, 1)
    
    return texts_cleaned, labels_binary

def train_model(): # for training model
    texts, labels = load_and_prepare_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Vectorize text 
    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Logistic Regression for model training
    model = LogisticRegression(
        random_state=42, max_iter=1000, class_weight="balanced"
    )
    model.fit(X_train_vec, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_vec)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=["Normal","Hate/Offensive"]))
    
    # Save model and vectorizer
    pickle.dump(model, open("hate_speech_model.pkl", "wb"))
    pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
    
    return model, vectorizer

def test_model(): # for training model
    model = pickle.load(open("hate_speech_model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    
    test_texts = [
        "I love this beautiful day!",
        "You are such an idiot and should die",
        "Great weather today, perfect for a walk",
        "I hate all these stupid people",
        "Have a wonderful day everyone!"
    ]
    
    for text in test_texts:
        cleaned = preprocess_text(text)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]
        
        result = "Hate/Offensive" if pred == 1 else "Normal"
        confidence = max(prob)
        
        print(f"Text: '{text}'")
        print(f"Prediction: {result} (Confidence: {confidence:.4f})")
        print("-"*50)

if __name__ == "__main__": # used to stop the code from running automatically while importing the file
    model, vectorizer = train_model()
    test_model()