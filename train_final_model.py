import numpy as np
import pandas as pd
import re
import nltk
import string
import pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
except:
    pass

def clean_text_final(text):
    """Final improved text preprocessing"""
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

def create_balanced_dataset(df):
    """Create a more balanced dataset using undersampling and keyword filtering"""
    print("Creating balanced dataset...")
    
    # Separate classes
    hate_speech = df[df['class'] == 0]
    offensive = df[df['class'] == 1] 
    neither = df[df['class'] == 2]
    
    print(f"Original counts - Hate: {len(hate_speech)}, Offensive: {len(offensive)}, Neither: {len(neither)}")
    
    # Create positive/neutral keywords to help identify "Neither" class better
    positive_keywords = ['love', 'thank', 'great', 'wonderful', 'amazing', 'beautiful', 
                        'good', 'nice', 'happy', 'joy', 'awesome', 'fantastic', 
                        'excellent', 'perfect', 'sweet', 'kind', 'help', 'please']
    
    # Filter offensive language to remove potentially mislabeled positive content
    def likely_mislabeled(text):
        text_lower = text.lower()
        # Check if text contains positive words and no negative words
        has_positive = any(word in text_lower for word in positive_keywords)
        negative_words = ['hate', 'stupid', 'idiot', 'damn', 'shit', 'fuck', 'kill', 'die']
        has_negative = any(word in text_lower for word in negative_words)
        return has_positive and not has_negative
    
    # Move likely mislabeled offensive content to "Neither"
    mislabeled_mask = offensive['tweet'].apply(likely_mislabeled)
    mislabeled_offensive = offensive[mislabeled_mask].copy()
    mislabeled_offensive['class'] = 2  # Change to "Neither"
    mislabeled_offensive['labels'] = 'Neither'
    
    # Keep non-mislabeled offensive content
    clean_offensive = offensive[~mislabeled_mask]
    
    print(f"Moved {len(mislabeled_offensive)} potentially mislabeled samples from Offensive to Neither")
    
    # Combine the corrected data
    corrected_neither = pd.concat([neither, mislabeled_offensive], ignore_index=True)
    
    # Balance the dataset by undersampling the majority class
    target_size = min(len(hate_speech) * 3, len(corrected_neither), len(clean_offensive) // 2)
    
    balanced_hate = hate_speech.sample(n=min(len(hate_speech), target_size), random_state=42)
    balanced_neither = corrected_neither.sample(n=min(len(corrected_neither), target_size), random_state=42)
    balanced_offensive = clean_offensive.sample(n=min(len(clean_offensive), target_size * 2), random_state=42)
    
    balanced_df = pd.concat([balanced_hate, balanced_neither, balanced_offensive], ignore_index=True)
    
    print(f"Balanced dataset counts:")
    print(balanced_df['class'].value_counts())
    
    return balanced_df

def train_final_model():
    """Train the final improved model"""
    print("Loading dataset...")
    
    try:
        dataset = pd.read_csv('Twitter.csv')
        print(f"Successfully loaded Twitter.csv")
    except FileNotFoundError:
        print("Error: Could not find Twitter.csv")
        return False
    
    print(f"Original dataset shape: {dataset.shape}")
    print(f"Original class distribution:\n{dataset['class'].value_counts()}")
    
    # Create labels mapping
    dataset["labels"] = dataset["class"].map({0: "Hate Speech",
                                              1: "Offensive Language", 
                                              2: "Neither"})
    
    # Create balanced dataset
    balanced_dataset = create_balanced_dataset(dataset)
    
    # Prepare data
    data = balanced_dataset[["tweet", "labels"]].copy()
    
    print("Cleaning text data...")
    data.loc[:, 'tweet'] = data['tweet'].apply(clean_text_final)
    
    # Remove empty tweets after cleaning
    data = data[data['tweet'].str.strip() != '']
    data = data[data['tweet'].str.len() > 2]  # Keep only tweets with meaningful content
    
    print(f"Final data shape: {data.shape}")
    
    # Prepare features and target
    X = data['tweet'].values
    y = data['labels'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training final model with optimized parameters...")
    
    # Create pipeline with TF-IDF and Logistic Regression
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 3),  # Use unigrams, bigrams, and trigrams
            min_df=3,
            max_df=0.8,
            strip_accents='ascii',
            lowercase=True
        )),
        ('classifier', LogisticRegression(
            class_weight='balanced',
            random_state=42,
            max_iter=1000,
            C=1.0
        ))
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Model accuracy: {accuracy:.4f}")
    print(f"Weighted F1-score: {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Test with positive sentences
    print("\nTesting with positive sentences:")
    test_sentences = [
        'I love you',
        'Have a great day',
        'Thank you so much',
        'This is wonderful',
        'You are kind',
        'Beautiful weather today',
        'Great job on the project',
        'Hope you have fun',
        'Good morning everyone',
        'Nice to meet you'
    ]
    
    print('=' * 60)
    correct_predictions = 0
    
    for sentence in test_sentences:
        prediction = pipeline.predict([sentence])[0]
        probabilities = pipeline.predict_proba([sentence])[0]
        classes = pipeline.classes_
        
        print(f"Text: '{sentence}'")
        print(f"Prediction: {prediction}")
        
        # Check if prediction is correct (should be "Neither" for positive sentences)
        is_correct = prediction == "Neither"
        correct_predictions += is_correct
        
        max_prob = max(probabilities)
        confidence = "High" if max_prob > 0.7 else "Medium" if max_prob > 0.5 else "Low"
        print(f"Confidence: {confidence} ({max_prob:.3f})")
        
        prob_dict = {classes[i]: f'{probabilities[i]:.3f}' for i in range(len(classes))}
        print(f"Probabilities: {prob_dict}")
        print(f"Correct: {'✓' if is_correct else '✗'}")
        print('-' * 40)
    
    accuracy_on_positive = correct_predictions / len(test_sentences)
    print(f"\nAccuracy on positive sentences: {accuracy_on_positive:.2%} ({correct_predictions}/{len(test_sentences)})")
    
    # Save the final model
    print("Saving final model...")
    with open('hate_speech_model_final.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    
    print("Final model training complete!")
    print("- hate_speech_model_final.pkl (contains the complete pipeline)")
    
    return True

if __name__ == "__main__":
    train_final_model()
