import pandas as pd
import numpy as np
import re
import string
import nltk
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    print("Step 1: Downloading required NLTK resources...")
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    stop_words_set = set(stopwords.words('english'))
    stemmer = nltk.SnowballStemmer('english')

    print("Step 2: Loading dataset Twitter.csv...")
    df = pd.read_csv('Twitter.csv')
    
    # Map labels to human-readable format
    label_map = {0: 'hate_speech', 1: 'offensive_language', 2: 'normal'}
    df['labels'] = df['class'].map(label_map)
    df = df[['tweet', 'labels']].dropna()
    
    print(f"Dataset shape: {df.shape}")

    print("Step 3: Cleaning text using corrected stemming algorithm...")
    def clean_text_corrected(text):
        if pd.isna(text):
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Remove URLs, mentions, hashtags, html tags, bracketed text
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        text = re.sub(r"@\w+|#\w+", "", text)
        text = re.sub(r"\[.*?\]", "", text)
        text = re.sub(r"<.*?>+", "", text)
        text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
        text = re.sub(r"\n", " ", text)
        text = re.sub(r"\w*\d\w*", "", text)

        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Corrected: stem individual words and filter out stopwords
        words = [stemmer.stem(word) for word in text.split(' ') if word not in stop_words_set and len(word) > 0]
        return ' '.join(words)

    df['cleaned_tweet'] = df['tweet'].apply(clean_text_corrected)
    
    # Filter out empty tweets
    df = df[df['cleaned_tweet'].str.strip().str.len() > 0]
    print(f"Cleaned dataset shape: {df.shape}")

    X = df['cleaned_tweet'].tolist()
    Y = df['labels'].tolist()

    print("Step 4: Transforming text using TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    X_vec = vectorizer.fit_transform(X)

    print("Step 5: Splitting dataset into training and testing sets...")
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_vec, Y, test_size=0.2, random_state=42, stratify=Y
    )

    print("Step 6: Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, Y_train)

    print("Step 7: Evaluating model performance...")
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f"Test Set Accuracy: {accuracy * 100:.2f}%")

    # Classification Report
    c_report = classification_report(Y_test, Y_pred, output_dict=True)
    c_report_text = classification_report(Y_test, Y_pred)
    print("\nClassification Report:")
    print(c_report_text)

    # Confusion Matrix
    classes = list(model.classes_)  # Order of classes in the model
    cm = confusion_matrix(Y_test, Y_pred, labels=classes)
    print("\nConfusion Matrix:")
    print(cm)

    print("Step 8: Computing feature coefficients for dashboard analytics...")
    feature_names = vectorizer.get_feature_names_out()
    coef_dict = {}
    
    for i, class_label in enumerate(classes):
        # Sort coefficients in descending order
        top_indices = np.argsort(model.coef_[i])[::-1][:50]
        top_features = [(feature_names[idx], float(model.coef_[i][idx])) for idx in top_indices]
        coef_dict[class_label] = top_features

    # Combine everything into metadata
    metadata = {
        'accuracy': accuracy,
        'classification_report': c_report,
        'confusion_matrix': cm.tolist(),
        'classes': classes,
        'top_features': coef_dict
    }

    print("Step 9: Serializing model, vectorizer, and metadata...")
    joblib.dump(model, 'model.joblib')
    joblib.dump(vectorizer, 'vectorizer.joblib')
    joblib.dump(metadata, 'model_metadata.joblib')
    print("Model, Vectorizer, and Metadata saved successfully!")

if __name__ == "__main__":
    main()
