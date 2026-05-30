# Hate Speech Detection using Machine Learning

A Machine Learning project that detects hate speech and offensive language from text data using Natural Language Processing (NLP).

This project uses TF-IDF Vectorization and Logistic Regression to classify text as:
- Hate/Offensive
- Normal

The project also includes a Flask web application for real-time predictions.

---

# About The Project

I made this project to learn:
- Natural Language Processing (NLP)
- Text preprocessing
- Machine Learning classification
- TF-IDF Vectorization
- Flask web development
- Model deployment

The system:
- Cleans and preprocesses text
- Converts text into numerical vectors
- Trains a machine learning model
- Predicts whether text contains hate speech
- Returns prediction confidence score

---

# Technologies Used

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- Pickle
- HTML
- CSS
- JavaScript

---

# Project Structure

```text
Hate-Speech-Detection/
│
├── static/                         # Static files
│   ├── css/                        # CSS files
│   ├── js/                         # JavaScript files
│
├── templates/                      # HTML templates
│   └── index.html
│
├── Twitter.csv                     # Dataset
├── hate_speech_model.pkl           # Trained ML model
├── vectorizer.pkl                  # Saved TF-IDF vectorizer
│
├── train_model.py                  # Model training script
├── app.py                          # Flask web application
│
├── requirements.txt                # Project dependencies
├── .gitignore
│
└── README.md                       # Project documentation
```

---

# Features

- Hate speech detection
- Offensive language detection
- Text preprocessing
- TF-IDF Vectorization
- Logistic Regression model
- Flask web application
- Confidence score prediction
- Web interface for testing

---

# Machine Learning Workflow

1. Load Twitter dataset  
2. Clean and preprocess text  
3. Remove URLs, mentions, hashtags, and punctuation  
4. Convert text into TF-IDF vectors  
5. Split dataset into train and test sets  
6. Train Logistic Regression model  
7. Evaluate model accuracy  
8. Save trained model and vectorizer  
9. Predict hate speech from user input  

---

# Installation

## Step 1 — Clone the Repository

```bash
git clone https://github.com/V-A-N-S-H/Hate-Speech-Detection.git
cd Hate-Speech-Detection
```

---

## Step 2 — Install Requirements

```bash
pip install -r requirements.txt
```

---

# How To Run The Project

## Step 1 — Train The Model

```bash
python train_model.py
```

This will:
- Train the model
- Save `hate_speech_model.pkl`
- Save `vectorizer.pkl`

---

## Step 2 — Run Flask Application

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

---

# Model Used

- Logistic Regression
- TF-IDF Vectorizer

---

# Dataset

- Twitter Hate Speech Dataset

---

# Example Predictions

| Input Text | Prediction |
|---|---|
| I love this beautiful day | Normal |
| You are such an idiot | Hate/Offensive |
| Have a wonderful day | Normal |

---

# Requirements

```text
flask
flask-cors
numpy
pandas
scikit-learn
pickle
```
