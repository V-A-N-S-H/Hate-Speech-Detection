# Hate Speech Detection Using NLP

A Machine Learning and Natural Language Processing (NLP) project that detects hate speech, offensive language, and normal text using TF-IDF Vectorization and Logistic Regression. The application is built with Streamlit and provides real-time text classification along with interactive analytics and model insights.

## Live Demo

Deployment:
https://hate-speech-detection-5ouxxvzrbpfe4vzzt4uj2b.streamlit.app/

---

## Features

* Detects Hate Speech, Offensive Language, and Normal text
* Real-time text classification
* Advanced text preprocessing
* TF-IDF Vectorization
* Logistic Regression classifier
* Interactive analytics dashboard
* Model performance metrics
* Confidence score visualization
* Feature attribution highlighting
* Streamlit web application

---

## Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* NLTK
* Joblib

---

## Project Structure

```text
Hate-Speech-Detection-Using-NLP/
│
├── app.py                     # Streamlit application
├── train_model.py             # Model training script
├── main.ipynb                 # Data preprocessing and model development
├── Twitter.csv                # Dataset
├── model.joblib               # Trained Logistic Regression model
├── vectorizer.joblib          # TF-IDF Vectorizer
├── model_metadata.joblib      # Model evaluation metrics
├── requirements.txt           # Project dependencies
├── .gitignore
└── README.md
```

---

## How It Works

1. Load the Twitter Hate Speech dataset.
2. Clean and preprocess the text by:

   * Converting text to lowercase
   * Removing URLs, mentions, hashtags, punctuation, and stopwords
   * Applying stemming using NLTK.
3. Convert cleaned text into TF-IDF feature vectors.
4. Split the dataset into training and testing sets.
5. Train a Logistic Regression classifier.
6. Evaluate the model using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
7. Save the trained model, vectorizer, and metadata.
8. Accept user input through the Streamlit application.
9. Predict whether the text belongs to:

   * Hate Speech
   * Offensive Language
   * Normal

---

## Dataset

The project uses the Twitter Hate Speech Dataset containing labeled tweets for text classification.

### Classes

* Hate Speech
* Offensive Language
* Normal

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/V-A-N-S-H/Hate-Speech-Detection-Using-NLP.git
```

### 2. Navigate to the project directory

```bash
cd Hate-Speech-Detection-Using-NLP
```

### 3. Install the required dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

---

## Machine Learning Workflow

* Data Cleaning
* Text Preprocessing
* Stopword Removal
* Stemming
* TF-IDF Vectorization
* Train-Test Split
* Logistic Regression
* Model Evaluation
* Real-Time Prediction

---

## Model Used

* Logistic Regression

---

## NLP Techniques

* Text Cleaning
* Tokenization
* Stopword Removal
* Stemming
* TF-IDF Vectorization

---

## Model Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Classification Report

---

## Future Improvements

* Deep Learning models (LSTM, GRU)
* Transformer-based models (BERT, RoBERTa)
* Multilingual hate speech detection
* Batch prediction using CSV upload
* Explainable AI (SHAP/LIME)
* API deployment using FastAPI

---

## Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a new branch.
3. Commit your changes.
4. Push the branch.
5. Open a Pull Request.

---

## Author

**Vansh**

GitHub: https://github.com/V-A-N-S-H

---

## Support

If you found this project helpful, consider giving it a star on GitHub.
# Hate Speech Detection Using NLP

A Machine Learning and Natural Language Processing (NLP) project that detects hate speech, offensive language, and normal text using TF-IDF Vectorization and Logistic Regression. The application is built with Streamlit and provides real-time text classification along with interactive analytics and model insights.

## Live Demo

Deployment:
https://hate-speech-detection-5ouxxvzrbpfe4vzzt4uj2b.streamlit.app/

---

## Features

* Detects Hate Speech, Offensive Language, and Normal text
* Real-time text classification
* Advanced text preprocessing
* TF-IDF Vectorization
* Logistic Regression classifier
* Interactive analytics dashboard
* Model performance metrics
* Confidence score visualization
* Feature attribution highlighting
* Streamlit web application

---

## Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* NLTK
* Joblib

---

## Project Structure

```text
Hate-Speech-Detection-Using-NLP/
│
├── app.py                     # Streamlit application
├── train_model.py             # Model training script
├── main.ipynb                 # Data preprocessing and model development
├── Twitter.csv                # Dataset
├── model.joblib               # Trained Logistic Regression model
├── vectorizer.joblib          # TF-IDF Vectorizer
├── model_metadata.joblib      # Model evaluation metrics
├── requirements.txt           # Project dependencies
├── .gitignore
└── README.md
```

---

## How It Works

1. Load the Twitter Hate Speech dataset.
2. Clean and preprocess the text by:

   * Converting text to lowercase
   * Removing URLs, mentions, hashtags, punctuation, and stopwords
   * Applying stemming using NLTK.
3. Convert cleaned text into TF-IDF feature vectors.
4. Split the dataset into training and testing sets.
5. Train a Logistic Regression classifier.
6. Evaluate the model using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
7. Save the trained model, vectorizer, and metadata.
8. Accept user input through the Streamlit application.
9. Predict whether the text belongs to:

   * Hate Speech
   * Offensive Language
   * Normal

---

## Dataset

The project uses the Twitter Hate Speech Dataset containing labeled tweets for text classification.

### Classes

* Hate Speech
* Offensive Language
* Normal

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/V-A-N-S-H/Hate-Speech-Detection-Using-NLP.git
```

### 2. Navigate to the project directory

```bash
cd Hate-Speech-Detection-Using-NLP
```

### 3. Install the required dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

---

## Machine Learning Workflow

* Data Cleaning
* Text Preprocessing
* Stopword Removal
* Stemming
* TF-IDF Vectorization
* Train-Test Split
* Logistic Regression
* Model Evaluation
* Real-Time Prediction

---

## Model Used

* Logistic Regression

---

## NLP Techniques

* Text Cleaning
* Tokenization
* Stopword Removal
* Stemming
* TF-IDF Vectorization

---

## Model Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Classification Report

---

## Future Improvements

* Deep Learning models (LSTM, GRU)
* Transformer-based models (BERT, RoBERTa)
* Multilingual hate speech detection
* Batch prediction using CSV upload
* Explainable AI (SHAP/LIME)
* API deployment using FastAPI

---

## Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a new branch.
3. Commit your changes.
4. Push the branch.
5. Open a Pull Request.

---

## Author

**Vansh**

GitHub: https://github.com/V-A-N-S-H

---

## Support

If you found this project helpful, consider giving it a star on GitHub.
