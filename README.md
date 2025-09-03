# Hate Speech Detection Web Application

A modern web-based application that uses machine learning to classify text as hate speech, offensive language, or neutral content. Built with Flask backend and responsive HTML/CSS/JavaScript frontend.

## Features

- **Real-time Text Analysis**: Instantly classify text input as hate speech, offensive language, or neutral
- **Modern Web Interface**: Clean, responsive design that works on desktop and mobile
- **Confidence Scores**: View probability scores for each classification category
- **Text Preprocessing**: See how your text is processed before analysis
- **Live Character Counter**: Track input length with visual feedback
- **Error Handling**: Comprehensive error messages and validation

## Project Structure

```
hate_speech_detection/
├── app.py                 # Flask web application
├── train_model.py         # Model training script
├── main.ipynb            # Original Jupyter notebook (development)
├── Twitter.csv           # Training dataset
├── requirements.txt      # Python dependencies
├── templates/
│   └── index.html        # Main web interface
├── static/
│   ├── style.css         # CSS styling
│   └── script.js         # JavaScript functionality
├── hate_speech_model.pkl # Trained model (generated)
├── vectorizer.pkl        # Text vectorizer (generated)
└── README.md             # This file
```

## Installation & Setup

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Step 1: Install Dependencies

Navigate to the project directory and install required packages:

```bash
cd hate_speech_detection
pip install -r requirements.txt
```

### Step 2: Train the Model

Before running the web application, you need to train and save the model:

```bash
python train_model.py
```

This will:
- Load and preprocess the Twitter dataset
- Train a Decision Tree classifier
- Save the trained model and vectorizer as pickle files
- Display model accuracy

**Expected Output:**
```
Loading dataset...
Successfully loaded Twitter.csv
Cleaning text data...
Vectorizing text data...
Splitting data...
Training Decision Tree model...
Model accuracy: 0.8699
Saving model and vectorizer...
Model training complete! Files saved:
- hate_speech_model.pkl
- vectorizer.pkl
```

### Step 3: Run the Web Application

Start the Flask server:

```bash
python app.py
```

The application will start on `http://localhost:5000`

## Usage

### Web Interface

1. **Open your browser** and navigate to `http://localhost:5000`
2. **Enter text** in the textarea (up to 1000 characters)
3. **Click "Analyze Text"** or press `Ctrl+Enter`
4. **View results** including:
   - Classification (Hate Speech, Offensive Language, or Neither)
   - Confidence scores for each category
   - Processed text showing how the input was cleaned

### API Usage

You can also use the application programmatically via REST API:

#### Predict Endpoint

**URL:** `POST /predict`

**Request Body:**
```json
{
    "text": "Your text to analyze here"
}
```

**Response:**
```json
{
    "prediction": "Neither",
    "cleaned_text": "cleaned preprocessed text",
    "probabilities": {
        "Hate Speech": 0.1,
        "Offensive Language": 0.2,
        "Neither": 0.7
    }
}
```

#### Health Check

**URL:** `GET /health`

**Response:**
```json
{
    "status": "healthy",
    "model_loaded": true,
    "vectorizer_loaded": true
}
```

## Model Information

### Dataset
- **Source**: Twitter dataset with labeled tweets
- **Size**: 24,783 tweets
- **Labels**: 
  - `0`: Hate Speech
  - `1`: Offensive Language  
  - `2`: Neither

### Classification Categories

- **Hate Speech**: Content that attacks or discriminates against individuals or groups
- **Offensive Language**: Inappropriate or harmful language that doesn't necessarily target specific groups
- **Neither**: Neutral content without hate speech or offensive language

### Model Performance
- **Algorithm**: Decision Tree Classifier
- **Accuracy**: ~87%
- **Features**: Count Vectorization of preprocessed text

### Text Preprocessing Pipeline

1. **Lowercase conversion**
2. **URL removal**
3. **HTML tag removal**
4. **Punctuation removal**
5. **Stopword removal**
6. **Stemming** (reducing words to root form)

## Keyboard Shortcuts

- `Ctrl+Enter`: Analyze text
- `Escape`: Clear input and results

## Troubleshooting

### Common Issues

**1. ModuleNotFoundError**
```bash
# Install missing dependencies
pip install -r requirements.txt
```

**2. NLTK Data Missing**
```python
import nltk
nltk.download('stopwords')
```

**3. Model Files Not Found**
```bash
# Train the model first
python train_model.py
```

**4. Port Already in Use**
- Change the port in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Change port
```

### Performance Notes

- Text longer than 1000 characters will be rejected
- Model loading takes a few seconds on first startup
- Predictions are typically returned within 1-2 seconds

## Development

### Running in Development Mode

The Flask app runs in debug mode by default, which includes:
- Auto-reload on code changes
- Detailed error messages
- Interactive debugger

### Customization

- **Styling**: Modify `static/style.css`
- **Frontend Logic**: Update `static/script.js`
- **Backend Logic**: Edit `app.py`
- **Model**: Retrain using `train_model.py`

## Security Considerations

- Input validation (length limits, content sanitization)
- Error handling to prevent information disclosure
- No persistent storage of user input
- Rate limiting can be added for production use

## License

This project is for educational purposes. Please ensure compliance with applicable laws and ethical guidelines when using hate speech detection systems.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the console output for error messages
3. Ensure all dependencies are installed correctly
4. Verify that model files were generated successfully
