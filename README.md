# Career Guidance Chatbot

A machine learning-powered chatbot that provides career guidance based on user interests and questions. Built with Streamlit frontend and scikit-learn for classification.

## 🎯 Features

- **Natural Language Processing**: Understands user questions about career interests
- **ML-Powered Classification**: Uses trained models to predict relevant career roles
- **Interactive Web Interface**: Beautiful Streamlit frontend with chat history
- **Confidence Scoring**: Shows how confident the model is about its predictions
- **Multiple Suggestions**: Provides top 3 career recommendations
- **Real-time Processing**: Instant responses to user queries

## 📁 Project Structure

```
career_chatbot_project/
├── app.py                          # Streamlit frontend application
├── train_model.py                  # Model training script
├── career_guidance_dataset.csv     # Dataset (to be downloaded)
├── intent_model.pkl               # Trained model (generated after training)
├── vectorizer.pkl                 # TF-IDF vectorizer (generated after training)
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the Dataset

Download the career guidance dataset and place it in the project directory as `career_guidance_dataset.csv`.

The dataset should contain columns:
- `Role`: Career role (e.g., "Data Scientist", "Product Manager")
- `Question`: User question about that role
- `Answer`: Informative response about the career

### 3. Train the Model

```bash
python train_model.py
```

This will:
- Load and preprocess the dataset
- Train a Logistic Regression classifier
- Evaluate model performance
- Save the trained model and vectorizer

### 4. Run the Chatbot

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 🤖 How It Works

1. **Text Preprocessing**: User input is cleaned, lowercased, and lemmatized
2. **Feature Extraction**: TF-IDF vectorization converts text to numerical features
3. **Classification**: Logistic Regression model predicts the most relevant career role
4. **Results Display**: Shows the top prediction with confidence score and alternatives

## 💡 Sample Questions

Try asking questions like:
- "I love working with data and numbers"
- "I want to create mobile applications"
- "I'm interested in helping people solve problems"
- "I enjoy designing user interfaces"
- "I want to work in cybersecurity"

## 📊 Model Performance

The model uses:
- **Algorithm**: Logistic Regression
- **Feature Extraction**: TF-IDF (5000 max features)
- **Text Preprocessing**: Lowercase, punctuation removal, lemmatization
- **Evaluation Metrics**: Accuracy and F1-score

## 🛠️ Technical Details

### Dependencies
- `streamlit`: Web interface framework
- `pandas`: Data manipulation and analysis
- `scikit-learn`: Machine learning algorithms
- `nltk`: Natural language processing
- `joblib`: Model serialization

### Model Architecture
- **Input**: Natural language questions about career interests
- **Preprocessing**: Text cleaning and TF-IDF vectorization
- **Classification**: Logistic Regression with 54 career role classes
- **Output**: Predicted career role with confidence score

## 🔧 Customization

### Adding New Career Roles
1. Update the dataset with new roles and corresponding questions/answers
2. Retrain the model using `python train_model.py`

### Improving Model Performance
- Try different algorithms (SVM, Naive Bayes) by modifying `train_model.py`
- Adjust TF-IDF parameters (max_features, n-grams)
- Add more sophisticated text preprocessing

### UI Customization
- Modify `app.py` to change the Streamlit interface
- Add new features like role descriptions, salary information, etc.

## 🎥 Demo Video

Create a short video demonstration showing:
1. Model training process
2. Starting the Streamlit application
3. Asking sample questions
4. Reviewing predictions and confidence scores
5. Exploring chat history

## 📈 Future Enhancements

- **Advanced NLP**: Use pre-trained transformers (BERT, GPT)
- **Database Integration**: Store user interactions and feedback
- **Role Descriptions**: Add detailed career information
- **Salary Predictions**: Include salary range estimates
- **Skills Matching**: Match user skills to role requirements
- **Learning Path**: Suggest courses and certifications

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is for educational purposes as part of a machine learning internship program.

---

**Built with ❤️ using Python, Streamlit, and scikit-learn**
