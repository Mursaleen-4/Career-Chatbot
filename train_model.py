"""
Career Guidance Chatbot - Model Training Script
This script loads the career guidance dataset, preprocesses the text data,
trains a machine learning classifier, and saves the trained model.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, f1_score
import joblib
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class CareerGuidanceModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.lemmatizer = WordNetLemmatizer()
        
    def preprocess_text(self, text):
        """
        Preprocess text data by:
        - Converting to lowercase
        - Removing punctuation
        - Removing extra whitespace
        - Optional lemmatization
        """
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Lemmatization (optional)
        words = text.split()
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
        text = ' '.join(lemmatized_words)
        
        return text
    
    def load_and_preprocess_data(self, csv_file_path):
        """
        Load the career guidance dataset and preprocess it
        """
        print("Loading dataset...")
        df = pd.read_csv(csv_file_path)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        # Handle both column name formats
        role_col = 'Role' if 'Role' in df.columns else 'role'
        question_col = 'Question' if 'Question' in df.columns else 'question'
        answer_col = 'Answer' if 'Answer' in df.columns else 'answer'
        
        print(f"Number of unique roles: {df[role_col].nunique()}")
        
        # Check for missing values
        print("\nMissing values:")
        print(df.isnull().sum())
        
        # Remove rows with missing questions
        df = df.dropna(subset=[question_col, role_col])
        
        # Preprocess the questions
        print("Preprocessing text data...")
        df['Processed_Question'] = df[question_col].apply(self.preprocess_text)
        
        # Store column names for later use
        df.attrs['role_col'] = role_col
        df.attrs['question_col'] = question_col
        df.attrs['answer_col'] = answer_col
        
        return df
    
    def train_model(self, df):
        """
        Train the classification model
        """
        print("Preparing data for training...")
        
        # Features and labels
        role_col = df.attrs.get('role_col', 'role')
        X = df['Processed_Question']
        y = df[role_col]
        
        # Split the data (disable stratify for small datasets)
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError:
            # If stratify fails due to small dataset, split without stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        # Vectorize the text
        print("Vectorizing text data...")
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        X_test_vectorized = self.vectorizer.transform(X_test)
        
        # Train the model
        print("Training the model...")
        self.model.fit(X_train_vectorized, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_vectorized)
        
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"\nModel Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred))
        
        return accuracy, f1
    
    def save_model(self):
        """
        Save the trained model and vectorizer
        """
        print("Saving model and vectorizer...")
        joblib.dump(self.model, 'intent_model.pkl')
        joblib.dump(self.vectorizer, 'vectorizer.pkl')
        print("Model and vectorizer saved successfully!")
    
    def predict_career_role(self, question):
        """
        Predict career role for a given question
        """
        processed_question = self.preprocess_text(question)
        question_vectorized = self.vectorizer.transform([processed_question])
        prediction = self.model.predict(question_vectorized)[0]
        confidence = self.model.predict_proba(question_vectorized).max()
        
        return prediction, confidence

def main():
    """
    Main function to train and save the career guidance model
    """
    # Initialize the model
    career_model = CareerGuidanceModel()
    
    # Load and preprocess data
    try:
        df = career_model.load_and_preprocess_data('career_guidance_dataset.csv')
    except FileNotFoundError:
        print("Error: 'career_guidance_dataset.csv' not found!")
        print("Please download the dataset and place it in the same directory.")
        return
    
    # Train the model
    accuracy, f1 = career_model.train_model(df)
    
    # Save the model
    career_model.save_model()
    
    # Test with a sample question
    print("\nTesting with a sample question:")
    sample_question = "I want to work with data and analyze business trends"
    prediction, confidence = career_model.predict_career_role(sample_question)
    print(f"Question: {sample_question}")
    print(f"Predicted Role: {prediction}")
    print(f"Confidence: {confidence:.4f}")

if __name__ == "__main__":
    main()
