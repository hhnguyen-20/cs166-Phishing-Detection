import pickle
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load the pickled logistic regression model
with open('logistic_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the vectorizer used during training
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Text preprocessing function
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Join tokens back into text
    text = ' '.join(tokens)

    # Return empty string if text is empty after preprocessing
    return text if text else ' '

# Example email text
email_text = "im horny lmao"

# Preprocess the email text
preprocessed_email_text = preprocess_text(email_text)

# Vectorize the preprocessed email text using the same vectorizer
X_email = vectorizer.transform([preprocessed_email_text])

# Make predictions on the vectorized email text
prediction = model.predict(X_email)

# Output the prediction
print("Predicted label:", prediction)
