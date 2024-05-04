import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load preprocessed test dataset
data = pd.read_csv('preprocessed_test_dataset.csv')

# Load the pickled logistic regression model
with open('logistic_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the vectorizer used during training
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Transform the new dataset using the same vectorizer
X_test = vectorizer.transform(data['body'])

# Make predictions on the transformed test data
predictions = model.predict(X_test)

# Same labels are available for evaluation
y_test = data['label']

# Calculate accuracy
print("Accuracy:", accuracy_score(y_test, predictions))
print('Precision:', precision_score(y_test, predictions))
print('Recall:', recall_score(y_test, predictions))
print('F1 score:', f1_score(y_test, predictions))