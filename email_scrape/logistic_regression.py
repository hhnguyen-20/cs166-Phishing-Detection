from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import pandas as pd

# Load preprocessed dataset
data = pd.read_csv('preprocessed_train_dataset.csv')

# Extract features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform((data['body']))
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=62)

# Train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))
print('F1 score:', f1_score(y_test, y_pred))

# Save model to file
with open('logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save vectorizer to file
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)