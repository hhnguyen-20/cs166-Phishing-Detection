import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Data loading + pre processing
df= pd.read_csv("Phishing_Email.csv")
df.head()
df.isna().sum()
df = df.dropna()
df.shape

# Evaluate the distribution of safe/unsafe emails in a chart
"""
# Count the occurrences of each E-mail type. 
email_type_counts = df['Email Type'].value_counts()
print(email_type_counts)

# Create the bar chart
# Create a list of unique email types
unique_email_types = email_type_counts.index.tolist()

# Define a custom color map 
color_map = {
    'Phishing Email': 'red',
    'Safe Email': 'green',}

# Map the colors to each email type
colors = [color_map.get(email_type, 'gray') for email_type in unique_email_types]

# Create the bar chart with custom colors
plt.figure(figsize=(8, 6))
plt.bar(unique_email_types, email_type_counts, color=colors)
plt.xlabel('Email Type')
plt.ylabel('Count')
plt.title('Distribution of Email Types with Custom Colors')
plt.xticks(rotation=45)

# Show the chart
plt.tight_layout()
plt.show()
"""

# Undersampling resampling technique used
Safe_Email = df[df["Email Type"]== "Safe Email"]
Phishing_Email = df[df["Email Type"]== "Phishing Email"]
Safe_Email = Safe_Email.sample(Phishing_Email.shape[0])

# Check shape again
Safe_Email.shape,Phishing_Email.shape

# Create a new Data with the balanced E-mail types
Data= pd.concat([Safe_Email, Phishing_Email], ignore_index = True)
Data.head()

# split the data into a metrix of features X and Dependent Variable y
X = Data["Email Text"].values
y = Data["Email Type"].values

# Split the data 
from sklearn.model_selection import train_test_split
X_train,x_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Building RFC Model
# Importing Libraries for the model, Tfidf and Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# define the Classifier
classifier = Pipeline([("tfidf",TfidfVectorizer() ),("classifier",RandomForestClassifier(n_estimators=10))])# add another hyperparamters as U want

# Train the model
classifier.fit(X_train,y_train)

# Save the trained model using the pickle library
import pickle
with open('random_forest_classifier.pkl', 'wb') as file:
    pickle.dump(classifier, file)

# Prediction
y_pred = classifier.predict(x_test)

# Checking Accuracy
# Importing classification_report,accuracy_score,confusion_matrix
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix

# accuracy_score
print(accuracy_score(y_test,y_pred))

# confusion_matrix
confusion_matrix(y_test,y_pred)

#classification_report
classification_report(y_test,y_pred)
