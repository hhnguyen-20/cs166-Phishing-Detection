import numpy as np
import pandas as pd
import seaborn as sns
import pickle

import matplotlib.pyplot as plt


df = pd.read_csv('/Users/isiahketton/Downloads/CEAS_08.csv')
df.head()
df.tail()
df.info()

df.isnull().sum()
df.duplicated().sum()

df.dropna(inplace=True)
df.isnull().sum()

from sklearn.naive_bayes import ComplementNB, BernoulliNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

models = [ComplementNB(), BernoulliNB(), MultinomialNB()]
model_names = ['ComplementNB', 'BernoulliNB', 'MultinomialNB']


accuracies = []

url_text = df['urls'].astype(str)
text = df['sender'] + ' ' + df['receiver'] + ' ' + df['date'] + ' ' + ['subject'] + ' ' + ['body'] + ' ' + url_text
convert_feature = TfidfVectorizer()

X = convert_feature.fit_transform(text)
Y = df['label']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.8, random_state=42)

for model in models:
  model.fit(X_train, Y_train)
  pred = model.predict(X_test)
  
  accuracies.append(accuracy_score(Y_test, pred))

axis = np.arange(len(model_names))

for i, model_name in enumerate(model_names):
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracies[i]}")

# confusion_matrix
confusion_matrix(Y_test, pred)

# classification_report
classification_report(Y_test, pred)

for i, model in enumerate(models):
    model_name = model_names[i]
    model_filename = f"{model_name}.pkl"
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)