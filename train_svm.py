# -*- coding: utf-8 -*-
"""cs166 - Phishing Email Detection using SVM.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1JnSALcD-LwHOelQKzCKAy7FpEVdX84kW
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#Import the Dataset
df= pd.read_csv("datasets/CEAS_08.csv")

df.head(10)

#Drop tha Na values
df = df.dropna()

# Check NAN values
df.isna().sum()

#datasets shape
df.shape

# Count the occurrences of each E-mail type.
email_type_counts = df['label'].value_counts()
print(email_type_counts)

# Create the bar chart
plot = sns.catplot(
    data=df,
    x='label',
    kind='count',
    palette='Set1',
    hue='label').set_axis_labels("Email Type");

# Show the chart
plot.fig.suptitle("Distribution of Email Types with Custom Colors", y=1.1);

# We will use undersapling technique
Safe_Email = df[df["label"]== 1]
Phishing_Email = df[df["label"]== 0]
Safe_Email = Safe_Email.sample(Phishing_Email.shape[0])

# lets check the shape again
Safe_Email.shape,Phishing_Email.shape

# lest create a new Data with the balanced E-mail types
Data = pd.concat([Safe_Email, Phishing_Email], ignore_index = True)
Data

# split the dataset into a metrix of features X and Dependent Variable y
X = Data["subject"].values.astype('U')
y = Data["label"].values

# lets splitting Our Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 0)

from sklearn.feature_extraction.text import CountVectorizer

# Converting String to Integer
cv = CountVectorizer()
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)

# Build SVM Model
# Importing SVM
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

#Create the Pipeline
SVM = Pipeline([("SVM", SVC(C = 100, gamma = "auto"))])

# traing the SVM model
SVM.fit(X_train, y_train)
print("SVM Model Trained Successfully")

# y_pred. for SVM model
s_ypred = SVM.predict(X_test)

SVM.score(X_test, y_test)

# check the SVM model accuracy
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

print(accuracy_score(y_test, s_ypred))

# confusion_matrix
confusion_matrix(y_test, s_ypred)

# classification_report
classification_report(y_test, s_ypred)