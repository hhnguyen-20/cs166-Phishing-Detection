import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

with open("/Users/isiahketton/Desktop/Bayes Model/BernoulliNB.pkl", 'rb') as f:
    loaded_model = pickle.load(f)
with open("/Users/isiahketton/Desktop/Bayes Model/ComplementNB.pkl", 'rb') as f:
    loaded_model2 = pickle.load(f)
with open("/Users/isiahketton/Desktop/Bayes Model/MultinomialNB.pkl", 'rb') as f:
    loaded_model3 = pickle.load(f)
    
df = pd.read_csv('/Users/isiahketton/Downloads/processed_data.csv')
df.head()
df.tail()
df.info()

df.isnull().sum()
df.duplicated().sum()

df.dropna()
df.isnull().sum()



text = df['email_from'] + ' ' + ['subject'] + ' ' + ['body']
convert_feature = TfidfVectorizer()

X = convert_feature.fit_transform(text)
Y = df['label']

loaded_model.fit(X, Y)
pred = loaded_model.predict(X)

print(f"Model: {"BernoulliNB"}")
print(f"Accuracy: {accuracy_score(Y, pred)}")


loaded_model2.fit(X, Y)
pred = loaded_model2.predict(X)

print(f"Model: {"ComplementNB"}")
print(f"Accuracy: {accuracy_score(Y, pred)}")


loaded_model3.fit(X, Y)
pred = loaded_model3.predict(X)

print(f"Model: {"MultinomialNB"}")
print(f"Accuracy: {accuracy_score(Y, pred)}")

