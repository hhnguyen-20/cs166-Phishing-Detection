import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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


# Define a function to check if a string contains English words
def contains_english_words(text):
    # Regular expression pattern to match English words
    pattern = r'\b[a-zA-Z]+\b'  # Matches one or more alphabetic characters

    # Search for the pattern in the text
    return bool(re.search(pattern, text))


# Load the dataset
data = pd.read_csv('/Users/muyilin/cs166-Phishing-Detection/email_scrape/CEAS_08.csv')

# Extract only email body and label
data = data[['body', 'label']]

# Apply preprocessing on body
data['preprocessed_body'] = data['body'].apply(preprocess_text)
# Remove any row whose body does not contain anymore words after removing stopwords
data = data[data['body'].apply(contains_english_words)]
# Reset index
data.reset_index(drop=True, inplace=True)
# Replace the 'body' column with the preprocessed version
data['body'] = data['preprocessed_body']
# Drop the 'preprocessed_body' column
data.drop(columns=['preprocessed_body'], inplace=True)

# Save dataframe as csv
data.to_csv('preprocessed_train_dataset.csv', index=False)
