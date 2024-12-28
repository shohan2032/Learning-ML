!pip install nltk
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab') # Download the punkt_tab resource

# Initialize stop words and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

import pandas as pd
import matplotlib.pyplot as plt
import string
import re

# Load dataset
data = pd.read_csv('/content/eng.csv')

# Dataset Overview
num_rows = data.shape[0]
num_features = data.shape[1]
missing_values = data.isnull().sum().sum()
class_distribution = data[['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']].sum()

# Text length and word count distribution
data['text_length'] = data['text'].apply(len)
plt.figure(figsize=(8, 5))
plt.hist(data['text_length'], bins=20, color='blue', alpha=0.7, edgecolor='black')
plt.title('Histogram of Text Lengths (Dataset)')
plt.xlabel('Text Length (characters)')
plt.ylabel('Frequency')
plt.savefig('updated_text_length_histogram.png')

data['word_count'] = data['text'].apply(lambda x: len(x.split()))
plt.figure(figsize=(8, 5))
plt.hist(data['word_count'], bins=20, color='green', alpha=0.7, edgecolor='black')
plt.title('Histogram of Word Counts (Dataset)')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.savefig('updated_word_count_histogram.png')

# Label distribution
plt.figure(figsize=(8, 5))
class_distribution.plot(kind='bar', color='orange', edgecolor='black')
plt.title('Label Distribution (Dataset)')
plt.xlabel('Emotions')
plt.ylabel('Frequency')
plt.savefig('updated_label_distribution.png')

# Preprocessing
def preprocess_text(text):
    # 1. Convert to lowercase
    text = text.lower()
    # 2. Remove punctuation and numerical values
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 3. Remove trailing spaces
    text = text.strip()
    # 4. Tokenize text
    tokens = word_tokenize(text)
    # 5. Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    # 6. Lemmatize tokens
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # 7. Join tokens back to string
    return ' '.join(tokens)

data['processed_text'] = data['text'].apply(preprocess_text)

# Save preprocessed data
data.to_csv('updated_preprocessed_data.csv', index=False)
