import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')

import re


# Download necessary NLTK resources
nltk.download('punkt')


def preprocess_text(text):
    # 1. Convert all characters to lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # 3. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 4. Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])

    text = ''.join(char if char.isalpha() or char.isspace() else '' for char in text)
    # Optional: Remove extra spaces
    text = ' '.join(text.split())

    words = text.split()
    new_text = []
    for word in words:
        # Check if the word is in the abbreviation dictionary
        word_lower = word.lower()
        slangs = pd.read_csv("Abbreviations and Slang.csv")
        # Convert DataFrame to dictionary
        abbreviation_dict = pd.Series(slangs.Text.values, index=slangs.Abbreviations).to_dict()
        if word_lower in abbreviation_dict:
            # Replace it with the full form
            new_text.append(abbreviation_dict[word_lower])
        else:
            # Keep the word as is
            new_text.append(word)
    text = ' '.join(new_text)

    tokens = word_tokenize(text)
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    text = ' '.join(stemmed_tokens)

    return text
