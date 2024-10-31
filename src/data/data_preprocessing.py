import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer

try:
    train_data = pd.read_csv('./data/raw/train.csv')
    test_data = pd.read_csv('./data/raw/test.csv')
except FileNotFoundError as e:
    raise Exception(f"CSV file not found: {e}")
except pd.errors.EmptyDataError:
    raise Exception("One or both CSV files are empty.")
except pd.errors.ParserError as e:
    raise Exception(f"Error parsing the CSV file: {e}")

# Download required NLTK data
try:
    nltk.download('wordnet')
    nltk.download('stopwords')
except Exception as e:
    raise Exception(f"Error downloading NLTK data: {e}")

def lemmatization(text):
    try:
        lemmatizer = WordNetLemmatizer()
        text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
        return text
    except Exception as e:
        raise Exception(f"Lemmatization error: {e}")

def remove_stop_words(text):
    try:
        stop_words = set(stopwords.words("english"))
        return " ".join([word for word in str(text).split() if word not in stop_words])
    except Exception as e:
        raise Exception(f"Error removing stop words: {e}")

def removing_numbers(text):
    try:
        return ''.join([char for char in text if not char.isdigit()])
    except Exception as e:
        raise Exception(f"Error removing numbers: {e}")

def lower_case(text):
    try:
        return " ".join([word.lower() for word in text.split()])
    except Exception as e:
        raise Exception(f"Lowercase conversion error: {e}")

def removing_punctuations(text):
    try:
        text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
        text = re.sub('\s+', ' ', text).strip()  # Remove extra whitespace
        return text
    except Exception as e:
        raise Exception(f"Punctuation removal error: {e}")

def removing_urls(text):
    try:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub('', text)
    except Exception as e:
        raise Exception(f"URL removal error: {e}")

def remove_small_sentences(df):
    try:
        for i in range(len(df)):
            if len(df.text.iloc[i].split()) < 3:
                df.text.iloc[i] = np.nan
    except KeyError:
        raise Exception("The DataFrame does not have a 'text' column.")
    except Exception as e:
        raise Exception(f"Error removing small sentences: {e}")

def normalize_text(df):
    try:
        df.content = df.content.apply(lower_case)
        df.content = df.content.apply(remove_stop_words)
        df.content = df.content.apply(removing_numbers)
        df.content = df.content.apply(removing_punctuations)
        df.content = df.content.apply(removing_urls)
        df.content = df.content.apply(lemmatization)
        return df
    except KeyError:
        raise Exception("The DataFrame does not have a 'content' column.")
    except Exception as e:
        raise Exception(f"Error normalizing text: {e}")

try:
    train_processed_data = normalize_text(train_data)
    test_processed_data = normalize_text(test_data)
except Exception as e:
    print(f"Error processing data: {e}")

# Create a new directory to save processed data
data_path = os.path.join("data", "processed")

try:
    os.makedirs(data_path, exist_ok=True)
except PermissionError:
    raise Exception(f"Permission denied: Unable to create directory '{data_path}'.")
except Exception as e:
    raise Exception(f"Error creating directory: {e}")

try:
    train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
    test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
    print("Processed data saved successfully.")
except Exception as e:
    raise Exception(f"Error saving processed data: {e}")