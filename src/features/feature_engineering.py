import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
import yaml

# Load parameters from YAML
try:
    with open('params.yaml', 'r') as file:
        max_features = yaml.safe_load(file)['feature_engineering']['max_features']
except FileNotFoundError:
    raise Exception("Error: 'params.yaml' file not found.")
except KeyError as e:
    raise Exception(f"Error: Missing key {e} in 'params.yaml'.")
except yaml.YAMLError as e:
    raise Exception(f"Error parsing YAML file: {e}")

# Load the processed data
try:
    train_data = pd.read_csv('./data/processed/train_processed.csv')
    test_data = pd.read_csv('./data/processed/test_processed.csv')
except FileNotFoundError as e:
    raise Exception(f"CSV file not found: {e}")
except pd.errors.EmptyDataError:
    raise Exception("One or both CSV files are empty.")
except pd.errors.ParserError as e:
    raise Exception(f"Error parsing the CSV file: {e}")

# Handle missing values
try:
    train_data.fillna('', inplace=True)
    test_data.fillna('', inplace=True)
except Exception as e:
    raise Exception(f"Error handling missing values: {e}")

# Prepare training and testing data
try:
    X_train = train_data['content'].values
    y_train = train_data['sentiment'].values
    X_test = test_data['content'].values
    y_test = test_data['sentiment'].values
except KeyError as e:
    raise Exception(f"Error: Missing column {e} in the DataFrame.")
except Exception as e:
    raise Exception(f"Error extracting data: {e}")

# Apply Bag of Words (CountVectorizer)
try:
    vectorizer = CountVectorizer(max_features=max_features)
    X_train_bow = vectorizer.fit_transform(X_train)  # Fit and transform training data
    X_test_bow = vectorizer.transform(X_test)        # Transform test data using the same vectorizer
except ValueError as e:
    raise Exception(f"Vectorization error: {e}")

# Convert the transformed data to DataFrames and add labels
try:
    train_df = pd.DataFrame(X_train_bow.toarray())
    train_df['label'] = y_train
    test_df = pd.DataFrame(X_test_bow.toarray())
    test_df['label'] = y_test
except Exception as e:
    raise Exception(f"Error creating DataFrames: {e}")

# Create a new directory to save the transformed data
data_path = os.path.join("data", "features")

try:
    os.makedirs(data_path, exist_ok=True)  # Avoid error if directory already exists
except PermissionError:
    raise Exception(f"Permission denied: Unable to create directory '{data_path}'.")
except Exception as e:
    raise Exception(f"Error creating directory: {e}")

# Save the DataFrames to CSV files
try:
    train_df.to_csv(os.path.join(data_path, "train_bow.csv"), index=False)
    test_df.to_csv(os.path.join(data_path, "test_bow.csv"), index=False)
    print("Feature data saved successfully.")
except Exception as e:
    raise Exception(f"Error saving CSV files: {e}")
