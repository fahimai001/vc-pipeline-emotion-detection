import numpy as np
import pandas as pd
import pickle
import os
import json

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Load the model
try:
    with open('model.pkl', 'rb') as model_file:
        clf = pickle.load(model_file)
except FileNotFoundError:
    raise Exception("Error: 'model.pkl' file not found. Ensure the model is trained and saved correctly.")
except pickle.UnpicklingError as e:
    raise Exception(f"Error loading the model: {e}")
except Exception as e:
    raise Exception(f"Unexpected error during model loading: {e}")

# Load the test data
try:
    test_data = pd.read_csv('./data/features/test_bow.csv')
except FileNotFoundError as e:
    raise Exception(f"Test CSV file not found: {e}")
except pd.errors.EmptyDataError:
    raise Exception("The test CSV file is empty.")
except pd.errors.ParserError as e:
    raise Exception(f"Error parsing the CSV file: {e}")

# Extract features and labels from the test data
try:
    X_test = test_data.iloc[:, 0:-1].values
    y_test = test_data.iloc[:, -1].values
except IndexError as e:
    raise Exception(f"Error accessing data: {e}. Ensure the CSV contains both features and labels.")
except Exception as e:
    raise Exception(f"Unexpected error during data extraction: {e}")

# Make predictions
try:
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
except ValueError as e:
    raise Exception(f"Prediction error: {e}. Check input data dimensions.")
except Exception as e:
    raise Exception(f"Unexpected error during prediction: {e}")

# Calculate evaluation metrics
try:
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
except ValueError as e:
    raise Exception(f"Error calculating metrics: {e}. Check the input labels.")
except Exception as e:
    raise Exception(f"Unexpected error during metric calculation: {e}")

# Store metrics in a dictionary
metrics_dic = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'auc': auc
}

# Save metrics to a JSON file
try:
    with open('metrics.json', 'w') as file:
        json.dump(metrics_dic, file, indent=4)
    print("Metrics saved successfully to 'metrics.json'.")
except PermissionError:
    raise Exception("Permission denied: Unable to save 'metrics.json'.")
except Exception as e:
    raise Exception(f"Error saving metrics: {e}")
