import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import GradientBoostingClassifier
import yaml

# Load parameters from YAML file
try:
    with open('params.yaml', 'r') as file:
        params = yaml.safe_load(file)['model_building']
except FileNotFoundError:
    raise Exception("Error: 'params.yaml' file not found.")
except KeyError as e:
    raise Exception(f"Error: Missing key {e} in 'params.yaml'.")
except yaml.YAMLError as e:
    raise Exception(f"Error parsing YAML file: {e}")

# Load training data
try:
    train_data = pd.read_csv('./data/features/train_bow.csv')
except FileNotFoundError as e:
    raise Exception(f"CSV file not found: {e}")
except pd.errors.EmptyDataError:
    raise Exception("The train CSV file is empty.")
except pd.errors.ParserError as e:
    raise Exception(f"Error parsing the CSV file: {e}")

# Prepare features and labels
try:
    X_train = train_data.iloc[:, 0:-1].values
    y_train = train_data.iloc[:, -1].values
except IndexError as e:
    raise Exception(f"Error accessing data: {e}. Ensure the CSV contains both features and labels.")
except Exception as e:
    raise Exception(f"Error extracting features and labels: {e}")

# Initialize and train the model
try:
    clf = GradientBoostingClassifier(
        n_estimators=params['n_estimators'],
        learning_rate=params['learning_rate']
    )
    clf.fit(X_train, y_train)
except KeyError as e:
    raise Exception(f"Missing parameter {e} in YAML file.")
except ValueError as e:
    raise Exception(f"Model training error: {e}")
except Exception as e:
    raise Exception(f"Unexpected error during model training: {e}")

# Save the model using pickle
try:
    with open('model.pkl', 'wb') as model_file:
        pickle.dump(clf, model_file)
    print("Model saved successfully as 'model.pkl'.")
except PermissionError:
    raise Exception("Permission denied: Unable to save 'model.pkl'.")
except Exception as e:
    raise Exception(f"Error saving the model: {e}")
