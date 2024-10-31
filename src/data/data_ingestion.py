import numpy as np
import pandas as pd
import yaml
import os
from sklearn.model_selection import train_test_split

def load_params(params_path):
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
            test_size = params['data_ingestion']['test_size']
        return test_size
    except FileNotFoundError:
        raise Exception(f"Error: '{params_path}' not found.")
    except KeyError as e:
        raise Exception(f"Missing key {e} in the YAML file.")
    except yaml.YAMLError as e:
        raise Exception(f"YAML parsing error: {e}")

def read_data(url):
    try:
        df = pd.read_csv(url)
        return df
    except pd.errors.EmptyDataError:
        raise Exception("The CSV file is empty.")
    except pd.errors.ParserError:
        raise Exception("Error while parsing the CSV file.")
    except Exception as e:
        raise Exception(f"Failed to read data from URL: {e}")

def process_data(df):
    try:
        # Ensure 'tweet_id' column exists before dropping
        if 'tweet_id' in df.columns:
            df.drop(columns=['tweet_id'], inplace=True)
        else:
            raise KeyError("'tweet_id' column not found in DataFrame.")
        
        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])]
        final_df = final_df.copy()
        final_df['sentiment'].replace({'happiness': 1, 'sadness': 0}, inplace=True)

        return final_df
    except KeyError as e:
        raise Exception(f"Data processing error: {e}")

def save_data(data_path, train_data, test_data):
    try:
        os.makedirs(data_path, exist_ok=True)  # Avoids error if directory already exists
        train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)
    except PermissionError:
        raise Exception(f"Permission denied: Unable to create directory '{data_path}'.")
    except Exception as e:
        raise Exception(f"Error saving data: {e}")

def main():
    try:
        test_size = load_params('params.yaml')
        df = read_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
        final_df = process_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        data_path = os.path.join("data", "raw")
        save_data(data_path, train_data, test_data)
        print("Data processing and saving completed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
