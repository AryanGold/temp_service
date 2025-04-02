import os
import pandas as pd
import numpy as np

def get_parent_directory():
    # Get the full path of the current script including the filename
    script_path = os.path.realpath(__file__)
    # Extract the directory part from the full path
    directory = os.path.dirname(script_path)
    parts = directory.split(os.sep)
    if 'volatility_models' in parts:
        index = parts.index('volatility_models')
        parent_directory = os.sep.join(parts[:index])
    else:
        parent_directory = directory  # fallback if 'volatility_models' not found
    return parent_directory


def make_json_serializable2(df):
    #json_str = df.to_json(date_format='iso', orient='split')
    #df_new = pd.read_json(json_str, orient='split')

    # Convert datetime columns to string

    for col in df.select_dtypes(include=['datetime64', 'datetime64[ns]']).columns:
        df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Convert any other non-serializable types here
    # Example: Convert numpy integers to Python integers
    for col in df.select_dtypes(include=[np.number]).columns:
        if np.issubdtype(df[col].dtype, np.integer):
            df[col] = df[col].astype(int)
        elif np.issubdtype(df[col].dtype, np.floating):
            df[col] = df[col].astype(float)

    return df



def make_json_serializable(df):
    # Store original datetime columns and their formats
    datetime_cols = df.select_dtypes(include=['datetime64', 'datetime64[ns]'])

    # Convert datetime columns to string for JSON serialization
    for col in datetime_cols.columns:
        df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Replace NaN, inf, and -inf in float columns to make them JSON serializable
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].apply(lambda x: None if np.isnan(x) or np.isinf(x) or x == np.inf or x == -np.inf else x)

    # Convert DataFrame to JSON and back to DataFrame to simulate serialization
    json_str = df.to_json()
    df = pd.read_json(json_str)

    # Convert datetime columns back to datetime and format them
    for col in datetime_cols.columns:
        df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d %H:%M:%S')


    return df.fillna('')

