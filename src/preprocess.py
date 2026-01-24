import pandas as pd 
import sys
import yaml
import os
import numpy as np

#Load parameters from params.yaml
params = yaml.safe_load(open("params.yaml"))['preprocess']

def preprocess_data(input_path, output_path):
    # Load the raw data
    data = pd.read_csv(input_path, parse_dates=['time'], index_col='time')
    data = data.rename(columns={'c':'price'})
    data['returns'] = np.log(data['price'] / data['price'].shift(1))
    data['direction'] = np.where(data['returns'] > 0, 1, 0)
    
    features = []  
    for lag in range(1, params['lags'] + 1):
        data[f'returns_lag_{lag}'] = data['returns'].shift(lag)
        features.append(f'returns_lag_{lag}')
    data.dropna(inplace=True)


    # Save the cleaned data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path)
    print(f"Preprocessed data saved to {output_path}")

if __name__ == "__main__":
    preprocess_data(params['input'], params['output'])