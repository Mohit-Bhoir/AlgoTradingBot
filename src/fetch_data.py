import pandas as pd
import tpqoa
import yaml
import os

# Load params
params = yaml.safe_load(open("params.yaml"))['data_fetch']

def fetch_data(config_path, instrument, start, end, granularity, output_path):
    # Ensure config path is handled correctly relative to execution
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    api = tpqoa.tpqoa(config_path)
    
    print(f"Fetching {instrument} from {start} to {end}...")
    df = api.get_history(instrument=instrument, start=start, end=end, granularity=granularity, price="M")
    
    if df.empty:
        raise ValueError("No data fetched from Oanda API")

    df = df[['c']]
    df = df.rename(columns={'c': 'price'}) # Rename here to avoid renaming in preprocess if desired, or keep as 'c'
    # Keeping it as 'c' to match existing preprocess logic which renames 'c' -> 'price'
    # Actually, let's keep it raw as the API returns it, but ensure column 'c' is there.
    # The existing preprocess.py does: data = data.rename(columns={'c':'price'}). 
    # So we keep column name as 'c' in the raw file to avoid breaking preprocess.py.
    df = df.rename(columns={'price': 'c'}) if 'price' in df.columns else df # basic safety
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)
    print(f"Raw data saved to {output_path}")

if __name__ == "__main__":
    fetch_data(
        params['config_path'],
        params['instrument'],
        params['start'],
        params['end'],
        params['granularity'],
        params['output']
    )
