from dotenv import load_dotenv
import configparser
import pandas as pd
import tpqoa
import yaml
import os

# Load params
params = yaml.safe_load(open("params.yaml"))['data_fetch']

def fetch_data(config_path, instrument, start, end, granularity, output_path):
    # Ensure config path is handled correctly relative to execution
    def get_oanda_credentials():
        load_dotenv()
        account_id = os.environ.get("OANDA_ACCOUNT_ID")
        access_token = os.environ.get("OANDA_ACCESS_TOKEN")
        account_type = os.environ.get("OANDA_ACCOUNT_TYPE")
        if account_id and access_token and account_type:
            config = configparser.ConfigParser()
            config['oanda'] = {
                'account_id': account_id,
                'access_token': access_token,
                'account_type': account_type
            }
            with open("oanda_temp.cfg", "w") as f:
                config.write(f)
            return "oanda_temp.cfg"
        return None

    conf_path = get_oanda_credentials()
    if not conf_path:
        raise Exception("Oanda credentials not found in .env file.")
    api = tpqoa.tpqoa(conf_path)
    
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
