#Import necessary libraries
import pandas as pd
import tpqoa as tp
from dotenv import load_dotenv
import os
import configparser

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
api = tp.tpqoa(conf_path)
start = "2023-01-01"
end = "2025-12-31"


df = api.get_history(instrument="EUR_USD", start=start, end=end, granularity="M15", price="M")
df = df[['c']]
df.to_csv('../data/raw/data.csv')