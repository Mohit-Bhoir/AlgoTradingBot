#Import necessary libraries
import pandas as pd
import tpqoa as tp 

api = tp.tpqoa('oanda.cfg') #Add oanda Config file here
start = "2023-01-01"
end = "2025-12-31"


df = api.get_history(instrument="EUR_USD", start=start, end=end, granularity="M15", price="M")
df = df[['c']]
df.to_csv('../data/raw/data.csv')