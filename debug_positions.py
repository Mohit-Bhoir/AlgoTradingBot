import tpqoa
import os
import pandas as pd

conf_path = "API CONNECT/oanda.cfg"
if os.path.exists(conf_path):
    api = tpqoa.tpqoa(conf_path)
    print("Positions:", api.get_positions())
    
    # Also verify current prices/setup
    print("Account Summary:", api.get_account_summary())
