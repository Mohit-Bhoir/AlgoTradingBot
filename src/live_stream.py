import pandas as pd
import numpy as np
import tpqoa
from datetime import datetime, timedelta, timezone
import time
import pickle
import warnings
import yaml
import os
import xgboost as xgb
import json
import configparser
import streamlit as st

warnings.filterwarnings("ignore", category=SyntaxWarning)

class MLTrader(tpqoa.tpqoa):
    def __init__(self, conf_file, instrument, bar_length, units, model_path, lags):
        super().__init__(conf_file)
        self.instrument = instrument
        self.bar_length = pd.to_timedelta(bar_length)
        self.tick_data = pd.DataFrame()
        self.raw_data = None
        self.data = None 
        self.last_bar = None
        self.units = units
        self.position = 0
        self.profits = []
        
        # ML specific
        self.lags = lags
        
        # Load Model
        self.model = load_model(model_path)
        if self.model is None:
            # Disable trading or use random predictions for demo
            pass

        # Data versioning path
        self.stream_data_path = os.path.join("data", "streamed", "streamed_data.csv")
        self.trades_path = os.path.join("data", "streamed", "trades.json")
        self.tick_log_path = os.path.join("data", "streamed", "latest_tick.json")
        os.makedirs(os.path.dirname(self.stream_data_path), exist_ok=True)
    
    def get_most_recent(self, days = 5):
        print("Getting most recent data...")
        while True:
            time.sleep(2)
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            now = now - timedelta(microseconds = now.microsecond)
            past = now - timedelta(days = days)
            
            try:
                # Fetch history
                df = self.get_history(instrument = self.instrument, start = past, end = now,
                                   granularity = "S5", price = "M", localize = False).c.dropna().to_frame()
                df.rename(columns = {"c":self.instrument}, inplace = True)
                df = df.resample(self.bar_length, label = "right").last().dropna().iloc[:-1]
                self.raw_data = df.copy()
                self.last_bar = self.raw_data.index[-1]
                
                if pd.to_datetime(datetime.now(timezone.utc)) - self.last_bar < self.bar_length:
                    break
            except Exception as e:
                print(f"Error getting history: {e}")
                time.sleep(5)

        print("Initial data fetched.")
                
    def on_success(self, time, bid, ask):
        recent_tick = pd.to_datetime(time)
        mid_price = (ask + bid)/2
        
        # Save latest tick to JSON for UI
        try:
            with open(self.tick_log_path, 'w') as f:
                json.dump({
                    "time": str(recent_tick),
                    "bid": bid, 
                    "ask": ask, 
                    "price": mid_price,
                    "position": self.position,
                    "cum_pnl": sum(self.profits)
                }, f)
        except Exception:
            pass # Ignore write collisions

        # Print tick for log capture
        print(f"TICK: {recent_tick} | Price: {mid_price:.5f}")

        df = pd.DataFrame({self.instrument: mid_price}, 
                          index = [recent_tick])
        self.tick_data = pd.concat([self.tick_data, df])
        
        if recent_tick - self.last_bar > self.bar_length:
            print(f"\nBar completed at {recent_tick}")
            self.resample_and_join()
            self.define_strategy()
            self.execute_trades()
            self.save_stream_data()
        else:
             # Optional: print progress or just keep silent to avoid log spam
             pass
    
    def resample_and_join(self):
        self.raw_data = pd.concat([self.raw_data, self.tick_data.resample(self.bar_length, 
                                                                          label="right").last().ffill().iloc[:-1]])
        self.tick_data = self.tick_data.iloc[-1:]
        self.last_bar = self.raw_data.index[-1]
    
    def define_strategy(self):
        df = self.raw_data.copy()
        
        # Rename for consistency
        df = df.rename(columns={self.instrument: 'price'})
        
        # Feature Engineering
        df["returns"] = np.log(df['price'] / df['price'].shift())
        
        features = []
        for lag in range(1, self.lags + 1):
            col = f"returns_lag_{lag}"
            df[col] = df["returns"].shift(lag)
            features.append(col)
        
        df.dropna(inplace=True)
        
        if not df.empty:
            # Predict
            try:
                # Extract last row features
                X = df.iloc[[-1]][features]
                pred = self.model.predict(X)[0]
                
                # Convert prediction to position
                # 1 -> Long, 0 -> Short (as per train.py logic direction=1 (up) or 0 (down))
                # Strategy: Go Long if 1, Go Short if 0
                df["position"] = np.where(pred == 1, 1, -1)
                self.data = df.copy()
                print(f"Prediction: {pred} | Strategy Position: {df['position'].iloc[-1]}")
            except Exception as e:
                print(f"Prediction error: {e}")
            
    def sync_position(self):
        try:
            positions = self.get_positions()
            units = 0
            for pos in positions:
                if pos.get('instrument') == self.instrument:
                    long_units = int(float(pos.get('long', {}).get('units', 0)))
                    short_units = int(float(pos.get('short', {}).get('units', 0)))
                    units = long_units - short_units
                    break
            if units > 0:
                self.position = 1
            elif units < 0:
                self.position = -1
            else:
                self.position = 0
            print(f"Synced Position: {self.position} (Net Units: {units})")
        except Exception as e:
            print(f"Error syncing position: {e}")

    def execute_trades(self):
        if self.data is None or self.data.empty:
            return

        # Always sync with Oanda before deciding
        self.sync_position()

        try:
            current_pos = self.data["position"].iloc[-1]
            
            if current_pos == 1:
                if self.position == 0:
                    order = self.create_order(self.instrument, self.units, suppress = True, ret = True)
                    self.report_trade(order, "GOING LONG")
                elif self.position == -1:
                    order = self.create_order(self.instrument, self.units * 2, suppress = True, ret = True) 
                    self.report_trade(order, "GOING LONG")
                self.position = 1
            elif current_pos == -1: 
                if self.position == 0:
                    order = self.create_order(self.instrument, -self.units, suppress = True, ret = True)
                    self.report_trade(order, "GOING SHORT")
                elif self.position == 1:
                    order = self.create_order(self.instrument, -self.units * 2, suppress = True, ret = True)
                    self.report_trade(order, "GOING SHORT")
                self.position = -1
            else:
                 print(f"Holding position: {self.position}. No trade needed.")
        except Exception as e:
            print(f"Trade execution error: {e}")
    
    def report_trade(self, order, going):
        time = order["time"]
        units = order["units"]
        price = order["price"]
        pl = float(order["pl"])
        self.profits.append(pl)
        cumpl = sum(self.profits)
        print("\n" + 100* "-")
        print("{} | {}".format(time, going))
        print("{} | units = {} | price = {} | P&L = {} | Cum P&L = {}".format(time, units, price, pl, cumpl))
        print(100 * "-" + "\n")
        
        # Log trade to JSON for UI
        trade_record = {
            "time": str(time),
            "type": going,
            "units": units,
            "price": price,
            "pl": pl,
            "cum_pl": cumpl
        }
        
        trades = []
        if os.path.exists(self.trades_path):
             try:
                 with open(self.trades_path, "r") as f:
                     trades = json.load(f)
             except: pass
        
        trades.append(trade_record)
        with open(self.trades_path, "w") as f:
            json.dump(trades, f, indent=4)
        
    def save_stream_data(self):
        # Save the new bar data for DVC versioning
        if self.raw_data is not None:
             try:
                 # Last completed bar
                 last_bar = self.raw_data.iloc[[-1]].copy()
                 last_bar.rename(columns={self.instrument: 'c'}, inplace=True)
                 last_bar.index.name = 'time'
                 
                 header = not os.path.exists(self.stream_data_path)
                 last_bar.to_csv(self.stream_data_path, mode='a', header=header)
                 print(f"Data saved to {self.stream_data_path}")
             except Exception as e:
                 print(f"Error saving stream data: {e}")

def get_oanda_config():
    # Try to read from environment variables first
    account_id = os.environ.get("OANDA_ACCOUNT_ID")
    access_token = os.environ.get("OANDA_ACCESS_TOKEN")
    account_type = os.environ.get("OANDA_ACCOUNT_TYPE")
    if account_id and access_token and account_type:
        return {
            "account_id": account_id,
            "access_token": access_token,
            "account_type": account_type
        }
    # Only show error if not running in demo mode
    if os.environ.get("DEMO_MODE", "0") == "1":
        st.info("Running in demo mode. No Oanda credentials loaded.")
        return None
    st.error("Config file not found at API CONNECT/oanda.cfg and no environment variables set.")
    return None

def write_temp_oanda_cfg():
    """Write a temporary oanda.cfg file from environment variables if not present."""
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

def load_model(model_path):
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    else:
        print("Model file not found. Running in demo mode.")
        # Return a dummy model or None
        return None

def run_bot(units=100000):
    if not os.path.exists("params.yaml"):
        print("params.yaml not found.")
        return

    params = yaml.safe_load(open("params.yaml"))
    
    granularity = params['data_fetch']['granularity']
    # Simple mapping
    if granularity == "M15":
        bar_len = "15min"
    elif granularity == "M1":
        bar_len = "1min"
    elif granularity == "H1":
        bar_len = "1H"
    else:
        bar_len = "15min"

    conf_file = write_temp_oanda_cfg() or "API CONNECT/oanda.cfg"
    if not os.path.exists(conf_file):
        print("Oanda credentials not found. Please set them as Streamlit secrets for live account data.")
        # Optionally, run in demo mode or exit
    trader = MLTrader(
        conf_file=conf_file,
        instrument=params['data_fetch']['instrument'],
        bar_length=bar_len,
        units=units,
        model_path=params['train']['model_path'],
        lags=params['preprocess']['lags']
    )
    
    trader.get_most_recent()
    trader.stream_data(trader.instrument)

if __name__ == "__main__":
    # Allow units to be passed via env var for dynamic control
    units = int(os.environ.get("TRADING_UNITS", 100000))
    run_bot(units=units)

