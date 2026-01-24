
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")


class IterativeBase():
    ''' Base class for iterative (event-driven) backtesting of trading strategies.
    '''


    def __init__(self, symbol, start, end, amount, use_spread = True, data_path="data/processed/data_clean.csv"):
        '''
        Parameters
        ----------
        symbol: str
            ticker symbol (instrument) to be backtested
        start: str
            start date for data import
        end: str
            end date for data import
        amount: float
            initial amount to be invested per trade
        use_spread: boolean (default = True) 
            whether trading costs (bid-ask spread) are included
        data_path: str
            Path to the processed data csv
        '''
        self.symbol = symbol
        self.start = start
        self.end = end
        self.initial_balance = amount
        self.current_balance = amount
        self.units = 0
        self.trades = 0
        self.position = 0
        self.use_spread = use_spread
        self.data_path = data_path
        self.history = [] # To store equity curve data
        self.get_data()
    
    def get_data(self):
        ''' Imports the data from source.
        '''
        try:
            raw = pd.read_csv(self.data_path, parse_dates=["time"], index_col="time")
            # Filter by date if start/end logic is applied inside the CSV filter, 
            # but usually we just slice.
            # Convert index to datetime if it's not
            if not isinstance(raw.index, pd.DatetimeIndex):
                 raw.index = pd.to_datetime(raw.index)

            # Ensure we have data in range
            raw = raw.sort_index()
            if self.start is not None and self.end is not None:
                raw_filtered = raw.loc[self.start:self.end].copy()
                if raw_filtered.empty:
                     # Fallback if slice fails or date strings don't match index format
                     print("Warning: Data slice returned empty. Using full data.")
                     raw_filtered = raw.copy()
                raw = raw_filtered
            else:
                raw = raw.copy()

            # Ensure 'returns' exists (preprocess creates it usually)
            if "returns" not in raw.columns:
                 raw["returns"] = np.log(raw.price / raw.price.shift(1))
            
            # --- HARDCODED SPREAD LOGIC ---
            # User requirement: half spread cost = 0.00005 per unit.
            # This implies spread = 0.0001
            raw["spread"] = 0.0001 
            
            self.data = raw.dropna()
        except Exception as e:
            print(f"Error loading data: {e}")
            self.data = pd.DataFrame()


    def plot_data(self, cols = None):  
        ''' Plots the closing price for the symbol.
        '''
        if cols is None:
            cols = "price"
        self.data[cols].plot(figsize = (12, 8), title = self.symbol)
    
    def get_values(self, bar):
        ''' Returns the date, the price and the spread for the given bar.
        '''
        date = self.data.index[bar]
        price = round(self.data.price.iloc[bar], 5)
        spread = round(self.data.spread.iloc[bar], 5)
        return date, price, spread
    
    def print_current_balance(self, bar):
        ''' Prints out the current (cash) balance.
        '''
        date, price, spread = self.get_values(bar)
        print("{} | Current Balance: {}".format(date, round(self.current_balance, 2)))
        
    def buy_instrument(self, bar, units = None, amount = None):
        ''' Places and executes a buy order (market order).
        '''
        date, price, spread = self.get_values(bar)
        if self.use_spread:
            price += spread/2 # ask price
        if amount is not None: # use units if units are passed, otherwise calculate units
            units = int(amount / price)
        self.current_balance -= units * price # reduce cash balance by "purchase price"
        self.units += units
        self.trades += 1
        # print("{} |  Buying {} for {}".format(date, units, round(price, 5)))
        self.store_history(bar)
    
    def sell_instrument(self, bar, units = None, amount = None):
        ''' Places and executes a sell order (market order).
        '''
        date, price, spread = self.get_values(bar)
        if self.use_spread:
            price -= spread/2 # bid price
        if amount is not None: # use units if units are passed, otherwise calculate units
            units = int(amount / price)
        self.current_balance += units * price # increases cash balance by "purchase price"
        self.units -= units
        self.trades += 1
        # print("{} |  Selling {} for {}".format(date, units, round(price, 5)))
        self.store_history(bar)

    def store_history(self, bar):
        '''Stores the current net asset value and date'''
        date, price, _ = self.get_values(bar)
        nav = self.current_balance + self.units * price
        self.history.append({'time': date, 'equity': nav})

    
    def print_current_position_value(self, bar):
        ''' Prints out the current position value.
        '''
        date, price, spread = self.get_values(bar)
        cpv = self.units * price
        print("{} |  Current Position Value = {}".format(date, round(cpv, 2)))
    
    def print_current_nav(self, bar):
        ''' Prints out the current net asset value (nav).
        '''
        date, price, spread = self.get_values(bar)
        nav = self.current_balance + self.units * price
        print("{} |  Net Asset Value = {}".format(date, round(nav, 2)))
        
    def close_pos(self, bar):
        ''' Closes out a long or short position (go neutral).
        '''
        date, price, spread = self.get_values(bar)
        print(75 * "-")
        print("{} | +++ CLOSING FINAL POSITION +++".format(date))
        self.current_balance += self.units * price # closing final position (works with short and long!)
        self.current_balance -= (abs(self.units) * spread/2 * self.use_spread) # substract half-spread costs
        print("{} | closing position of {} for {}".format(date, self.units, price))
        self.units = 0 # setting position to neutral
        self.trades += 1
        perf = (self.current_balance - self.initial_balance) / self.initial_balance * 100
        self.print_current_balance(bar)
        print("{} | net performance (%) = {}".format(date, round(perf, 2) ))
        print("{} | number of trades executed = {}".format(date, self.trades))
        print(75 * "-")