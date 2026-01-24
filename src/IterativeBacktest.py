
from IterativeBase import *

class IterativeBacktest(IterativeBase):
    ''' Class for iterative (event-driven) backtesting of trading strategies.
    '''

    # helper method
    def go_long(self, bar, units = None, amount = None):
        if self.position == -1:
            self.buy_instrument(bar, units = -self.units) # if short position, go neutral first
        if units:
            self.buy_instrument(bar, units = units)
        elif amount:
            if amount == "all":
                amount = self.current_balance
            self.buy_instrument(bar, amount = amount) # go long

    # helper method
    def go_short(self, bar, units = None, amount = None):
        if self.position == 1:
            self.sell_instrument(bar, units = self.units) # if long position, go neutral first
        if units:
            self.sell_instrument(bar, units = units)
        elif amount:
            if amount == "all":
                amount = self.current_balance
            self.sell_instrument(bar, amount = amount) # go short

    def test_sma_strategy(self, SMA_S, SMA_L):
        ''' 
        Backtests an SMA crossover strategy with SMA_S (short) and SMA_L (long).
        
        Parameters
        ----------
        SMA_S: int
            moving window in bars (e.g. days) for shorter SMA
        SMA_L: int
            moving window in bars (e.g. days) for longer SMA
        '''
        
        # nice printout
        stm = "Testing SMA strategy | {} | SMA_S = {} & SMA_L = {}".format(self.symbol, SMA_S, SMA_L)
        print("-" * 75)
        print(stm)
        print("-" * 75)
        
        # reset 
        self.position = 0  # initial neutral position
        self.trades = 0  # no trades yet
        self.current_balance = self.initial_balance  # reset initial capital
        self.get_data() # reset dataset
        
        # prepare data
        self.data["SMA_S"] = self.data["price"].rolling(SMA_S).mean()
        self.data["SMA_L"] = self.data["price"].rolling(SMA_L).mean()
        self.data.dropna(inplace = True)

        # sma crossover strategy
        for bar in range(len(self.data)-1): # all bars (except the last bar)
            date, price, spread = self.get_values(bar)
            current_hour = date.hour

            if 12 <= current_hour < 16:
                if self.data["SMA_S"].iloc[bar] > self.data["SMA_L"].iloc[bar]: # signal to go long
                    if self.position in [0, -1]:
                        self.go_long(bar, amount = "all") # go long with full amount
                        self.position = 1  # long position
                elif self.data["SMA_S"].iloc[bar] < self.data["SMA_L"].iloc[bar]: # signal to go short
                    if self.position in [0, 1]:
                        self.go_short(bar, amount = "all") # go short with full amount
                        self.position = -1 # short position
            else:
                if self.position != 0:
                    self.close_pos(bar) 
                    self.position = 0
            
            self.store_history(bar)
        self.close_pos(bar+1) # close position at the last bar
        
        
    def test_con_strategy(self, window = 1):
        ''' 
        Backtests a simple contrarian strategy.
        
        Parameters
        ----------
        window: int
            time window (number of bars) to be considered for the strategy.
        '''
        
        # nice printout
        stm = "Testing Contrarian strategy | {} | Window = {}".format(self.symbol, window)
        print("-" * 75)
        print(stm)
        print("-" * 75)
        
        # reset 
        self.position = 0  # initial neutral position
        self.trades = 0  # no trades yet
        self.current_balance = self.initial_balance  # reset initial capital
        self.get_data() # reset dataset
        
        # prepare data
        self.data["rolling_returns"] = self.data["returns"].rolling(window).mean()
        self.data.dropna(inplace = True)
        
        # Contrarian strategy
        for bar in range(len(self.data)-1): # all bars (except the last bar)
            date, price, spread = self.get_values(bar)
            current_hour = date.hour
            
            if 12 <= current_hour < 16:
                if self.data["rolling_returns"].iloc[bar] <= 0: #signal to go long
                    if self.position in [0, -1]:
                        self.go_long(bar, amount = "all") # go long with full amount
                        self.position = 1  # long position
                elif self.data["rolling_returns"].iloc[bar] > 0: #signal to go short
                    if self.position in [0, 1]:
                        self.go_short(bar, amount = "all") # go short with full amount
                        self.position = -1 # short position
            else:
                if self.position != 0:
                    self.close_pos(bar) 
                    self.position = 0
            
            self.store_history(bar)
        self.close_pos(bar+1) # close position at the last bar
        
        
    def test_boll_strategy(self, SMA, dev):
        ''' 
        Backtests a Bollinger Bands mean-reversion strategy.
        
        Parameters
        ----------
        SMA: int
            moving window in bars (e.g. days) for simple moving average.
        dev: int
            distance for Lower/Upper Bands in Standard Deviation units
        '''
        
        # nice printout
        stm = "Testing Bollinger Bands Strategy | {} | SMA = {} & dev = {}".format(self.symbol, SMA, dev)
        print("-" * 75)
        print(stm)
        print("-" * 75)
        
        # reset 
        self.position = 0  # initial neutral position
        self.trades = 0  # no trades yet
        self.current_balance = self.initial_balance  # reset initial capital
        self.get_data() # reset dataset
        
        # prepare data
        self.data["SMA"] = self.data["price"].rolling(SMA).mean()
        self.data["Lower"] = self.data["SMA"] - self.data["price"].rolling(SMA).std() * dev
        self.data["Upper"] = self.data["SMA"] + self.data["price"].rolling(SMA).std() * dev
        self.data.dropna(inplace = True) 
        
        # Bollinger strategy
        for bar in range(len(self.data)-1): # all bars (except the last bar)
            date, price, spread = self.get_values(bar)
            current_hour = date.hour

            if 12 <= current_hour < 16:
                if self.position == 0: # when neutral
                    if self.data["price"].iloc[bar] < self.data["Lower"].iloc[bar]: # signal to go long
                        self.go_long(bar, amount = "all") # go long with full amount
                        self.position = 1  # long position
                    elif self.data["price"].iloc[bar] > self.data["Upper"].iloc[bar]: # signal to go Short
                        self.go_short(bar, amount = "all") # go short with full amount
                        self.position = -1 # short position
                elif self.position == 1: # when long
                    if self.data["price"].iloc[bar] > self.data["SMA"].iloc[bar]:
                        if self.data["price"].iloc[bar] > self.data["Upper"].iloc[bar]: # signal to go short
                            self.go_short(bar, amount = "all") # go short with full amount
                            self.position = -1 # short position
                        else:
                            self.sell_instrument(bar, units = self.units) # go neutral
                            self.position = 0
                elif self.position == -1: # when short
                    if self.data["price"].iloc[bar] < self.data["SMA"].iloc[bar]:
                        if self.data["price"].iloc[bar] < self.data["Lower"].iloc[bar]: # signal to go long
                            self.go_long(bar, amount = "all") # go long with full amount
                            self.position = 1 # long position
            else:
                if self.position != 0:
                    self.close_pos(bar) 
                    self.position = 0

            self.store_history(bar)
        self.close_pos(bar+1) # close position at the last bar

    def test_xgboost_strategy(self, model, preparation_func=None):
        ''' 
        Backtests the XGBoost strategy.
        Iteration logic:
        - Only trade between 12 and 16 UTC.
        - Position rules:
             1 (Long): prediction == 1
            -1 (Short): prediction == 0 (if binary classification is used for Direction)
        '''
        stm = "Testing XGBoost Strategy | {} | Time Filter: 12-16 UTC".format(self.symbol)
        print("-" * 75)
        print(stm)
        print("-" * 75)
        
        # Reset
        self.position = 0
        self.trades = 0
        self.current_balance = self.initial_balance
        self.get_data() # Reset data
        
        # --- FEATURE ENGINEERING ---
        feature_cols = [col for col in self.data.columns if "lag" in col]
        
        self.data.dropna(inplace=True)

        for bar in range(len(self.data)-1):
            date, price, spread = self.get_values(bar)

            current_hour = date.hour
            
            if 12 <= current_hour < 16:
                # Active trading window
                row = self.data.iloc[bar]
                # Careful: The pipeline "Dropped" columns like 'time', 'price', 'returns'.
                # 'feature_cols' should strictly be the LAGS.
                features = row[feature_cols].values.reshape(1, -1)
                
                try:
                    pred = model.predict(features)[0] # 0 or 1
                except:
                    # e.g. column mismatch
                    continue
                
                # Logic: 1 -> Long, 0 -> Short
                if pred == 1:
                    if self.position in [0, -1]:
                        self.go_long(bar, amount="all")
                        self.position = 1
                elif pred == 0:
                    if self.position in [0, 1]:
                        self.go_short(bar, amount="all")
                        self.position = -1
            else:
                # Outside Window -> Close Positions
                if self.position != 0:
                    self.close_pos(bar) 
                    self.position = 0
            
            self.store_history(bar)

        self.close_pos(bar+1)

     