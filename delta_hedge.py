import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

def get_close(ticker):
    df = pd.read_csv(f'./prices/{ticker}.csv', index_col=0, parse_dates=True)
    close = df['Adj Close']
    return close

def european_vanilla_put(S, K, T, r, sigma):
    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: risk free interest rate
    #sigma: volatility of underlying asset

    d1 = (np.log(S/K) + (r + 0.5*sigma**2) * T) / (sigma*np.sqrt(T))
    d2 = (np.log(S/K) + (r - 0.5*sigma**2) * T) / (sigma*np.sqrt(T))
    
    put = (K * np.exp(-r*T) * norm.cdf(-d2, 0.0, 1.0) - S * norm.cdf(-d1, 0.0, 1.0))
    
    return put

class Account:
    def __init__(self, name, inifund):
        self._name = name
        self._inifund = inifund
        self._cash = 0
        self._stock_positions = 0
        self._put_positions = 0
        self._put_strike_price = 0     
        self._dashboard = pd.DataFrame(columns=('Cash','Stock','Options','NAV'), dtype=float)
        
    def deposit(self, amount):
        self._cash += amount

    def long_stock(self, price, share):
        self._cash -= price*share
        self._stock_positions += share

    def long_put_options(self, price, strike):
        share = self._stock_positions
        self._cash -= price*share
        self._put_positions = share
        self._put_strike_price = strike

    def close_put_options(self, price):
        self._cash += price*self._put_positions
        self._put_positions = 0
        self._put_strike_price = None

    def net_asset_value(self, last, tradingdaysleft, sigma):
        cash_val = self._cash
        stock_val = self._stock_positions*last
        put_val = self._put_positions*european_vanilla_put(
                    S=last,
                    K=self._put_strike_price,
                    T=tradingdaysleft/252,
                    r=.025,
                    sigma=sigma)
        nav = cash_val + stock_val + put_val
        return cash_val, stock_val, put_val, nav
    
    def settlement(self, at, *args):
        vals = self.net_asset_value(*args)
        self._dashboard.loc[at] = vals
        return vals

class Strategy:
    def _set_args(self, kwargs):
        if not hasattr(self, '_args'): self._args = {}
        for key,val in kwargs.items(): 
            setattr(self, f'_{key}', val)
            self._args = {**self._args, **kwargs}
    
    def __init__(self, **kwargs):
        self._set_args(kwargs)
        self._acc = Account('Active', inifund=1e7)
        self._acc_bm = Account('Benchmark', inifund=1e7)
        self._stockprices = get_close(self._stock_ticker)
        self._vix = get_close('^VIX')

    def run(self, **kwargs):
        self._set_args(kwargs)
        timeline = self._stockprices.loc[self._start:].index        
        for i,today in enumerate(timeline):
            # daily use
            last = self._stockprices.loc[today]
            window = self._stockprices.loc[:today].iloc[-self._window:]
            sigma_hist = np.log(window).diff().std()*252**.5
            sigma = self._vix.loc[today]/100 if self._volsim.upper()=='VIX' else sigma_hist
            # initialize account
            if i==0:
                for acc in (self._acc, self._acc_bm):
                    acc.deposit(self._inifund)
                    acc.long_stock(last, round(self._inifund/last))
            # buy insurance
            if i%self._reinsur_freq==0:
                old_put = european_vanilla_put(
                            S=last, 
                            K=self._acc._put_strike_price, 
                            T=(self._time_to_maturity-self._reinsur_freq)/252, 
                            r=.025, 
                            sigma=sigma)
                self._acc.close_put_options(old_put)
                new_strike = round(last*(1+self._strike_offset))
                new_put = european_vanilla_put(
                            S=last, 
                            K=new_strike, 
                            T=self._time_to_maturity/252, 
                            r=.025, 
                            sigma=sigma)
                self._acc.long_put_options(new_put, new_strike)
                
            # at every day end
            print(f'{i:5d} | {today.date()} | vix:{sigma:6.2%},histsig:{sigma_hist:6.2%} ', end='')
            for acc in (self._acc, self._acc_bm):
                ttm = self._time_to_maturity-i%self._time_to_maturity
                nav = acc.settlement(today, last, ttm, sigma)[-1]
                print(f' | {acc._name}: {nav:12,.2f}', end='')
            print(end='\t\t\r')
        return self

    def evaluate(self, **kwargs):
        self._set_args(kwargs)
        df = pd.concat((self._acc._dashboard['NAV'], self._acc_bm._dashboard['NAV']), axis=1)
        fig, ax = plt.subplots(2, 1, figsize=(16,8), sharex=True, gridspec_kw={'height_ratios':(3,1,)})
        # performance chart
        title = ', '.join((f'{k}={v}' for k,v in self._args.items()))
        for name,ts in df.iteritems():
            def metrics(name, ts):
                def cal_sharpe(ts, rf=0.025):
                    lndiffs = np.log(ts).diff()
                    mu = lndiffs.mean()*255
                    sigma = lndiffs.std()*252**.5
                    sharpe = (mu-rf)/sigma
                    return mu, sigma, sharpe
                def cal_drawdown(ts):
                    ts = np.log(ts)
                    run_max = np.maximum.accumulate(ts)
                    end = (run_max - ts).idxmax()
                    start = (ts.loc[:end]).idxmax()
                    low = ts.at[end]
                    high = ts.at[start]
                    dd = np.exp(low)/np.exp(high)-1
                    pts = {'high':start, 'low':end}
                    duration = len(ts.loc[start:end])
                    return dd, pts, duration
                mu, sigma, sharpe = cal_sharpe(ts)
                dd, pts, duration = cal_drawdown(ts)
                text = (f'\n{name} |mu:{mu:.2%} | sigma:{sigma:.2%} | sharpe:{sharpe:.2%} | '
                        f'drawdown:{dd:.2%} ({pts["high"].date()}-{pts["low"].date()}, {duration}d)')
                return text
            title += metrics(name, ts)
            ax[0].plot(ts, label=name)
        ax[0].set_title(title)
        # ratio chart
        ratio = self._acc._dashboard['Options']/self._acc._dashboard['NAV']
        ax[1].plot(ratio)
        ax[1].set_title('Options value as percentage of NAV')
        plt.show()        
        return self

if __name__ == "__main__":
    Strategy(
        stock_ticker='SPY',     # underlying stock to long
        inifund=1e6,            # initial fund
        time_to_maturity=21*12, # duration of put option
        reinsur_freq=21,        # how often to roll over to new contract
        strike_offset= -.02,    # -ve is OTM put, +ve is ITM put
    ).run(
        start='2000-01-01',     # backtest start date
        volsim='vix',           # 'vix': use VIX as sigma, or 'hist': use historical volatility as sigma
        window=10,              # window length to calculate historical sigma
    ).evaluate()