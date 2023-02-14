import numpy as np
from scipy.stats import norm

N = norm.cdf

def bs_call(stock_price, strike_price, time, risk_free, sigma):
    d1 = (np.log(stock_price/strike_price) + (risk_free+ sigma**2/2)*time) / (sigma*np.sqrt(time))
    d2 = d1 - sigma * np.sqrt(time)
    return stock_price * N(d1) - strike_price * np.exp(-risk_free*time)* N(d2)

def bs_puts(stock_price, strike_price, time, risk_free, sigma):
    d1 = (np.log(stock_price/strike_price) + (risk_free + sigma**2/2)*time) / (sigma*np.sqrt(time))
    d2 = d1 - sigma* np.sqrt(time)
    return strike_price*np.exp(-risk_free*time)*N(-d2) - stock_price*N(-d1)