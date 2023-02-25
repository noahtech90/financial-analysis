import pandas as pd
import numpy as np
from scipy.stats import norm
import statsmodels.api as sm


### Black Scholes Valuation ###
N = norm.cdf

def bs_call(stock_price, strike_price, time, risk_free, sigma):
    d1 = (np.log(stock_price/strike_price) + (risk_free+ sigma**2/2)*time) / (sigma*np.sqrt(time))
    d2 = d1 - sigma * np.sqrt(time)
    return stock_price * N(d1) - strike_price * np.exp(-risk_free*time)* N(d2)

def bs_puts(stock_price, strike_price, time, risk_free, sigma):
    d1 = (np.log(stock_price/strike_price) + (risk_free + sigma**2/2)*time) / (sigma*np.sqrt(time))
    d2 = d1 - sigma* np.sqrt(time)
    return strike_price*np.exp(-risk_free*time)*N(-d2) - stock_price*N(-d1)

### Black Scholes Valuation End ###

### 3 Factor Model ###
def fama_french_3_factor_weighting(ff3_monthly, stock_data):
    """Analyze Stock Return Series to determine relationship with fama 3 factor model    
    """

    ff3_monthly.rename(columns={"date_ff_factors": 'Date'}, inplace=True)
    ff3_monthly.set_index('Date', inplace=True)

    stock_returns = stock_data['Adj Close'].resample('M').last().pct_change().dropna()
    stock_returns.name = "Month_Rtn"
    ff_data = ff3_monthly.merge(stock_returns,on='Date')

    X = ff_data[['Mkt-RF', 'SMB', 'HML']]
    y = ff_data['Month_Rtn'] - ff_data['RF']
    X = sm.add_constant(X)
    ff_model = sm.OLS(y, X).fit()
    
    parameter_data = []
    row = 1
    for param in ff_model.params:
        parameter = {}
        parameter["parameter"] = ff_model.summary().tables[1][row][0].data.strip()
        parameter["coef"] = float(ff_model.summary().tables[1][row][1].data.strip())
        parameter["P>|t|"] = float(ff_model.summary().tables[1][row][4].data.strip())
        
        parameter_data.append(parameter)
        row += 1
    
    return pd.DataFrame(parameter_data)

def three_factor_expected_return(ff3_monthly, b1, b2, b3):
    rf = ff3_monthly['RF'].mean()
    market_premium = ff3_monthly['Mkt-RF'].mean()
    size_premium = ff3_monthly['SMB'].mean()
    value_premium = ff3_monthly['HML'].mean()

    expected_monthly_return = rf + b1 * market_premium + b2 * size_premium + b3 * value_premium 
    expected_yearly_return = ((expected_monthly_return + 1) ** 12) - 1
    print("Expected yearly return: " + str(expected_yearly_return))
    return expected_yearly_return

