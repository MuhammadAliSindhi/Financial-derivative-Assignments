#!/usr/bin/env python
# coding: utf-8

# # Group Members:
#     Muhammad Ali P180089 
#     Section: 8A
#     Assignment=04

# In[1]:


import pandas as pd
import numpy as np
import datetime as dt # so that pandas can recognize dates properly
import matplotlib.pyplot as plt # for visualization
from matplotlib import style
import pandas_datareader.data as web # to collect data


# In[2]:


style.use('ggplot')
tickers = ['JWN', 'UBER','RCL']
start = dt.datetime(2014, 1, 1)
end = dt.datetime(2020, 11, 20)
returns = pd.DataFrame() # create an empty data frame, returns.
returns


# # Portfolio Optimization:

# In[ ]:





# In[3]:


for ticker in tickers:
    data = web.DataReader(ticker, 'yahoo', start, end)
data


# In[4]:


for ticker in tickers:
    data = web.DataReader(ticker, 'yahoo', start, end)
    data[ticker] = data['Adj Close'].pct_change() # add a column to data frame, data, and store returns in it.
data


# In[5]:


13.972697 / 13.512726 - 1


# In[6]:


for ticker in tickers:
    data = web.DataReader(ticker, 'yahoo', start, end) 
    data[ticker] = data['Adj Close'].pct_change()
    
    if returns.empty:
        returns = data[[ticker]]
    else:
        returns = returns.join(data[[ticker]], how = 'outer')
data


# In[7]:


returns


# In[8]:


type(returns)


# ## Determine the portfolio weights:

# In[9]:


number_of_portfolios = 5
for portfolio in range(number_of_portfolios):
    weights = np.random.random_sample(len(tickers))
    print(weights)


# In[10]:


weights


# In[11]:


weights[0] + weights[1] # these will change every time we run the cells above.


# In[12]:


np.sum(weights)


# In[13]:


weights / np.sum(weights) # to impose constraint on weights to be equal to 1.


# In[14]:


weights = weights / np.sum(weights)


# In[15]:


weights[0] + weights[1] # Now the sum of the weights invested in both the assets is precesiely equal to 1.


# In[16]:


weights = 0 # to start fresh for the following loop.
weights


# In[17]:


number_of_portfolios = 5
for portfolio in range(number_of_portfolios):
    weights = np.random.random_sample(len(tickers))
    weights = weights / np.sum(weights)
    print(weights)


# In[18]:


portfolio_return = []
portfolio_risk = []
sharpe_ratio = []
portfolio_weights = []
rf = 0


# In[19]:


weights = 0 # to start fresh for the following loop.
weights


# In[20]:


number_of_portfolios = 5
for portfolio in range(number_of_portfolios):
    # Generate random portfolio weights
    weights = np.random.random_sample(len(tickers))
    weights = np.round((weights / np.sum(weights)), 3) # round-off to 3 decimal points
    portfolio_weights.append(weights)

print(portfolio_weights)


# ## Compute Annualized Portfolio Returns:

# In[21]:


returns # the data frame, returns, contains daily returns of the 2 stocks.


# In[22]:


returns.mean() 


# In[23]:


weights # the weights of both the assets in a portfolio.


# In[24]:


weights[0], weights[1]


# In[25]:


0.002186 * weights[0] + 0.000582 * weights[1] 


# In[26]:


np.sum(returns.mean() * weights) 


# In[27]:


np.sum(returns.mean() * weights) * 252 


# In[28]:


portfolio_return, portfolio_risk, sharpe_ratio, portfolio_weights, rf, weights


# In[29]:


portfolio_return = []
portfolio_risk = []
sharpe_ratio = []
portfolio_weights = []
rf = 0


# In[30]:


weights = 0


# In[31]:


number_of_portfolios = 5
for portfolio in range(number_of_portfolios):
    # Generate random portfolio weights
    weights = np.random.random_sample(len(tickers))
    weights = np.round((weights / np.sum(weights)), 3) # round-off to 3 decimal points
    portfolio_weights.append(weights)
    #Generate annualized portfolio return
    annualized_return = np.sum(returns.mean() * weights) * 252
    annualized_return = np.round((annualized_return), 3)
    portfolio_return.append(annualized_return)

print(portfolio_weights)
print(portfolio_return)


# ## Compute Covariance Matrix and Portfolio's Risk:

# In[32]:


returns


# In[33]:


returns.cov()


# In[34]:


returns.cov() * 252


# In[35]:


covariance_matrix = returns.cov() * 252
covariance_matrix


# In[36]:


weights


# In[37]:


weights.T # step A in excel.


# In[38]:


np.dot(covariance_matrix, weights) # step B in excel.


# In[39]:


np.dot(weights.T, np.dot(covariance_matrix, weights)) # step C in excel.


# In[40]:


portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
portfolio_variance


# In[41]:


standard_deviation = np.sqrt(portfolio_variance)
standard_deviation


# In[42]:


portfolio_return, portfolio_risk, sharpe_ratio, portfolio_weights, weights, annualized_return, rf


# In[43]:


portfolio_return = []
portfolio_risk = []
sharpe_ratio = []
portfolio_weights = []
rf = 0


# In[44]:


weights = 0
annualized_return = 0


# In[45]:


number_of_portfolios = 50000
for portfolio in range(number_of_portfolios):
    # Generate random portfolio weights
    weights = np.random.random_sample(len(tickers))
    weights = np.round((weights / np.sum(weights)), 3) # round-off to 3 decimal points
    portfolio_weights.append(weights)
    #Generate annualized portfolio return
    annualized_return = np.sum(returns.mean() * weights) * 252
    portfolio_return.append(annualized_return)
    # Generate Portfolio risk
    covariance_matrix = returns.cov() * 252
    portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
    portfolio_standard_deviation = np.sqrt(portfolio_variance)
    portfolio_risk.append(portfolio_standard_deviation)

print(portfolio_weights)
print(portfolio_return)
print(portfolio_risk)


# ## Compute Sharpe Ratio

# In[46]:


annualized_return # the portfolio's expected annual return.


# In[47]:


rf


# In[48]:


annualized_return - rf


# In[49]:


portfolio_standard_deviation


# In[50]:


shrp_ratio = (annualized_return - rf) / portfolio_standard_deviation


# In[51]:


shrp_ratio
# risk premium per unit of risk. The higher the shrp ratio is, the better it is.


# In[52]:


portfolio_return, portfolio_risk, sharpe_ratio, portfolio_weights, weights, annualized_return, rf


# In[53]:


portfolio_return = []
portfolio_risk = []
sharpe_ratio = []
portfolio_weights = []
rf = 0


# In[54]:


weights = 0
annualized_return = 0


# In[55]:


number_of_portfolios = 50000
for portfolio in range(number_of_portfolios):
    # Generate random portfolio weights
    weights = np.random.random_sample(len(tickers))
    weights = np.round((weights / np.sum(weights)), 3) # round-off to 3 decimal points
    portfolio_weights.append(weights)
    #Generate annualized portfolio return
    annualized_return = np.sum(returns.mean() * weights) * 252
    portfolio_return.append(annualized_return)
    # Generate Portfolio risk
    covariance_matrix = returns.cov() * 252
    portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
    portfolio_standard_deviation = np.sqrt(portfolio_variance)
    portfolio_risk.append(portfolio_standard_deviation)
    #Generate Sharpe Ratio
    shrp_ratio = (annualized_return - rf) / portfolio_standard_deviation
    sharpe_ratio.append(shrp_ratio)

print(portfolio_weights)
print(portfolio_return)
print(portfolio_risk)
print(sharpe_ratio)


# In[56]:


type(portfolio_weights), type(portfolio_return), type(portfolio_risk), type(sharpe_ratio)


# In[57]:


portfolio_weights = np.array(portfolio_weights)
portfolio_return = np.array(portfolio_return)
portfolio_risk = np.array(portfolio_risk)
sharpe_ratio = np.array(sharpe_ratio)


# In[58]:


print(portfolio_weights)
print(portfolio_return)
print(portfolio_risk)
print(sharpe_ratio)


# In[59]:


type(portfolio_weights), type(portfolio_return), type(portfolio_risk), type(sharpe_ratio)


# In[60]:


portfolio_metrics = [portfolio_return, portfolio_risk, sharpe_ratio, portfolio_weights]
portfolio_metrics


# In[61]:


portfolio_df = pd.DataFrame(portfolio_metrics)
portfolio_df


# In[62]:


portfolio_df = portfolio_df.T
portfolio_df


# In[63]:


portfolio_df.columns = ['Return', 'Risk', 'Sharpe', 'Weights']
portfolio_df


# ## Everything Together:

# In[64]:


returns, portfolio_return, portfolio_risk, sharpe_ratio, portfolio_weights, rf, weights, annualized_return


# In[65]:


returns = pd.DataFrame()
portfolio_return = []
portfolio_risk = []
sharpe_ratio = []
portfolio_weights = []
rf = 0
weights = 0
annualized_return = 0


# In[66]:


tickers = ['JWN', 'UBER','RCL']
returns = pd.DataFrame()


# In[67]:


for ticker in tickers:
    data = web.DataReader(ticker, 'yahoo', start, end) 
    data[ticker] = data['Adj Close'].pct_change()
    
    if returns.empty:
        returns = data[[ticker]]
    else:
        returns = returns.join(data[[ticker]], how = 'outer')
        
returns


# In[68]:


number_of_portfolios = 50000
for portfolio in range(number_of_portfolios):
    # Generate random portfolio weights
    weights = np.random.random_sample(len(tickers))
    weights = np.round((weights / np.sum(weights)), 3) # round-off to 3 decimal points
    portfolio_weights.append(weights)
    #Generate annualized portfolio return
    annualized_return = np.sum(returns.mean() * weights) * 252
    portfolio_return.append(annualized_return)
    # Generate Portfolio risk
    covariance_matrix = returns.cov() * 252
    portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
    portfolio_standard_deviation = np.sqrt(portfolio_variance)
    portfolio_risk.append(portfolio_standard_deviation)
    #Generate Sharpe Ratio
    shrp_ratio = (annualized_return - rf) / portfolio_standard_deviation
    sharpe_ratio.append(shrp_ratio)


# In[69]:


portfolio_weights = np.array(portfolio_weights)
portfolio_return = np.array(portfolio_return)
portfolio_risk = np.array(portfolio_risk)
sharpe_ratio = np.array(sharpe_ratio)


# In[70]:


portfolio_metrics = [portfolio_return, portfolio_risk, sharpe_ratio, portfolio_weights]


# In[71]:


portfolio_df = pd.DataFrame(portfolio_metrics)


# In[72]:


portfolio_df = portfolio_df.T


# In[73]:


portfolio_df.columns = ['Return', 'Risk', 'Sharpe', 'Weights']
portfolio_df


# ### Identify a minimum risk portfolio:

# In[74]:


portfolio_df


# In[75]:


portfolio_df['Risk'].astype(float).idxmin() # returns the row index of minimum risk, which in this case is 1.


# In[76]:


portfolio_df.iloc[portfolio_df['Risk'].astype(float).idxmin()] 


# In[77]:


min_risk_portfolio = portfolio_df.iloc[portfolio_df['Risk'].astype(float).idxmin()]
min_risk_portfolio


# In[78]:


max_return_portfolio = portfolio_df.iloc[portfolio_df['Return'].astype(float).idxmax()]
max_return_portfolio


# In[79]:


max_sharpe_portfolio = portfolio_df.iloc[portfolio_df['Sharpe'].astype(float).idxmax()]
max_sharpe_portfolio


# In[80]:


print('Minimum Risk Portfolio')
print(min_risk_portfolio)
print(tickers)
print('')

print('Maximum Return Portfolio')
print(max_return_portfolio)
print(tickers)
print('')

print('Maximum Sharpe Ratio Portfolio')
print(max_sharpe_portfolio)
print(tickers)
print('')


# ### Visualization:

# In[81]:


plt.figure(figsize = (10, 5))
plt.scatter(portfolio_risk, portfolio_return, c = portfolio_return / portfolio_risk) # c for colorbar based on sharpe.

plt.title('Portfolio Optimization', fontsize = 26)

plt.xlabel('Volatility', fontsize = 20)
plt.ylabel('Return', fontsize = 20)

plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)

plt.colorbar(label = 'Sharpe Ratio')

plt.show()


# ### Comparison with individual assets:

# In[82]:


returns


# In[83]:


returns.mean() # daily returns of the two stocks.


# In[84]:


returns.std() # dailty standard deviation of the two stocks.


# In[85]:


returns.mean() * 252 # annual returns of the two stocks.


# In[86]:


returns.std() * 252 # annual standard deviations of the two stocks.


# In[87]:


returns.std() * np.sqrt(252) 


# In[ ]:




