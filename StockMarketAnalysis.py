
# coding: utf-8

# In[147]:

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')
sns.set_style('whitegrid')
from pandas.io.data import DataReader


# In[148]:

from datetime import datetime
from __future__ import division


# In[149]:

tech_list = ['AAPL','GOOG','MSFT','AMZN', 'NXPI', 'CSCO', 'QCOM','INTC', 'HPQ','IBM']


# In[150]:

end = datetime.now()
start = datetime(end.year-1,end.month,end.day)


# In[151]:

for stock in tech_list:
    globals()[stock] = DataReader(stock, 'yahoo', start, end)


# In[152]:

NXPI.head()


# In[153]:

NXPI.describe()


# In[154]:

NXPI.info()


# In[155]:

NXPI['Adj Close'].plot (legend=True, figsize=(10,5))


# In[156]:

NXPI['Volume'].plot (legend = True, figsize =(10,5))


# In[157]:

moving_average = [10,20,50]


# In[158]:

for ma in moving_average :
    column_name = "MA for %s days" %(str(ma))
    NXPI [column_name] = pd.rolling_mean(NXPI['Adj Close'],ma)


# In[159]:

NXPI[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(subplots=False, figsize =(10,5))


# In[160]:

NXPI['Daily Return'] = NXPI['Adj Close'].pct_change()
NXPI['Daily Return'].plot(figsize=(10,4), legend = True, linestyle ='--', marker ='o')


# In[161]:

sns.distplot(NXPI['Daily Return'].dropna(), bins=100,color = 'purple')


# In[162]:

NXPI['Daily Return'].hist(bins=100)


# In[163]:

closing_df = DataReader(tech_list,'yahoo',start, end)['Adj Close']


# In[164]:

closing_df.head()


# In[165]:

tech_rets = closing_df.pct_change()


# In[166]:

tech_rets.head()


# In[167]:

sns.jointplot('NXPI','CSCO', tech_rets, kind='scatter', color='blue')


# In[168]:

sns.pairplot(tech_rets.dropna())


# In[169]:

return_fig = sns.PairGrid(tech_rets.dropna())
return_fig.map_upper(plt.scatter,color='purple')
return_fig.map_lower(sns.kdeplot,cmap='cool_d')
return_fig.map_diag(plt.hist,bins=30)


# In[170]:

rets = tech_rets.dropna()


# In[171]:

area = np.pi*20

plt.scatter(rets.mean(),rets.std(),s=area)

plt.xlabel('Expected Return')
plt.ylabel('Risk')
for label, x, y in zip (rets.columns, rets.mean(), rets.std()):
     plt.annotate(
            label,
            xy= (x,y), xytext = (75,75),
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            arrowprops = dict(arrowstyle = '-', connectionstyle='arc3, rad=-0.1'))


