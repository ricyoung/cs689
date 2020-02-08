#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !jupyter nbconvert --to script HW1-Task3.ipynb


# In[2]:


# Task 3: Visualize the housing data using violin plot. (30 points)
# Richard young
# https://medium.com/@haydar_ai/learning-data-science-day-9-linear-regression-on-boston-housing-dataset-cd62a80775efab


# In[3]:


#!pip install statsmodels 


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
import statsmodels.api as sm
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")

plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = [20, 9]
plt.rcParams['figure.dpi'] = 200

# special matplotlib argument for improved plots
from matplotlib import rcParams


# In[5]:


from sklearn.datasets import load_boston
boston = load_boston()


# In[6]:


print(boston.data.shape)


# In[7]:


print(boston.feature_names)


# In[8]:


# print(boston.DESCR)


# In[9]:


# boston_pd = pd.DataFrame(boston.data)
bos = pd.DataFrame(boston.data)
bos.columns = boston.feature_names
print(bos.head())
print(bos.shape)


# In[10]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(bos)


# In[11]:


# X
print(rescaledX[0:3,:])


# In[12]:


# https://seaborn.pydata.org/generated/seaborn.violinplot.html
ax = sns.violinplot(data=rescaledX,inner='quartile')
rcParams['figure.figsize'] = 11.7,8.27
ax.set_xticklabels(bos.columns)
sns.despine()
ax.set_title('Distribution of Boston Housing Data Set', fontsize=16);

