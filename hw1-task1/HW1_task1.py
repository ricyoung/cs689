#!/usr/bin/env python
# coding: utf-8

# In[9]:


get_ipython().system('jupyter nbconvert --to script HW1_task1.ipynb')


# In[ ]:


# Task 1: Visualize the MNIST data using PCA. Reduce the data dimension to two or three and plot the
# data of reduced dimension. Must plot all the data of ten groups (0 to 9). (40 points)


# In[ ]:


# richard young


# In[6]:


# Read MNIST data
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
data = pd.read_csv('MNIST_100.csv')


# In[7]:


# make two variables - X and y
y = data.iloc[:, 0]
X = data.drop('label', axis=1)


# In[8]:


# Visualize data

pca = PCA(n_components=2)
pca.fit(X)
PCAX = pca.transform(X)
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 25
fig_size[1] = 25
plt.rcParams["figure.figsize"] = fig_size

#could use other methods to plot all sets of number - but this is clear to understand

plt.scatter(PCAX[0:100, 0], PCAX[0:100, 1], c='black', label="0", alpha=0.3, edgecolors='none') # Digit 0
plt.scatter(PCAX[100:200, 0], PCAX[100:200, 1], c='blue', label="1",alpha=0.3,  edgecolors='none')# Digit 1
plt.scatter(PCAX[200:300, 0], PCAX[200:300, 1], c='red', label="2",alpha=0.3, edgecolors='none')# Digit 2
plt.scatter(PCAX[300:400, 0], PCAX[300:400, 1], c='green', label="3", alpha=0.3,edgecolors='none')# Digit 3
plt.scatter(PCAX[400:500, 0], PCAX[400:500, 1], c='brown', label="4",alpha=0.3,  edgecolors='none')# Digit 4
plt.scatter(PCAX[500:600, 0], PCAX[500:600, 1], c='cyan', label="5",alpha=0.3, edgecolors='none')# Digit 5
plt.scatter(PCAX[600:700, 0], PCAX[600:700, 1], c='magenta', label="6", alpha=0.3, edgecolors='none')# Digit 6
plt.scatter(PCAX[700:800, 0], PCAX[700:800, 1], c='orange', label="7", alpha=0.3, edgecolors='none')# Digit 7
plt.scatter(PCAX[800:900, 0], PCAX[800:900, 1], c='violet', label="8", alpha=0.3, edgecolors='none')# Digit 8
plt.scatter(PCAX[900:1000, 0], PCAX[900:1000, 1], c='pink', label="9", alpha=0.3, edgecolors='none')# Digit 9

plt.legend()
plt.show()


# In[ ]:




