#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pylab as plt
from sklearn.cluster import KMeans as km
c = pd.read_csv('C:\\Users\\Bhanu Teja\\Downloads\\crime_data.csv (1)\\crime_data.csv')
c


# In[ ]:


c.describe()


# In[17]:


def fun(i):
    x = (i-i.min())/(i.max()-i.min())
    return x
df_norm = fun(c.iloc[:,1:])
df_norm.describe()


# In[18]:


#elbow curve
wss = []
k = list(range(2,7))
for i in k:
    kmeans = km(n_clusters = i)
    kmeans.fit(df_norm)
    wss.append(kmeans.inertia_)
wss


# In[19]:


plt.plot(k,wss,'ro-');plt.xlabel("number of clusters");plt.ylabel('total with in ss')


# In[20]:


model = km(n_clusters=4)
model.fit(df_norm)
model.labels_


# In[21]:


m = pd.Series(model.labels_)
c['clust'] = m


# In[22]:


c.head()


# In[23]:


c = c.iloc[:,[5,0,1,2,3,4]]
c.head()


# In[33]:


c.iloc[:,1:].groupby(c.clust).mean()


# In[34]:


c.to_csv("kmeans_crime.csv",encoding = 'utf-8')
import os
os.getcwd()


# ## Observations

# *Here what i have obseved is there are 50 records in the dataset i plotted elbow curve for that data and analysed graph and decided to take 4 clusters and performed KMeans clustering from sklearn library after clustering i grouped data based on cluster and applied mean funtion to that and gave logical names to that*

# In[ ]:




