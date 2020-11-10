#!/usr/bin/env python
# coding: utf-8

# In[154]:


import pandas as pd
import matplotlib.pylab as plt
from sklearn.cluster import KMeans as km
df = pd.read_csv('C:\\Users\\Bhanu Teja\\Documents\\airlines.csv')
df.describe()


# In[155]:


#normalization
def fun(i):
    x = ((i-i.min())/(i.max()-i.min()))
    return(x)
df_norm = fun(df.iloc[:,1:])
df_norm.describe()


# In[157]:


#elbow curve
wss = []
k = list(range(10,100,5))
for i in k:
    kmeans = km(n_clusters = i)
    kmeans.fit(df_norm)
    wss.append(kmeans.inertia_)
wss


# In[158]:


plt.plot(k,wss,'ro-');plt.xlabel('number of clusters');plt.ylabel('total with in ss')


# In[159]:


model = km(n_clusters = 40)
model.fit(df_norm)
model.labels_


# In[160]:


m = pd.Series(model.labels_)
df['clust']=m


# In[161]:


df.head()


# In[162]:


df = df.iloc[:,[0,12,1,2,3,4,5,6,7,8,9,10,11]]
df.head()


# In[163]:


df.iloc[:,2:].groupby(df.clust).mean()


# In[166]:


df.to_csv("Kmeans__airlines.csv", encoding = "utf-8")

import os
os.getcwd()


# ## Observations

# *Here what i have obseved is there are 4000 records in the dataset i plotted elbow curve for that data and analysed graph and decided to take 40 clusters and performed KMeans clustering from sklearn library after clustering i grouped data based on cluster and applied mean funtion to that and gave logical names to that*

# In[ ]:




