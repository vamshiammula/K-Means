#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pylab as plt
from sklearn.cluster import KMeans as km
f = pd.read_csv('C:\\Users\\Bhanu Teja\\Downloads\\Insurance Dataset.csv\\Insurance Dataset.csv')
f


# In[4]:


f.describe()


# In[341]:


def fun(i):
    x = (i-i.min())/(i.max()-i.min())
    return x
df_norm = fun(f.iloc[:,:])
df_norm.describe()


# In[342]:


#elbow curve
wss = []
k = list(range(2,10))
for i in k:
    kmeans = km(n_clusters = i)
    kmeans.fit(df_norm)
    wss.append(kmeans.inertia_)
wss


# In[343]:


plt.plot(k,wss,'ro-');plt.xlabel('number of clusters');plt.ylabel('total with in ss')


# In[344]:


model = km(n_clusters = 5)
model.fit(df_norm)
model.labels_


# In[345]:


m = pd.Series(model.labels_)
f['clust']=m


# In[346]:


f.head()


# In[347]:


f = f.iloc[:,[5,0,1,2,3,4]]
f.head()


# In[348]:


f.iloc[:,:].groupby(f.clust).mean()


# In[339]:


f.to_csv('kmeans__insurance_dataset.csv',encoding = 'utf-8')
import os
os.getcwd()


# ## Observations

# *Here what i have obseved is there are 100 records in the dataset i plotted elbow curve for that data and analysed graph and decided to take 5 clusters and performed KMeans clustering from sklearn library after clustering i grouped data based on cluster and applied mean funtion to that and gave logical names to that*

# In[ ]:




