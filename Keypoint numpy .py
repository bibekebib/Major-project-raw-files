#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np


# In[17]:


data = np.load('final_test_256.npy', allow_pickle=True)


# In[18]:





# In[19]:


len(globals)


# In[20]:


# globals[0][0][0]
total = []
for i in range(len(globals)):
  total.append((globals[i][0][0],globals[i][1]))


# In[21]:


total


# In[22]:


df = pd.DataFrame(total)


# In[23]:


df[0][1]


# In[24]:


df


# In[25]:


df_test = pd.concat([pd.DataFrame(df[0].values.tolist()), df[1]], axis=1)


# In[26]:


df_test


# In[27]:


df_test.to_csv('keypoints_final_test_256.csv')


# In[ ]:




