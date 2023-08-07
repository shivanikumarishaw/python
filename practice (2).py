#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


defaulter_df = pd.read_csv("breast-cancer.csv")
defaulter_df.head()


# In[4]:


print("Size of the data : ", defaulter_df.shape)


# In[5]:


print("Target variable frequency distribution : \n", defaulter_df["diagnosis"].value_counts())


# In[6]:


plt.figure(figsize = (10,8))
sns.countplot(defaulter_df)
plt.title("Frequency distribution of the target variable - Default")
plt.show()


# In[7]:


X = defaulter_df[["radius_mean", "texture_mean"]]
y = defaulter_df["diagnosis"]


# #### Train-test Split

# In[8]:


from sklearn.model_selection import train_test_split


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)


# In[10]:


print("Size of training data : ", X_train.shape[0])
print("Size of test data : ", X_test.shape[0])


# #### Normalization

# In[11]:


from sklearn.preprocessing import MinMaxScaler


# In[12]:


min_max = MinMaxScaler()
min_max.fit(X_train)
train_transformed = min_max.transform(X_train)
transformed = min_max.transform(X_test)
transformed


# In[13]:


X_train["radius_mean_normalized"] = train_transformed[:,0]
X_train["texture_mean_normalized"] = train_transformed[:,1]
X_train.head()


# In[14]:


X_test["radius_mean_normalized"] = transformed[:,0]
X_test["texture_mean_normalized"] = transformed[:,1]
X_test.head()


# In[15]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# In[16]:


knn =  KNeighborsClassifier(n_neighbors = 5, metric = "euclidean")
knn.fit(X_train[["radius_mean","texture_mean"]], y_train)
predictions = knn.predict(X_test[["radius_mean","texture_mean"]])
test_accuracy = accuracy_score(y_test, predictions)
test_accuracy


# In[17]:


knn =  KNeighborsClassifier(n_neighbors = 5, metric = "euclidean")
knn.fit(X_train[["radius_mean_normalized","texture_mean_normalized"]], y_train)
predictions = knn.predict(X_test[["radius_mean_normalized","texture_mean_normalized"]])
test_accuracy = accuracy_score(y_test, predictions)
test_accuracy


# In[24]:


from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report


# In[25]:


cm = confusion_matrix(y_test, predictions)
pd.DataFrame(cm, columns = ["No", "Yes"], index = ["No", "Yes"])


# In[26]:


38/(5+38)


# In[27]:


print(classification_report(y_test,predictions))

