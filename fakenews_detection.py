#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Dependancies
import pandas as pd
import numpy as np

import re
import nltk

nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


stpwords = stopwords.words("english")

print(stpwords)


# In[13]:


# Loading CSV into pandas
news_data = pd.read_csv("NB-fakenews.csv")

print(news_data.head(10))
print(news_data.shape)  # 20800 rows and 5 columns

print(news_data.isnull().sum())

#  Replacing missing values with isnull

news_data = news_data.fillna("")

# Merging Author and Title column togheter

news_data['content'] = news_data['author']+' '+ news_data['title']
news_data.isnull().sum()


# ### So the author and the title columns are merged with eachother

# In[18]:


news_data['content']


# In[20]:


news_data


# In[21]:


# The target lebels are in the label column 0= reliable 1= unrelaible

x = news_data.drop(columns = 'label', axis = 1)
y = news_data['label']


# In[22]:


x


# In[23]:


y


# The dataframe is set x variable excuding labels
# The label is set to y from dataframe

# ### Stemming: process of reducing a word to its root word
#
# Example: Action, Actor, Acting = Act

# In[26]:


port_stem = PorterStemmer()


# In[58]:


def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content) # Substitude eveyry character excluding a-zA-Z
    stemmed_content = stemmed_content.lower() # make lower case
    stemmed_content = stemmed_content.split() # split and make list
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')] # make a new list of words that are not stopwords (meaningless)
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content



# In[59]:


news_data['content'] = news_data['content'].apply(stemming)


# In[60]:


news_data['content']


# In[61]:


# So above we have removed numbers, symbols, commas etc out from the content columns. ..


# In[62]:


x = news_data['content'].values
y = news_data['label'].values


# In[63]:


x.shape


# In[71]:


y


# In[74]:


y.shape


# In[66]:


# Lets convert the values of content column, x, into vector values


# In[ ]:


vectorizer = TfidfVectorizer()
vectorizer.fit(x)
x = vectorizer.transform(x)


# In[ ]:


x.shape


# ### Now its time to split our data and label, x and y, into training and test datasets.

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, stratify = y, random_state = 2)


# # Creating the model

# In[ ]:


model = LogisticRegression()


# In[79]:


model.fit(x_train, y_train)


# In[80]:


# Accuracy prediction


# In[81]:


test_pred = model.predict(x_train)


# In[84]:


trainingd_accuracy = accuracy_score(test_pred, y_train)


# In[85]:


trainingd_accuracy
print(trainingd_accuracy)

# In[ ]:
