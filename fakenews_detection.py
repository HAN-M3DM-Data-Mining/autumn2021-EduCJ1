#!/usr/bin/env python
# coding: utf-8

# # Fake news detection

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


# ## Data understanding
# 
# The dataset contains 20800 rows x 5 columns:
# - id: each entry contains and ID
# - title: news title
# - author: author's name/last name
# - text: news content
# - label: this is our target column, 1 = unreliable news & 0 = reliable news

# In[5]:


# Loading CSV into pandas
df = pd.read_csv("NB-fakenews.csv")


# In[6]:


df


# In[7]:


# Checking missing values
df.isnull().sum()


# # Data preparation 

# title & author are missing some values, however it is a small amount in comparison to 20800 rows. 
# 
# The missing entries will be replaced with NULL values. To be able to predict the label column the title & author columns will be used. However both columns are currently containing text values with many characters and symbols, which is making not usable for our model.
# 
# Steps:
# 
# - replace missing values with NULL
# - merge title and author into a new column
# 
# 

# In[9]:


#  Replacing missing values with isnull

df = df.fillna("")
df.isnull().sum()


# In[10]:


# Merging author and title column

df['merged'] = df['author']+' '+ df['title']
df


# In[18]:


# Checking the value caount of label
df['label'].value_counts()


# In[23]:


# newly merged column: title and author
df['merged']


# In[24]:


df.isnull().sum()


# In[26]:


# The target lebels are in the label column 0= reliable 1= unrelaible

x = df.drop(columns = 'label', axis = 1)
y = df['label']


# In[27]:


x.head(2)


# In[28]:


y.head(2)


# In[ ]:





# In[ ]:





# The dataframe is set x variable excuding labels
# The label is set to y from dataframe

# ## Stemming: process of reducing a word to its root word
# 
# Example: Action, Actor, Acting = Act
# 
# From the merged column we will extract/remove all characters/symbols that are not in a to z or A to Z. This is needed because we will use the the words in merged columns to vectorize our data in merged, so that we can apply our prediction model. 
# 
# Steps:
# 
# - Define a function that 1) subtracts symbols and unneeded characters, 2) makes words in lower case, splits the words and sets into a list. 3) Loops through the list to remove meaningless words (that are in the list of stopwords module)
# - Create the model

# In[31]:


stem = PorterStemmer()


# In[36]:


def stemming(merged):
    stemm_merged = re.sub('[^a-zA-Z]', ' ', merged) # Substitude eveyry character excluding a-zA-Z
    stemm_merged = stemm_merged.lower() # make lower case 
    stemm_merged = stemm_merged.split() # split and make list
    stemm_merged = [stem.stem(word) for word in stemm_merged if not word in stopwords.words('english')] # make a new list of words that are not stopwords (meaningless)
    stemm_merged = ' '.join(stemm_merged)
    return stemm_merged
    


# In[48]:


df['merged']


# In[46]:


df['merged'] = df['merged'].apply(stemming)


# In[47]:


df['merged']


# The merged column is ready to be transformed into vectors. Since the model will make the predictions based on only author & title columns which is the merged column now, we will create a new dataframe for merged and another one for the labels. 

# In[49]:


x = df['merged'].values
y = df['label'].values


# In[56]:


print(x, len(x))


# In[58]:


print(y, len(y))


# In[59]:


vectorizer = TfidfVectorizer()
vectorizer.fit(x)
x = vectorizer.transform(x)


# In[63]:


x.shape


# # Creating the model
# 
# Let start with splitting the data into train and test:

# In[66]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, stratify = y, random_state = 2)


# Now we have 4 datasets: 
# - x_train (80% of df['merged']), 
# - x_test(20% of df['merged']),
# - y_train (80% of df['lebel']),
# - y_test (20% of df['lebel'])

# The model is using Logistic Regression Classifier to classify merged column into 0 or 1, reliable or unreliable. 

# In[74]:


model = LogisticRegression()


# In[75]:


model.fit(x_train, y_train) 


# # Deployment and Evaluation

# In[83]:


# Accuracy on training data
x_train_pred = model.predict(x_train)
training_accuracy = accuracy_score(x_train_pred, y_train)


# In[84]:


print('The accuracy of the model on the training data is: ', training_accuracy)


# Very high rate of accuracy on the training data, which makes sense because the model is trained on this dataset. Accuracy on the test data will tell how good the model is predicting. Lets measure the accuracy of the model on the test data:

# In[85]:


# Accuracy on test data
x_test_pred = model.predict(x_test)
test_accuracy = accuracy_score(x_test_pred, y_test)


# In[86]:


print('The accuracy of the model on the test data is: ', test_accuracy)


# The models performs well with 98% of accuracy, it less then 1% inaccurate compare to the the score on the training data.

# In[ ]: