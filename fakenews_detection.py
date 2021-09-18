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


# Loading CSV into pandas
news_data = pd.read_csv("NB-fakenews.csv")

print(news_data.head(10))
print(news_data.shape)  # 20800 rows and 5 columns

print(news_data.isnull().sum())

#  Replacing missing values with isnull

news_data = news_data.fillna("")

# Merging Author and Title column togheter

news_data["content"] = news_data["author" + " " + ["title"]]


print(news_data["content"])
