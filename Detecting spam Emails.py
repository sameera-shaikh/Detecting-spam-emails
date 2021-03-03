#!/usr/bin/env python
# coding: utf-8

# In[19]:


from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
import time
import numpy as np 
import pandas as pd 
from nltk.tokenize import RegexpTokenizer  


# In[2]:


pwd


# In[3]:


import os
os.chdir('C:/Users/Sameera/Desktop/dessertetion')


# In[5]:


df=pd.read_csv("spam_ham.csv")


# In[10]:


df.head(5)


# In[7]:


df.info()


# In[8]:


df.isnull().sum()


# In[12]:


x = df["Body"]
x_clnd_link = [re.sub(r"http\S+", "", text) for text in x]

print(x_clnd_link[0])


# In[15]:


pattern = "[^a-zA-Z0-9]"


# In[16]:


x_cleaned = [re.sub(pattern," ",text) for text in x_clnd_link]


# In[17]:


x_lowered = [text.lower() for text in x_cleaned]
print(x_lowered[0])


# In[21]:


import nltk
nltk.download()


# In[22]:


#tokenising
x_tokenized = [nltk.word_tokenize(text) for text in x_lowered]


# In[29]:


print(x_tokenized[0])


# In[30]:


nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemma = WordNetLemmatizer()


# In[31]:


words = ["bats","removed","cheers","good","stopped","went","fired","cleaner","beers"]
for word in words:
    print(lemma.lemmatize(word),end=" ")


# In[32]:


x_lemmatized = [[lemma.lemmatize(word) for word in text] for text in x_tokenized]


# In[33]:


print(x_lemmatized[0])


# In[34]:


#Removing Stopwords
stopwords = nltk.corpus.stopwords.words("english")
x_prepared = [[word for word in text if word not in stopwords] for text in x_lemmatized]


# In[35]:


print(x_prepared[0])


# In[37]:


vectorizer = CountVectorizer(max_features=20000)
x = vectorizer.fit_transform([" ".join(text) for text in x_prepared]).toarray()


# In[38]:


x.shape


# In[40]:


x_train,x_test,y_train,y_test = train_test_split(x,np.asarray(df["Label"]),random_state=42,test_size=0.2)
x_train.shape


# In[42]:


start_time = time.time()
NB = GaussianNB()
NB.fit(x_train,y_train)
end_time = time.time()

print(round(end_time-start_time,2))


# In[43]:


NB.score(x_test,y_test)


# In[44]:


from sklearn.metrics import confusion_matrix
y_pred = NB.predict(x_test)

conf = confusion_matrix(y_pred=y_pred,y_true=y_test)
import seaborn
seaborn.heatmap(conf,annot=True,fmt=".1f",linewidths=1.5)
import matplotlib.pyplot as plt
plt.show()


# In[ ]:




