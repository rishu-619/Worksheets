#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas  as pd
import numpy as np


# In[2]:


import nltk, re
nltk.download('stopwords') # load english stopwords
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import warnings
warnings.simplefilter("ignore")
warnings.warn("deprecated", DeprecationWarning)
warnings.simplefilter("ignore")


# In[3]:


data=pd.read_csv("messages.csv")
data.head(5)


# In[4]:


data=data.replace(np.nan,"",regex=True)
data.head()


# In[5]:


data.columns


# In[6]:


data.shape


# In[7]:


data.dtypes


# In[8]:


data.isnull().sum()


# In[9]:


import seaborn as sns
sns.countplot(x=data['label'])


# In[10]:


#data['information']=data['subject'].str.cat(data['message'],sep=' ')
data['information']= data["subject"].astype(str) +" "+ data["message"]
data.head()


# In[11]:


data1=data.drop(['subject','message'],axis=1)


# In[12]:


data1


# In[13]:


data1['label'].value_counts()


# In[14]:


import seaborn as sns
sns.countplot(x=data1['label'])


# In[15]:


#New column for Length of message
data1['length'] = data1.information.str.len()
data1.head(100)


# In[16]:


data1.shape


# In[17]:


def lower_case(series):
    return series.lower()


# In[18]:


data1['information']=data1['information'].apply(lower_case)


# In[19]:


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;-_*0-9+]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
def text_prepare(series):
    series=re.sub(r'^.+@[^\.].*\.[a-z]{2,}$','email',series)
    series=re.sub(r'^http\://[a-z0-9\-\.]+\.[a-z]{2,3}(/\S*)?$','webaddress',series)
    series=re.sub(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$','number',series)
    series=re.sub(r'\d+(\.\d+)?', 'numbr',series)
    series=re.sub(REPLACE_BY_SPACE_RE," ",series)
    series = re.sub(BAD_SYMBOLS_RE," ",series)
    series = re.sub(r'\s+'," ",series)
    return series


# In[20]:


tests = ["SQL server - any equivalent of Excel's CHOOSE function_?164441545, - https://medium.com/@datamonsters",
        "How to free  c++ memory vector<int> * arr?"]
for test in tests: print(text_prepare(test))


# In[21]:


data1['information']=data1['information'].map(text_prepare)
data1


# In[22]:


# Remove stopwords
import string
import nltk
from nltk.corpus import  stopwords

stop_words = set(stopwords.words('english') + ['u', 'ü', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure'])

data1['information'] = data1['information'].apply(lambda x: ' '.join(
    term for term in x.split() if term not in stop_words))


# In[23]:


# New column (clean_length) after puncuations,stopwords removal
data1['clean_length'] = data1.information.str.len()
data1


# In[24]:


print('original length',data1.length.sum())
print('clean length',data1.clean_length.sum())


# In[25]:


# Message distribution BEFORE cleaning
import matplotlib.pyplot as plt
f,ax = plt.subplots(1,2,figsize = (15,8))

sns.distplot(data1[data1['label']==1]['length'],bins=20,ax=ax[0],label='Spam messages distribution',color='r')

ax[0].set_xlabel('Spam data1 length')
ax[0].legend()

sns.distplot(data1[data1['label']==0]['length'],bins=20,ax=ax[1],label='ham messages distribution')
ax[1].set_xlabel('ham data1 length')
ax[1].legend()

plt.show()


# In[26]:


# Message distribution AFTER cleaning
f,ax = plt.subplots(1,2,figsize = (15,8))

sns.distplot(data1[data1['label']==1]['clean_length'],bins=20,ax=ax[0],label='Spam messages distribution',color='r')
ax[0].set_xlabel('Spam data1 length')
ax[0].legend()

sns.distplot(data1[data1['label']==0]['clean_length'],bins=20,ax=ax[1],label='ham messages distribution')
ax[1].set_xlabel('ham data1 length')
ax[1].legend()

plt.show()


# In[27]:


#Getting sense of loud words in spam 
from wordcloud import WordCloud


spams = data1['information'][data1['label']==1]

spam_cloud = WordCloud(width=700,height=500,background_color='white',max_words=50).generate(' '.join(spams))

plt.figure(figsize=(10,8),facecolor='r')
plt.imshow(spam_cloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# In[28]:


#Getting sense of loud words in ham 

hams = data1['information'][data1['label']==0]
spam_cloud = WordCloud(width=600,height=400,background_color='white',max_words=50).generate(' '.join(hams))
plt.figure(figsize=(10,8),facecolor='k')
plt.imshow(spam_cloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# In[29]:


# 1. Convert text into vectors using TF-IDF
# 2. Instantiate MultinomialNB classifier
# 3. Split feature and label
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,log_loss

tf_vec = TfidfVectorizer()

naive = MultinomialNB()

features = tf_vec.fit_transform(data1['information'])

X = features
y = data1['label']


# In[30]:


from sklearn.model_selection import train_test_split
X_train,x_test,Y_train,y_test=train_test_split(X,y,train_size=0.7)


# In[31]:


get_ipython().system('pip install imblearn ')


# In[32]:


#performing over sampling to reduce the over fitting
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
os=RandomOverSampler(0.70)
X_train_ns,Y_train_ns=os.fit_sample(X_train,Y_train)
print("The number of classes before fit {}".format(Counter(Y_train)))
print("The number of classes after fit {}".format(Counter(Y_train_ns)))


# In[33]:


naive.fit(X_train_ns,Y_train_ns)

y_pred= naive.predict(x_test)

print ('Final score = > ', accuracy_score(y_test,y_pred))


# In[34]:


print(classification_report(y_test,y_pred))


# In[35]:


# plot confusion matrix heatmap
conf_mat = confusion_matrix(y_test,y_pred)

ax=plt.subplot()

sns.heatmap(conf_mat,annot=True,ax=ax,linewidths=5,linecolor='y',center=0)

ax.set_xlabel('Predicted Labels');ax.set_ylabel('True Labels')

ax.set_title('Confusion matrix')
ax.xaxis.set_ticklabels(['ham','spam'])
ax.yaxis.set_ticklabels(['ham','spam'])
plt.show()


# In[36]:


conf_mat


# In[37]:


#Log loss metrics
log=log_loss(y_test,y_pred)


# In[38]:


print(log)


# In[ ]:




