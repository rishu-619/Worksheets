#!/usr/bin/env python
# coding: utf-8

# In[1]:


from urllib.request import urlopen
from urllib.error import HTTPError
from bs4 import BeautifulSoup
def getTitle(url):
    try:
        html = urlopen(url)
    except HTTPError as e:
        return None
    try:
        bsObj = BeautifulSoup(html.read(), "lxml")
        title = bsObj.body.h1
    except AttributeError as e:
        return None
    return title
print(getTitle("http://www.example.com/"))


# In[4]:


get_ipython().system('pip install icrawler')

from icrawler.builtin import GoogleImageCrawler

google_Crawler = GoogleImageCrawler(storage = {'root_dir': r'path'})

google_Crawler.crawl(keyword = 'query', max_num = 10)


# In[ ]:




