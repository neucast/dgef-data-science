from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

titles = []


def PageRetrival():
    local_website = 'https://www.elfinanciero.com.mx/'

    req = Request(local_website, headers={'User-Agent': 'Chrome/35.0.1916.47'})
    page = urlopen(req).read()

    soup = BeautifulSoup(page, 'html.parser')
    print('Page successfully downloaded!')
    return soup


def DataRetrival(soup):
    articles = soup.findAll('article')

    for art in range(len(articles)):
        try:
            titles.append(articles[art].h2.text)
        except:
            titles.append(np.nan)


soup = PageRetrival()
DataRetrival(soup)

titleDataFrame = pd.DataFrame(titles)
#print(titles)
#print(titleDataFrame)

print(titleDataFrame.iloc[0])


# page/robots.txt

# https://www.google.com/robots.txt

