import numpy as np
import pandas as pd
import string
import re
#import matplotlib.pyplot as plt
#%matplotlib inline
#plt.rcParams["figure.figsize"] = [16,12]

# from sklearn import manifold
from itertools import product, combinations
from nltk.corpus import stopwords
import random
from sklearn.model_selection import train_test_split
import dill
import requests
from bs4 import BeautifulSoup
from datetime import datetime



data = pd.read_csv('data/reviews_balance_label.csv')
data = data[data['emoji']!=7] 

c_dict = dill.load( open("data/morecommon_dict.pkl","rb"))
word2ind = dill.load( open("data/word2ind.pkl","rb"))
vocab = list(c_dict.keys())
emoji_dict = {0:'',1:'😀',2: '🤔',3: '😥',4:'😱',5:'😒',6:'👍'}

Len = 20

def one_hot(X):
    a = np.array(X)
    b = np.zeros((len(a), max(a)+1))
    b[np.arange(len(a)), a] = 1
    return b

def review_to_indices(X, word_to_index=word2ind, max_len = 15):
    m = len(X)
    X_indices = np.zeros((m, max_len))
    vocab = word_to_index.keys()
    for i in range(m): 
        review_words =X[i].lower().split()
        # Initialize j to 0
        j = 0
        # Loop over the words of sentence_words
        for w in review_words[:max_len]:
            if w in vocab:
                X_indices[i, j] = word_to_index[w]
            # Increment j to j + 1
                j = j + 1

    return X_indices

def get_reviews(url, max_reviewers=1, max_sentences=1):
    '''
    url: url of main page for a book
    max_reviewers: randomly select n reviewers
    max_sentences: randomly select n sentences from each reviewer
    return sentences in review
    '''
    short_review = []
    page = requests.get(url)
    soup = BeautifulSoup(page.text, "lxml")
    # get the review block
    parent = soup.find('div',attrs = {'class': 'bc-section listReviewsTabUS'})
    if parent is None:
        return short_review
    reviews = parent.select('p.bc-text.bc-spacing-small.bc-color-secondary')
    # randomly pick reviews
    selected_reviews = random.sample(reviews, min(max_reviewers, len(reviews)))
    
    for review in selected_reviews:
        sentences = [s.strip() for s in re.split('[\.\?\!]\s', review.text) if s not in [''] ]
        sentences = [s for s in sentences if len(s.lower().split()) <= 25]
        selected_sentences = random.sample(sentences, min(max_sentences, len(sentences)))
        short_review =  short_review + selected_sentences
    return short_review


def get_emoji(url, model, max_reviewers = 10, max_sentences=10):
    temp = get_reviews(url, max_reviewers, max_sentences)
    temp = [s for s in temp if s != '']
    t_ind = review_to_indices(temp, word_to_index=word2ind, max_len = Len)
    res = model.predict(t_ind)
    #
    # thresh_ind = np.max(res, axis=1) > thresh
    predicted = np.argmax(res, axis=1)
    emos = []
    for i, t in enumerate(temp):
        if predicted[i] != 0 :
            emos.append([t, emoji_dict[predicted[i]] , predicted[i] ])
    return emos