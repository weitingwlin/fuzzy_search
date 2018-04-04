
import numpy as np
import pandas as pd
import string
import re
import dill
from itertools import product, combinations
from nltk.corpus import stopwords
from scipy.stats import rankdata

c_dict = dill.load( open("app/data/morecommon_dict.pkl","rb"))
vocab = c_dict.keys()
books = pd.read_csv('app/data/fiction.csv')

def search_app(S, n = 5):
    res = fuzzy_find2(S, books, maxshow=n)
    return res


##### functions below
def trim_string(S):
    '''
    trim useless words if string is too long
    '''
    mystr1 = re.split('[\W\s]+', S)
    # split at punctuation or space

    mystr =[s.lower() for s in mystr1]
    # Remove "the", "a", "an"
    nonsense = ["the", "a", "an", "and", "to","on", "from", "in", "by"]
    mystr = [word for word in mystr if word.lower() not in nonsense]

    # remove more
    mystr_less = [word for word in mystr if word.lower() not in stopwords.words('english')]

    if len(mystr_less) > 0 :
        mystr = mystr_less

    # remove placeholder
    mystr_less = [s for s in mystr if s in vocab]
    if len(mystr_less) > 0 :
        mystr = mystr_less

    return mystr


def str2mat(instr, limit = 5, placeholder = None):
    '''
    Convert string to a vector base on average vector of the composing words.
    instr: the inpput string
    placeholder: for the non-vocabularies
    '''
    # make a place-holder: mean of three strange words
    if placeholder is None:
        ph = np.ones(300)* 3

    mystr = trim_string(instr)

    # number of words
    L = min(len(mystr), limit)

    ## padding up
    sheet = np.ones((300, limit))* 2
    for l in range(L):
        if (mystr[l] in vocab):
            sheet[:,l] = c_dict[mystr[l]]
        else:
            sheet[:,l] = ph

    return L, sheet


def compare_mats(M1, M2 , ph = 2, stress = 0.2, penalty = 0.5):
    n, limit = M1.shape
    L1 = sum(M1[0,:] != 2 ) # lenth of
    L2 = sum(M2[0,:] != 2 )
    # trim
    M1_trim = M1[:, 0:L1]
    M2_trim = M2[:, 0:L2]

    if L1 == 1:
        lin_dist = M2_trim - M1_trim
        euc_dist = [np.sqrt(sum(lin_dist[:,i]  ** 2)) for i in range(L2)]
        dist = min(euc_dist)
    elif L2 == 1:
        lin_dist = np.tile(M2_trim, L1) - M1_trim
        euc_dist = [np.sqrt(sum(lin_dist[:,i]  ** 2)) for i in range(L1)]
        # use mean if target is more than 1 words
        dist = np.mean(euc_dist)
    else:
        #
        ind_product = list(product(np.arange(L2), repeat=L1)) # select from M2 to match the size of M1
        ind_combination = list(combinations(np.arange(L2), L1))
        eucs = []
        for p, ind in enumerate(ind_product): # 2, (0,1):
            M2_p = M2_trim[:,list(ind)] # permuted M2'
            lin_dist = M2_p - M1_trim
            euc = [np.sqrt(sum(lin_dist[:,i]  ** 2)) for i in range(L1)]
            mean_euc = np.mean(euc)
            if ind not in ind_combination:
                mean_euc = mean_euc * (stress + 1)
            eucs.append(mean_euc)
        dist = min(eucs)
    # penalty for unequal list
    if L2 > L1:
        dist = dist +  ((L2 - L1)/(L1 + L2) * penalty)
    return dist

def compare_strs(S1, S2, limit = 5, placeholder = None):
    _, M1 = str2mat(S1, limit = limit)
    _, M2 = str2mat(S2, limit = limit)
    return compare_mats(M1, M2 , ph = placeholder)

def fuzzy_find2(mytitle, shelf, maxshow = 10, threshhold = 5):
    '''
    mytitle: the user input keyword for fuzzy search
    shelf: df with column named 'title', find book from
    maxshow: the max. number of result return.
    threshhold: threshhold of similarity for the "match"
    '''
    dist = []
    for s in shelf["title"]:
        dist.append(compare_strs(mytitle, s))
    dist = np.array(dist)

    fuzzy = np.where(dist < threshhold)[0]
    L = len(fuzzy)
    if L > maxshow:
        rankF = rankdata(dist, method='min')
        fuzzy = np.where(rankF <= maxshow)[0]

#     return shelf["title"][fuzzy], dist[fuzzy]
    return list(shelf["title"][fuzzy])
