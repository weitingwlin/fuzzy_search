# "Or something like that": a fuzzy search application for short texts

More often when what we search for something by keywords, what we actually mean is **"show me the butterfly book, or something like that"**. Most current search application base on word matching cannot handle the case well if we type in the wrong keywords. In order to solve this "or something like that" problem. I plan to build a "fuzzy search" app that can return results base on similarity in the meaning of keywords. This is possible using the  pre-trained word to vector model and some other natural language processing (NLP) skills.

The current prototype can return **"Dragonfly in amber"** when we search for **"Butterfly in resin"**, and return **"A Knight of the Seven Kingdoms"** when we search for **"Six empire"**. 

I also show here clustering of book titles based on meaning and a potential case of "book suggestion".

## Motivation

When we "search" for something from a database, we type a few words that we think are the "keywords". 

Say we are looking for a book called **"Dragonfly in amber"**, usually it is as simple as that:

<a href='https://www.audible.com/search/ref=a_hp_tseft?advsearchKeywords=dragonfly%20in%20amber&filterby=field-keywords'><img src='images/search_dragonfly.png' /></a>

### "Find the butterfly book, or something like that!"

However, more often when what we type in keywords, what we actually mean is **"show me this butterfly book, or something like that"**. In this case, most of current web application may not give you the book that you actually want.

Say, if somehow in your mind, you thought the book is called **"Butterfly in resin"**:

<a href='https://www.audible.com/search/ref=a_hp_tseft?advsearchKeywords=butterfly%20in%20resin&filterby=field-keywords'><img src='images/search_butterfly.png' /></a>

In this case you will not get the dragonfly book you want.

## Proposal

In order to solve this "or something like that" problem. I plan to build a "fuzzy search" app that can return results base on the meaning of keywords, in addition to exact match of words.This is possible using the [**word to vector** model pre-trained with google news](https://code.google.com/archive/p/word2vec/), and some other natural language processing (NLP) skills (e.g. dealing with stopwords).

I plan to used web scraping to get the database (e.g. book listing on audible.com), and create user interface using Django framework. The goal is to optimize the application, so the search result will be as similar to what the user might have in mind. 

## Preliminary results 

#### [Python notebook](https://github.com/weitingwlin/fuzzy_search/blob/master/Fuzzy_search.ipynb)

#### [Markdown file (if there is problem previewing notebook)](https://github.com/weitingwlin/fuzzy_search/blob/master/Fuzzy_search.md)

Base a preliminary database of ~300 book listing (bestsellers in three category) on audible.com, I created python function that can return titles similar to a search string (what user think is the title). 

I presented two cases below where there is no match between the words in the search string and the book title. The current prototype can return **"Dragonfly in amber"** when we search for **"Butterfly in resin"**, and return **"A Knight of the Seven Kingdoms"** when we search for **"Six empire"**.

#### Example: search for **"Butterfly in resin"**
```python
mytitle = "Butterfly in resin"
```

```python
(result, dist) = fuzzy_find(mytitle, maxshow = 3)
result
```


    162    Brie Masters Love in Submission: Submissive in...
    174     Cincuenta Sombras de Grey [Fifty Shades of Grey]
    267                                   Dragonfly in Amber
    Name: title, dtype: object


Here we are: 'Dragonfly in Amber' ...

I also show here clustering of book titles based on meaning and a potential case of "book suggestion". I visualized the similarity of some books based on the meaning of titles (and an anecdotal result that there is no simple way to distinguish books of "Erotica", "Military", or "Sci-Fi" base only on the titles).

#### How it works? Word clustering base on meaning

Titles with similar meaning are plotted closer together. We can make some sense out of the clustering:

1. there is a "War" zone on the left
2. a "Dirty" zone on top
3. and a "Capture" zone middle-bottom

Note that many titles clustered together do not have matching words (for example, "Captive in the Dark" and "Forever with you")

![png](images/output_37_1.png)

## Perspective
1. Commercial search (e.g. audible.com)
2. Personal contacts, messages, emails ("divorce lawyer", "family law attorney")
3. Suggestion for the next best things (e.g. butterfly necklace vs. dragonfly); especially useful when the catalog is small.
4. Other "small data" problems, e.g.
	* reviews on one item: (group similar comments together for more quantitative analysis)
	* FAQ 

## Future works and Challenges
1. Deal with string vectorization (better than taking average of words); incorporate other NLP tools
2. Web scrapping to get more data (for the demo project)
3. Build user interface (work with demo data; with option of user uploaded data base ([similar to this](https://weitingwlin.shinyapps.io/shinydemo/)))
4. Improve **word2vec** model
5. 

## code
* Fuzzy_search.ipynb: demo of the fuzzy search
* Books: exploratory data analysis
