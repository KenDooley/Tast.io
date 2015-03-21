"""
This script is used for scraping the reviews data for a single target
business who's most similar 'neighbors' we are searching for.

"""

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import numpy as np
import requests
import logging
import time

MIN_REVIEWS = 20
UNICODE_PUNC = u'!"#%\'()*+,-./:;<=>?@[\]^_`{|}~'
TABLE = dict((ord(char), u' ') for char in UNICODE_PUNC)
STEMMER = PorterStemmer()
STOPWORDS = stopwords.words('english')


# Proxy login information to avoid IP blocks
# AUTH = requests.auth.HTTPProxyAuth('XXXX', 'XXXX')
# PROXIES = {'http': 'http://XXXXXX'}

def tokenize(doc):
        """
        Tokenizes the input sting and returns the tokens in a list.

        INPUT: doc -- a string to be tokenized
        OUTPUT: a list of stemmed words (minus stopwords and punctuation) in
               the doc.
        """
        global STEMMER, STOPWORDS
        no_punc = remov_punc(doc)
        tokens = word_tokenize(no_punc.lower())
        lemmas_no_stop = [STEMMER.stem(word) for word in tokens
                          if word not in STOPWORDS]
        return lemmas_no_stop


def remov_punc(text):
        """
        Removes unicode punctuation from 'text' and replaces with a space.
        """
        global TABLE
        return text.translate(TABLE)


def doc2vec(tfidf_doc, w2v_model, dictionary):
    """
    Function for calculating a document vector for a single search document.
    This is done by calculating the weighted average word vector
    over all words in the document.  The weightings are determined by each
    word's TFIDF weight.  Words which appear in our corpus < 5 times are
    not considered.

    INPUT: tfidf_doc - a single gensim corpus object that has been tfidf
           vectorized
           w2v_model - a trained gensim word2vec model
           dictionary - a gensim dictionary object

    OUTPUT: a numpy array of dimension equal to the dimensionality of our
           trained Word2Vec model (300 in this case).
    """
    if len(tfidf_doc) == 0:
        return []

    tot_wgt = 0.0
    for word in tfidf_doc:
        try:
            a = w2v_model[dictionary[word[0]]]
            tot_wgt += word[1]
        except:
            logging.warning("%s not in Word2Vec" % word[0])

    output = np.zeros(300)
    for word in tfidf_doc:
        try:
            output += w2v_model[dictionary[word[0]]] * (word[1] / tot_wgt)
        except:
            logging.warning("%s not in dictionary" % word[0])
    return output


def get_npages(url):
    """
    This function returns the number of pages of reviews on Yelp for the
    restaurant who's main Yelp page is specified in 'url'.
    """

    text = try_requests(url)
    if len(text) > 0:
        soup = BeautifulSoup(text, "html.parser")
        pages = soup.find('div', class_="page-of-pages")
        if pages:
            pages = pages.text.split()[-1]
        else:
            pages = "1"
        return int(pages)
    return 0


def try_requests(url):
    """
    This function attempts to access the html of the target webpage in 'url'.
    If the attempt is unsuccessful, the function tries again three more times
    before giving up and throwing a warning.

    INPUT: url -- url address of target webpage to be loaded.
    OUTPUT: The html text of the target webpage
    """

    counter = 0
    try:
        r = requests.get(url)
    except:
        logging.warning('Could not access link %s' % url)
        return ""

    if r.status_code == 200:
        return r.text
    else:
        while counter < 3:
            counter += 1
            time.sleep(5)
            r = requests.get(url)
            if r.status_code == 200:
                return r.text
        logging.warning('Could not access link %s' % url)
        logging.warning("%d" % r.status_code)
        return ""


def get_search_vector(busi, phrase_file, dict_file, tfidf_file, word2vec_file):
    """
    This function scrapes the review texts for a given search restaurant.

    INPUT: busi - a Yelp search output JSON for a search target.
    OUTPUT: The doc2vec representation of the restaurant (array)

    """

    global MIN_REVIEWS
    logging.warning('getting reviews for ' + busi['name'])

    rev = []
    output = u""

    if (busi.get('review_count', 0) > MIN_REVIEWS):

        npages = get_npages(busi['url'])

        pagen = min(13, npages)
        for page in xrange(pagen):

            index = page * 40
            if page == 0:
                url = '%s' % busi['url']
            else:
                url = '%s?start=%d' % (busi['url'], index)
            text = try_requests(url)

            if len(text) > 0:
                soup = BeautifulSoup(text, 'html.parser')
                bucket = soup.find_all('p', itemprop='description')
                rev.extend([item.text.replace(u'\xa0', u' ') for item in bucket])

        output = u" ".join(rev)
    output = tfidf_file[dict_file.doc2bow(phrase_file[tokenize(output)])]
    return doc2vec(output, word2vec_file, dict_file)
