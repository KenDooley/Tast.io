#!/usr/bin/python
# -*- coding: latin-1 -*-

from pymongo import MongoClient
from pymongo import errors
from gensim import corpora, models, similarities
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from collections import defaultdict
import numpy as np
import re
import logging
import string
import unicodedata
import pickle
import sys

#### Parameters to access MongoDB
CLIENT = MongoClient()
DB1 = CLIENT.yelpNY
DB2 = CLIENT.yelpSF
COLL1 = DB1.restaurant
COLL2 = DB2.restaurant
REVIEW_CUTOFF = 'this.reviews.length>50'
MAX_REVIEWS = 500
NUM_DBS = 2

#### NLP Tools
TOKENIZER = RegexpTokenizer(r'\w+')
STEMMER = PorterStemmer()
STOPWORDS = stopwords.words('english')
UNICODE_PUNC = u'!"#%\'()*+,-./:;<=>?@[\]^_`{|}~'
TABLE = dict((ord(char), u' ') for char in UNICODE_PUNC)
DICTIONARY = corpora.Dictionary.load('../data/restaurants.dict')
CORPUS = corpora.MmCorpus('../data/corpus.mm')
TFIDF_MODEL = models.TfidfModel.load('../data/model.tfidf')
LSI_MODEL = models.LsiModel.load('../data/model.lsi')
#INDEX = similarities.MatrixSimilarity.load('../data/big_matrix_wgt.index')
INDEX = similarities.MatrixSimilarity.load('../data/w2v_matrix_wgt.index')
#INDEX = similarities.MatrixSimilarity.load('../data/w2v_matrix.index')
#INDEX = similarities.MatrixSimilarity.load('../data/big_matrix.index')
#INDEX = similarities.MatrixSimilarity.load('../data/restaurant.index')

#### Pickles
with open('../data/idx_name.obj', 'r') as f:
    IDX_NAME = pickle.load(f)

with open('../data/name_idx.obj', 'r') as g:
    NAME_IDX = pickle.load(g)

#with open('../data/big_matrix_wgt.obj', 'r') as h:
#with open('../data/big_matrix.obj', 'r') as h:
#with open('../data/w2v_matrix.obj', 'r') as h:
with open('../data/w2v_matrix_wgt.obj', 'r') as h:
    BIG_MATRIX = pickle.load(h)

### Log handling
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', 
                    level=logging.INFO)


class MyCorpus(object):
    def __init__(self):
        self.name_idx = {}
        self.idx_name = []

    def __iter__(self):
        count = 0
        for coll in (COLL1, COLL2):
            busi = coll.find({'reviews': {'$exists': 1}, '$where': REVIEW_CUTOFF})
            for idx, business in enumerate(busi):
                doc = ""
                print idx

                self.name_idx[business['name'].lower()] = count
                self.idx_name.append({'name': business['name'], 
                                      'rating': business.get('rating', -1),
                                      'snippet': business.get('snippet_text', ""),
                                      'url': business['url'],
                                      'location': business['location'].get('state_code', ''),
                                      'category': business.get('categories', []),
                                      'coordinate': business['location'].get('coordinate',[]),
                                      'address': business['location'].get('display_address', [])})
                count += 1

                N = len(business.get('reviews', []))
                end = N if N <= MAX_REVIEWS else MAX_REVIEWS
                reviews = business.get('reviews', [])[0:end]

                for num, review in enumerate(reviews):
                    doc += review['text']
                    doc += ' '

                final_doc = tokenize(doc) 
                yield DICTIONARY.doc2bow(final_doc)



def tokenize(doc, bigrams=False): ## add parameters here
    no_punc = remov_punc(doc)
    tokens = word_tokenize(no_punc.lower())
    lemmas_no_stop = [STEMMER.stem(word) for word in tokens if word not in STOPWORDS]
    if bigrams:
        for idx in xrange(len(lemmas_no_stop)-1):
            lemmas_no_stop.append(lemmas_no_stop[idx]+'_'+lemmas_no_stop[idx+1])
    return lemmas_no_stop
      
def remov_punc(text):
    try:
        return text.translate(TABLE)
    except:
        return re.sub(ur"\p{P}+", "", text)

# def bigram_preprocess(doc, deacc=True, lowercase=True, errors='ignore',
#     stemmer=None, stopwords=None):
#     """
#     Convert a document into a list of tokens.
#     Split text into sentences and sentences into bigrams.
#     the bigrams returned are the tokens
#     """
#     bigrams = []
#     #split doc into sentences
#     for sentence in SPLIT_SENTENCES.split(doc):
#         #split sentence into tokens
#         tokens = list(simple_preprocess(sentence, deacc, lowercase, errors=errors,
#             stemmer=stemmer, stopwords=stopwords))
#         #construct bigrams from tokens
#         if len(tokens) >1:
#             for i in range(0,len(tokens)-1):
#                 yield tokens[i] + '_' + tokens[i+1]

# def simple_preprocess(doc, deacc=True, lowercase=True, errors='ignore',
#     stemmer=None, stopwords=None):
#     """
#     Convert a document into a list of tokens.
#     This lowercases, tokenizes, stems, normalizes etc. -- the output are final,
#     utf8 encoded strings that won't be processed any further.
#     """
#     if not stopwords:
#         stopwords = []
#     #tokens = [token.encode('utf8') for token in
#     #            tokenize(doc, lowercase=lowercase, deacc=deacc, errors=errors)
#     #        if 2 <= len(token) <= 25 and
#     #            not token.startswith('_') and
#     #            token not in STOPWORDS]
#     #return tokens
#     for token in stem_tokenize(doc, lowercase=lowercase, deacc=deacc, errors=errors, stemmer=stemmer):
#         if 2 <= len(token) <= 25 and not token.startswith(u'_') and token not in stopwords:
#             yield token.encode('utf8')

# def stem_tokenize(doc, deacc=True, lowercase=True, errors="strict", stemmer=None):
#     """ Split into words and stem that word if a stemmer is given"""
#     if stemmer is None:
#         for token in tokenize(doc, lowercase=lowercase, deacc=deacc, errors=errors):
#             yield token
#     else:
#          for token in tokenize(doc, lowercase=lowercase, deacc=deacc, errors=errors):
#             yield stemmer.stemWord(token)

def get_query_vec(coll, name_list):
    
    lsi_vecs = []

    for name in name_list:
        doc = ''
    
        business = coll.find_one({'name': name})
        #for idx, business in enumerate(busi): #### add error handling when restaurant not found

        N = len(business.get('reviews', []))
        end = N if N <= MAX_REVIEWS else MAX_REVIEWS
        reviews = business.get('reviews', [])[0:end]

        for num, review in enumerate(reviews):
            doc += review['text']
            doc += ' '

        final_doc = tokenize(doc)
        vec_bow = DICTIONARY.doc2bow(final_doc)
        vec_tfidf = TFIDF_MODEL[vec_bow]
        vec_lsi = LSI_MODEL[vec_tfidf]
        lsi_vecs.append(vec_lsi)

    final_lsi_vec = merge_lsi_vecs(lsi_vecs)
    return final_lsi_vec


def merge_lsi_vecs(vectors):
    N = float(len(vectors))

    output = defaultdict(float)

    for idx, vec in enumerate(vectors):
        dictify = dict(vec)
        for key, value in dictify.iteritems():
            output[key] += value / N

    return output.items()

def get_doc_vec(doc):
    pass
# señor sisig
if __name__ == '__main__':

    names = ["rainforest cafe".decode('utf8')]
    state = 'CA'

    target_city = 'NY' if state == 'CA' else 'CA'

    ids = [NAME_IDX[item] for item in names]
    idx = NAME_IDX[names[0]]

    vec_lsi = reduce(lambda a,b: a+b, [BIG_MATRIX[loc] for loc in ids]) / \
                     float(len(ids))
    #vec_lsi = BIG_MATRIX[idx]
    #vec_lsi = LSI_MODEL[TFIDF_MODEL[CORPUS[idx]]]
    #vec_lsi = get_query_vec(COLL2, names)

    sims = INDEX[vec_lsi]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])

    output = []
    for idx, sim in enumerate(sims):
        if IDX_NAME[sim[0]]['location'] == state:
            output.append((sim[1], IDX_NAME[sim[0]]['name'].encode('utf8')))
            if len(output) == 30:
                break
        #sims[i][1], IDX_NAME[sims[i][0]]['name'], IDX_NAME[sims[i][0]]['location']

    for item in output:
        print item[0], item[1].decode('utf8')

    # busi = COLL.find({'name': 'House of Prime Rib'})
    # for idx, business in enumerate(busi):
    #     doc = ''
    #     N = len(business.get('reviews', []))
    #     end = N if N <= MAX_REVIEWS else MAX_REVIEWS
    #     reviews = business.get('reviews', [])[0:end]
    #     for num, review in enumerate(reviews):
    #         doc += review['text']
    #         doc += ' '

    #     final_doc = tokenize(doc)
    #     vec_bow = DICTIONARY.doc2bow(final_doc)

    #     vec_tfidf = TFIDF_MODEL[vec_bow]

    #     print vec_tfidf





    # busi = COLL.find({'reviews': {'$exists': 1}, '$where': REVIEW_CUTOFF})
    # for idx, business in enumerate(busi):
    #     doc = ""

    #     N = len(business.get('reviews', []))
    #     end = N if N <= MAX_REVIEWS else MAX_REVIEWS
    #     reviews = business.get('reviews', [])[0:end]

    #     for num, review in enumerate(reviews):
    #         doc += review['text']
    #         doc += ' '

    #     final_doc = [tokenize(doc)]  
    #     DICTIONARY.add_documents(final_doc)
    #     print "Finished record #%d, %s" % (idx, business['name'])

    # once_ids = [tokenid for tokenid, docfreq in DICTIONARY.dfs.iteritems() if docfreq == 1]
    # DICTIONARY.filter_tokens(once_ids)
    # DICTIONARY.compactify()
    # DICTIONARY.save('/tmp/restaurants.dict')


   