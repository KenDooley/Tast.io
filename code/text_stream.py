"""
The Class MyText is used to generate either a stream of tokenized sentences or
tokenized documents from the reviews of all businesses in our MongoDB.  A
'document' in this case is considered to be the list returned after
concatenating all of the sentences found across all reviews for a given
business.  This class is used to detect bi-grams over our entire corpus,
initialize the dictionary to use in our bag-of-words, and to generate the
corpus itself.  A 'phrase_file' can be specified in order to use bi-grams
during the tokenization step.

notes: make sure mongod running. use `sudo mongod` in terminal
"""
from pymongo import MongoClient
from gensim import corpora, models
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import logging
import re

# Parameters to access MongoDB
CLIENT = MongoClient()
DB1 = CLIENT.yelpNY
DB2 = CLIENT.yelpSF
COLL1 = DB1.restaurant
COLL2 = DB2.restaurants
COLLS = (COLL1, COLL2)
REVIEW_CUTOFF = 'this.reviews.length>50'
MAX_REVIEWS = 500

# NLP Tools
STEMMER = PorterStemmer()
STOPWORDS = stopwords.words('english')
UNICODE_PUNC = u'!"#%\'()*+,-./:;<=>?@[\]^_`{|}~'
TABLE = dict((ord(char), u' ') for char in UNICODE_PUNC)
SPLIT_SENTENCES = re.compile(u'[.!?:]\s+')  # split sentences on '.!?:'

# Log handling
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


class MyText(object):

    def __init__(self, phrase_file=None, dict_file=None, sentences=True):
        self.sentences = sentences
        self.name_idx = {}
        self.idx_name = []
        if phrase_file is None:
            self.phrases = None
        else:
            self.phrases = models.Phrases.load(phrase_file)

        if dict_file is None:
            self.dict = None
            self.corpus = False
        else:
            self.dict = corpora.Dictionary.load(dict_file)
            self.corpus = True

    def __tokenize(self, doc):
        """
        Tokenizes the input sting and returns the tokens in a list.

        INPUT: doc -- a string to be tokenized
        OUTPUT: a list of stemmed words (minus stopwords and punctuation) in
               the doc.
        """
        global STEMMER, STOPWORDS
        no_punc = self.__remov_punc(doc)
        tokens = word_tokenize(no_punc.lower())
        lemmas_no_stop = [STEMMER.stem(word) for word in tokens
                          if word not in STOPWORDS]
        return lemmas_no_stop

    def __remov_punc(self, text):
        """
        Removes unicode punctuation from 'text' and replaces with a space.
        """
        global TABLE
        return text.translate(TABLE)

    def __gen_all_sentences(self):
        """
        Yields a stream of tokenized sentences from the entire review corpus
        """
        global COLLS, REVIEW_CUTOFF
        for coll in COLLS:
            busi = coll.find({'reviews': {'$exists': 1},
                              '$where': REVIEW_CUTOFF})
            for idx, business in enumerate(busi):
                for sentence in self.__busi_sentences(business):
                    yield sentence

    def __busi_sentences(self, business):
        """
        Yields a stream of tokenized sentences from the reviews of
        'business'

        INPUT: business -- a dictionary containing information regarding a
        certain business
        """
        global MAX_REVIEWS
        print "Parsing data for %s, review_count = %d" % \
            (business.get('name', ""), business.get('review_count', -1))

        N = len(business.get('reviews', []))
        end = N if N <= MAX_REVIEWS else MAX_REVIEWS
        reviews = business.get('reviews', [])[0:end]

        for num, review in enumerate(reviews):
            for sentence in self.__review_sentence_parse(review['text']):
                yield sentence

    def __review_sentence_parse(self, doc):
        """ Parses a single review into sentences and returns them as
        tokenized lists.  Bigrams are considered if the phrase_file has
        been defined at initialization.
        INPUT: doc -- a text string
        """
        global SPLIT_SENTENCES
        if self.phrases is None:
            for sentence in SPLIT_SENTENCES.split(doc):
                yield self.__tokenize(sentence)
        else:
            for sentence in SPLIT_SENTENCES.split(doc):
                yield self.phrases[self.__tokenize(sentence)]

    def __busi_doc(self, business):
        """
        Returns a concatenated list of all sentences in the reviews
        for 'business'
        INPUT: business -- a dictionary containing information regarding a
        certain business
        """
        output = []
        for sentence in self.__busi_sentences(business):
                output.extend(sentence)
        return output

    def __gen_all_docs(self):
        """
        Yields a stream of tokenized documents from the entire review corpus.
        Also creates lookup dictinaries when defining the actual gensim
        corpus file.
        """
        global COLLS, REVIEW_CUTOFF
        count = 0
        for coll in COLLS:
            busi = coll.find({'reviews': {'$exists': 1},
                              '$where': REVIEW_CUTOFF})
            if self.corpus:
                for idx, business in enumerate(busi):
                    self.name_idx[business['name'].lower()] = count
                    self.idx_name.append({
                        'name': business['name'],
                        'rating': business.get('rating', -1),
                        'snippet': business.get('snippet_text', ""),
                        'location': business['location'].get('state_code', ''),
                        'category': business.get('categories', []),
                        'coordinate': business['location'].get('coordinate', []),
                        'address': business['location'].get('display_address', []),
                        'url': business.get('url', ''),
                        'review_count': business.get('review_count', -1)
                        })
                    count += 1
                    yield self.dict.doc2bow(self.__busi_doc(business))

            else:
                for idx, business in enumerate(busi):
                    yield self.__busi_doc(business)

    def __iter__(self):
        if self.sentences:
            for sentence in self.__gen_all_sentences():
                    yield sentence
        else:
            for doc in self.__gen_all_docs():
                    yield doc
