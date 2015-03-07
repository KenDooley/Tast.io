from pymongo import MongoClient
from pymongo import errors
from gensim import corpora, models, similarities
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
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
#NUM_DBS = 2

#### NLP Tools
TOKENIZER = RegexpTokenizer(r'\w+')
STEMMER = PorterStemmer()
STOPWORDS = stopwords.words('english')
UNICODE_PUNC = u'!"#%\'()*+,-./:;<=>?@[\]^_`{|}~'
TABLE = dict((ord(char), u' ') for char in UNICODE_PUNC)
#DICTIONARY = corpora.Dictionary.load('/tmp/restaurants.dict')
SPLIT_SENTENCES = re.compile(u'[.!?:]\s+')  # split sentences on '.!?:' characters
#TFIDF_MODEL = models.TfidfModel.load('/tmp/modeltfidfSF.lsi')

### Log handling
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', 
                    level=logging.INFO)


# class MyCorpus(object):
#     def __init__(self):
#         self.name_idx = {}
#         self.idx_name = []

#     def __iter__(self):
#         count = 0
#         for coll in (COLL1, COLL2):
#             busi = coll.find({'reviews': {'$exists': 1}, '$where': REVIEW_CUTOFF})
#             for idx, business in enumerate(busi):
#                 doc = ""
#                 print idx

#                 self.name_idx[business['name'].lower()] = count
#                 self.idx_name.append({'name': business['name'], 
#                                       'rating': business.get('rating', -1),
#                                       'snippet': business.get('snippet_text', ""),
#                                       'url': business['url'],
#                                       'location': business['location'].get('state_code', ''),
#                                       'category': business.get('categories', []),
#                                       'coordinate': business['location'].get('coordinate',[]),
#                                       'address': business['location'].get('display_address', [])})
#                 count += 1

#                 N = len(business.get('reviews', []))
#                 end = N if N <= MAX_REVIEWS else MAX_REVIEWS
#                 reviews = business.get('reviews', [])[0:end]

#                 for num, review in enumerate(reviews):
#                     doc += review['text']
#                     doc += ' '

#                 final_doc = tokenize(doc) 
#                 yield DICTIONARY.doc2bow(final_doc)



def tokenize(doc, bigrams=False): ## add parameters here
    no_punc = remov_punc(doc)
    tokens = word_tokenize(no_punc.lower())
    lemmas_no_stop = [STEMMER.stem(word) for word in tokens if word not in STOPWORDS]
    if bigrams:
        for idx in xrange(len(lemmas_no_stop)-1):
            lemmas_no_stop.append(lemmas_no_stop[idx]+'_'+lemmas_no_stop[idx+1])
    return lemmas_no_stop
      
def remov_punc(text):
    return text.translate(TABLE)

def review_sentence_parse(doc):
    for sentence in SPLIT_SENTENCES.split(doc):
        # no_punc = remov_punc(sentence)
        # tokens = word_tokenize(no_punc.lower())
        # lemmas_no_stop = [STEMMER.stem(word) for word in tokens if word not in STOPWORDS]
        yield ' '.join(tokenize(sentence))


def get_sentences(coll):
    busi = coll.find({'reviews': {'$exists': 1}, '$where': REVIEW_CUTOFF})
    for idx, business in enumerate(busi):
        print "Parsing data for %s, number %d, review_count = %d" % \
              (business.get('name', ""), idx, business.get('review_count', -1))

        N = len(business.get('reviews', []))
        end = N if N <= MAX_REVIEWS else MAX_REVIEWS
        reviews = business.get('reviews', [])[0:end]

        for num, review in enumerate(reviews):
            for sentence in review_sentence_parse(review['text']):
                yield sentence
            
def create_sentence_file(filename='review_sentences.obj'):
    sentences = []
    
    for coll in (COLL1, COLL2):
        for sentence in get_sentences(coll):
            sentences.append(sentence)

    f = open(filename, 'w') 
    pickle.dump(sentences, f)
    f.close()

    

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

            
def sentence_gen(sentence_file):
    with open(sentence_file, 'r') as f:
        sentences = pickle.load(f)

        for sentence in sentences:
            yield sentence.split()

if __name__ == '__main__':

    bigram = models.Phrases(sentence_gen('review_sentences.obj'))

    bigram.save('bigrams')

    # f = open('review_sentences.obj', 'r')
    # sentences = pickle.load(f)

    # #busi = COLL.find({'reviews': {'$exists': 1}, '$where': REVIEW_CUTOFF})
    # for idx in xrange(len(sentences)):
    #     sentences[idx] = sentences[idx].split()

    # model = models.Word2Vec(sentences, min_count=5, size=300, workers=2)

    # model.save('mymodel')

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


   