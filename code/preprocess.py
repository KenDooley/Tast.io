from pymongo import MongoClient
from pymongo import errors
from gensim import corpora, models, similarities
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import logging
import re
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
SPLIT_SENTENCES = re.compile(u'[.!?:]\s+')  # split sentences on '.!?:' characters
BIGRAMS = models.Phrases.load('bigrams')
#TFIDF_MODEL = models.TfidfModel.load('/tmp/modeltfidfSF.lsi')

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
                                      'location': business['location'].get('state_code', ''),
                                      'category': business.get('categories', []),
                                      'coordinate': business['location'].get('coordinate',[]),
                                      'address': business['location'].get('display_address', []),
                                      'url': business.get('url', ''),
                                      'review_count': business.get('review_count', -1)})
                count += 1

                N = len(business.get('reviews', []))
                end = N if N <= MAX_REVIEWS else MAX_REVIEWS
                reviews = business.get('reviews', [])[0:end]

                for num, review in enumerate(reviews):
                    doc += review['text']
                    doc += ' '

                final_doc = tokenize(doc) 
                yield DICTIONARY.doc2bow(final_doc)

class MySentences(object):
    def __iter__(self):
        for line in (BIGRAMS[sentence] for sentence in gen_all_sentences()):
            yield line

def tokenize(doc, bigrams=True): ## add parameters here
    no_punc = remov_punc(doc)
    tokens = word_tokenize(no_punc.lower())
    lemmas_no_stop = [STEMMER.stem(word) for word in tokens if word not in STOPWORDS]
    if bigrams:
        for idx in xrange(len(lemmas_no_stop)-1):
            lemmas_no_stop.append(lemmas_no_stop[idx]+'_'+lemmas_no_stop[idx+1])
    return lemmas_no_stop
      
def remov_punc(text):
    return text.translate(TABLE)


def serialize_corpus():
    corpus = MyCorpus()

    corpora.MmCorpus.serialize('corpus_bigrams.mm', corpus)

    file1 = open('name_idx_bigrams.obj', 'w') 
    file2 = open('idx_name_bigrams.obj', 'w') 

    pickle.dump(corpus.name_idx, file1)
    pickle.dump(corpus.idx_name, file2)

    file1.close()
    file2.close()


def tokenize(doc, bigrams=False): ## add parameters here
    no_punc = remov_punc(doc)
    tokens = word_tokenize(no_punc.lower())
    lemmas_no_stop = [STEMMER.stem(word) for word in tokens if word not in STOPWORDS]
    # if bigrams:
    #     for idx in xrange(len(lemmas_no_stop)-1):
    #         lemmas_no_stop.append(lemmas_no_stop[idx]+'_'+lemmas_no_stop[idx+1])
    return lemmas_no_stop

def remov_punc(text):
    return text.translate(TABLE)


def review_sentence_parse(doc):
    for sentence in SPLIT_SENTENCES.split(doc):
        # no_punc = remov_punc(sentence)
        # tokens = word_tokenize(no_punc.lower())
        # lemmas_no_stop = [STEMMER.stem(word) for word in tokens if word not in STOPWORDS]
        yield tokenize(sentence)

def busi_sentences(business):
    print "Parsing data for %s, review_count = %d" % \
              (business.get('name', ""), business.get('review_count', -1))

    N = len(business.get('reviews', []))
    end = N if N <= MAX_REVIEWS else MAX_REVIEWS
    reviews = business.get('reviews', [])[0:end]

    for num, review in enumerate(reviews):
        for sentence in review_sentence_parse(review['text']):
            yield sentence

def gen_all_sentences():
    counter = 0  ## remove later
    for coll in (COLL1, COLL2):
        busi = coll.find({'reviews': {'$exists': 1}, '$where': REVIEW_CUTOFF})
        for idx, business in enumerate(busi):
            print counter
            counter += 1
            for sentence in busi_sentences(business):
                yield sentence

def busi_doc(business):
    output = []
    for sentence in busi_sentences(business):
        output.extend(BIGRAMS[sentence])
    return output

def gen_all_docs():  ###### make a class
    counter = 0  ## remove later
    for coll in (COLL1, COLL2):
        busi = coll.find({'reviews': {'$exists': 1}, '$where': REVIEW_CUTOFF})
        for idx, business in enumerate(busi):
            print counter
            counter += 1
            yield busi_doc(business)


def build_phrases():
    bigram = models.Phrases(gen_all_sentences())
    bigram.save('bigrams')

def build_dictionary():
    bg_stream = (BIGRAMS[sentence] for sentence in gen_all_sentences())
    dictionary = corpora.Dictionary(bg_stream, prune_at=None)

    once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq == 1]
    dictionary.filter_tokens(once_ids)
    dictionary.compactify()
    dictionary.save('restaurants.dict')

def build_word2vec():
    bg_stream = MySentences()
    model = models.Word2Vec(bg_stream, size=300, workers=2)
    model.save('mymodel')

            
if __name__ == '__main__':

    build_word2vec()

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


   