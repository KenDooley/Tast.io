from pymongo import MongoClient
from pymongo import errors
from bs4 import BeautifulSoup
from gensim import corpora, models, similarities
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import logging
import string
import sys
import time
import pdb


#### Parameters to access MongoDB
CLIENT = MongoClient()
DB = CLIENT.yelp
COLL = DB.restaurant
REVIEW_CUTOFF = 'this.reviews.length>20'

#### NLP Tools
TOKENIZER = RegexpTokenizer(r'\w+')
STEMMER = PorterStemmer()
STOPWORDS = stopwords.words('english')
TABLE = string.maketrans("","")

### Log handling
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', 
                    level=logging.INFO)


class MyCorpus(object):
    def __iter__(self):
        busi = COLL.find({'reviews': {'$exists': 1}, '$where': REVIEW_CUTOFF})
        for business in busi:
            doc = ""
            reviews = business.get('reviews', [])
            for review in reviews:
                doc += review['text']
                doc += ' '


def tokenize(doc):
    no_punc = remov_punc(doc)
    tokens = word_tokenize(no_punc.lower())
    lemmas_no_stop = [STEMMER.stem(word) for word in tokens if word not in STOPWORDS]

def remov_punc(s):
    return s.translate(TABLE, string.punctuation)


def get_npages(url):
    text = try_requests(url)
    if len(text) > 0:
        soup = BeautifulSoup(text, "html.parser")
        pages = soup.find('div', class_="page-of-pages").text.split()[-1]
        return int(pages)
    return 0

def get_curr_rev_cnt(busi):
    return len(busi.get('reviews',[]))

def try_requests(url):
    counter = 0
    try:
        r = requests.get(url, proxies=proxies, auth=auth)
        if r.status_code == 200:
            return r.text
        else:
            while counter < 3:
                counter += 1
                time.sleep(5)
                r = requests.get(url, proxies=proxies, auth=auth)
                if r.status_code == 200:
                    return r.text
            print 'Could not access link %s' % url
            print r.status_code
            return ""
    except:
        print 'Could not access link %s' % url
        return ""


def get_reviews(busi, collection):
    print "getting reviews for " + busi['name']
    
    rev = []    
    #if busi.get('review_count', 0) < 1500:
    if (get_curr_rev_cnt(busi) < busi.get('review_count', 0)):
        collection.update({"id" : busi['id']}, { '$set' : { 'reviews': [] } })
        npages = get_npages(busi['url'])
    
        pagen = max(75,npages)
        for page in xrange(pagen):

            index = page * 40
            if page == 0:
                url = '%s' % busi['url']
            else:
                url = '%s?start=%d' % (busi['url'], index)
            text = try_requests(url)

            if len(text) > 0:
                soup = BeautifulSoup(text, "html.parser")
                rev.extend(soup.find_all('div', class_ = 'review'))
            
                #collection.update({"id" : busi['id']}, { '$push' : { 'reviews': { '$each': [ { 'html' : str(item) } for item in rev ]}}})
            time.sleep(0.5)
        try:
            collection.update({"id" : busi['id']}, { '$set' : { 'reviews': [ { 'html' : str(item) } for item in rev ]} })
        except:
            with open("failed.txt", "w") as f:
                for item in rev:
                  f.write("%s\n" % item)
    # else:
    #     print 'too many reviews ', busi['name']
            

def parse_reviews(busi, collection):
    print "getting stars for " + busi['name']

    business = collection.find_one({"id" : busi['id'] })
    for review in business['reviews']:
        soup = BeautifulSoup(review['html'], "html.parser")
 
        if soup.find(itemprop = "reviewRating"):
            review['rating'] = soup.find(itemprop = "ratingValue")['content']
            #collection.save(business)
            
if __name__ == '__main__':
    #for busi in coll.find({'name': 'The House'}):
    #    get_reviews(busi, coll)
    count = 0
    item_lst = list(coll.find({"name": { "$in": [ '/^i/i' ]}}).sort('_id'))

    start = int(sys.argv[1])
    end = int(sys.argv[2])
    sublst = item_lst[start:end]
    for busi in sublst:
        print 'restaurant: ', count
        get_reviews(busi, coll)
        count += 1

    #     parse_reviews(busi, coll)
    #     # five_stars = filter(lambda x: x['rating'] == '5.0', coll.find_one({"id" : busi['id']})['reviews'])