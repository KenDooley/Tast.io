"""
This script parses the html of each review for each restaurant in our database.
The rating, date, and text of each review are then stored in their own fields.
It is meant to be run as a multithreaded script called from multi_clean.py in
order to speed up parsing time.

notes: make sure mongod running. use `sudo mongod` in terminal
"""

from pymongo import MongoClient
from bs4 import BeautifulSoup
import time
import sys
import logging

# Mongo DB info
client = MongoClient()
db = client.yelpSF
coll = db.restaurant


def rating_exists(business):
    """
    Checks to see if the html of the reviews for the 'business' document has
    already been parsed by checking whether or not the last review in the
    document already contains the 'rating' field.
    """

    end = max(0, len(business.get('reviews', [{}])) - 1)
    return 'rating' in business.get('reviews', [{}])[end]


def parse_reviews(busi, collection):
    """
    This function parses the html of each review for a given business document,
    'busi', and saves the rating, date, and text of each review to the same
    document in our 'collection'.  It checks to see is the review html data
    has already been scraped and that it has not already been parsed.
    """

    logging.warning('parsing reviews for ' + busi['name'])

    business = collection.find_one({'_id': busi['_id']})
    if (len(business.get('reviews', [])) != 0) and not
    (rating_exists(business)):
        for review in business['reviews']:
            soup = BeautifulSoup(review['html'], "html.parser")

            if soup.find(itemprop='reviewRating'):
                review['rating'] = soup.find(itemprop='ratingValue')['content']
                review['date'] = soup.find(itemprop='datePublished')['content']
                review['text'] = soup.find(itemprop='description').text
        collection.save(business)

if __name__ == '__main__':

    big_list = list(coll.find({}, {'name': 1, '_id': 1}))
    start = int(sys.argv[1])
    end = min(int(sys.argv[2]), len(big_list))
    chunks = list(big_list[start:end])
    big_list = []

    name = 'log_%s_%s.txt' % (start, end)
    logging.basicConfig(filename=name, level=logging.WARNING)

    for busi in chunks:
        strt = time.time()
        parse_reviews(busi, coll)
        logging.warning('%d seconds' % (time.time() - strt))
    logging.warning('DONE')
