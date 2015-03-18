"""
This script is used for scraping the reviews data for all of the restaurants
for which we have metadata in our MongoDB.  It is meant to be run as a
multithreaded script called from multi_process.py in order to speed up
scraping time.

notes: make sure mongod running. use `sudo mongod` in terminal
"""

from pymongo import MongoClient
from bs4 import BeautifulSoup
import requests
import sys
import logging
import time

# Specify MongoDB details here
client = MongoClient()
db = client.yelpNY
coll = db.restaurant

# Proxy login information to avoid IP blocks
AUTH = requests.auth.HTTPProxyAuth('XXXX', 'XXXX')
PROXIES = {'http': 'http://XXXXXX'}


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


def get_curr_rev_cnt(busi):
    """
    This function counts the number of reviews for a given business, 'busi',
    that are already in our database.

    INPUT: busi - a document in our MongoDB collection containing data for a
                  specific restaurant.
    """

    return len(busi.get('reviews', []))


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
        r = requests.get(url, proxies=PROXIES, auth=AUTH)
    except:
        logging.warning('Could not access link %s' % url)
        return ""

    if r.status_code == 200:
        return r.text
    else:
        while counter < 3:
            counter += 1
            time.sleep(5)
            r = requests.get(url, proxies=PROXIES, auth=AUTH)
            if r.status_code == 200:
                return r.text
        logging.warning('Could not access link %s' % url)
        logging.warning("%d" % r.status_code)
        return ""


def get_reviews(busi, collection):
    """
    This function checks to see if the review data has been already scraped for
    a given restaurant, 'busi', and, if not, scrapes the data from Yelp and
    adds it to our 'collection'.  Only the first 1200 reviews for the business
    are scraped if that many exist.

    INPUT: busi - a document in our MongoDB collection containing data for a
                  specific restaurant.
           collection - the MongoDB collection that we will be using.

    """

    logging.warning('getting reviews for ' + busi['name'])

    rev = []

    if (busi.get('review_count', 0) > 0) and
    (get_curr_rev_cnt(busi) < min(1200, busi.get('review_count', 0))):

        collection.update({'id': busi['id']}, {'$set': {'reviews': []}})
        npages = get_npages(busi['url'])

        pagen = min(30, npages)
        for page in xrange(pagen):

            index = page * 40
            if page == 0:
                url = '%s' % busi['url']
            else:
                url = '%s?start=%d' % (busi['url'], index)
            text = try_requests(url)

            if len(text) > 0:
                soup = BeautifulSoup(text, 'html.parser')
                rev.extend(soup.find_all('div', class_='review'))
        try:
            collection.update({'id': busi['id']}, {'$set': {'reviews':
                              [{'html': str(item)} for item in rev]}})
        except:
            with open('failed.txt', 'w') as failed:
                for item in rev:
                    failed.write('%s\n' % item)


if __name__ == '__main__':

    lst = list(coll.find({"$or": [{"reviews": {"$size": 0}}, {"reviews":
                         {"$exists": 0}}]}).sort('_id'))

    start = int(sys.argv[1])
    end = min(len(lst), int(sys.argv[2]))

    name = "log_%d_%d.txt" % (start, end)
    logging.basicConfig(filename=name, level=logging.WARNING)

    count = 0

    sublst = list(lst[start:end])
    lst = []
    for busi in sublst:
        time.sleep(1)
        go = time.time()
        logging.warning('restaurant: %d ' % count)
        get_reviews(busi, coll)
        count += 1
        stop = time.time()
        logging.warning("took %d seconds" % (stop-go))
    sublst = []
    logging.warning("DONE")
