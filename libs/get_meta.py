"""
Script for generating restaurant metadata and inserting it into a MongoDB

notes: make sure mongod running. use `sudo mongod` in terminal
"""

import yelp.search as yelp
from pymongo import MongoClient
from pymongo import errors
import time
import argparse

"""
Insert user's Yelp API key information here
"""

KEY = "XXXX"
SECRET_KEY = "XXXX"
TOKEN = "XXXX"
SECRET_TOKEN = "XXXX"


class GetMeta(object):
    """ This class uses the Yelp search API to scrape business metadata and
    insert it into a MongoDB """

    def __init__(self, database, table_name,
                 key, secret_key, token, secret_token, params):
        client = MongoClient()
        self.database = client[database]
        self.table = self.database[table_name]
        self.key = key
        self.secret_key = secret_key
        self.token = token
        self.secret_token = secret_token
        self.params = params

    def make_request(self):
        """Using the search terms and API keys,
           connect and get from Yelp API"""

        return yelp.request(self.params, self.key, self.secret_key,
                            self.token, self.secret_token)

    def insert_business(self, rest):
        """
        INPUT:
        param rest -- Dictionary object containing meta-data to be
                      inserted in Mongo

        OUTPUT: None

        Inserts dictionary into MongoDB
        """

        if not self.table.find_one({"id": rest['id']}):
            # Make sure all the values are properly encoded
            for field, val in rest.iteritems():
                if type(val) == str:
                    rest[field] = val.encode('utf-8')

            try:
                print "Inserting restaurant " + rest['name']
                self.table.insert(rest)
            except errors.DuplicateKeyError:
                print "Duplicates"
        else:
            print "In collection already"

    def run(self):

        try:
            response = self.make_request()
            total_num = response['total']
            print 'Total number of entries for the query', total_num

            while self.params['offset'] < total_num:
                response = self.make_request()
                try:
                    for business in response['businesses']:
                        self.insert_business(business)
                except:
                    print 'TOO MANY RESTAURANTS IN CATEGORY:'
                    print self.params['category_filter']
                    print response
                self.params['offset'] += 20
                time.sleep(1)
        except:
            print response, self.params['category_filter']


def get_restaurant_metadata(city, db_name, coll_name):
    """
    Loads restaurant data for 'city' into the 'coll_name' collection
    of the 'db_name' database

    INPUT:  city -- city name in Yelp API format (string)
            db_name -- MongoDB name (string)
            coll_name -- MongoDB collection name (string)

    OUTPUT: None
    """

    categories = []
    neighborhoods = []

    with open('../data/restaurants.txt') as f:
        for line in f:
            temp = line.split(",")
            temp = temp[0].split("(")
            categories.append(temp[-1])

    for category in categories:

        PARAMS = {'location': city,
                  'term': 'restaurants',
                  'category_filter': category,
                  'limit': 20,
                  'offset': 0}

        yelp_meta = GetMeta(db_name, coll_name,
                            KEY, SECRET_KEY, TOKEN,
                            SECRET_TOKEN, PARAMS)
        yelp_meta.run()


if __name__ == '__main__':
    """
    Command line arguments are:
    1) city name in Yelp API format (string)
    2) MongoDB name (string)
    3) MongoDB collection name (string)
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', type=str, nargs=3)
    my_input = parser.parse_args().files

    get_city_metadata(my_input[0], my_input[1], my_input[2])
