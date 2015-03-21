#!/usr/bin/python
# -*- coding: utf8 -*-

from flask import Flask, url_for, redirect
from flask import render_template
from flask import request
from gensim import similarities, models, corpora
from geopy.distance import vincenty
import libs.scraper as scrape
import libs.yelp.search as yelp
import numpy as np
import json
import pickle

KEY = "YtYo3-wypceY7A63JaJPAw"
SECRET_KEY = "xLX6MySk_gFK3cGI_FzwethvqDU"
TOKEN = "BdQOXSrzwX6SO6vuAuzHLCUNOmGEMxCj"
SECRET_TOKEN = "1Xcrnf4VVGWQYk-bOWGacaipNtc"

INDEX = similarities.MatrixSimilarity.load('data/w2v_matrix_wgt.index')
BIGRAMS = models.Phrases.load('data/bigrams')
DICTIONARY = corpora.Dictionary.load('data/restaurants.dict')
TFIDF = models.TfidfModel.load('data/model.tfidf')
WORD2VEC = models.Word2Vec.load('data/mymodel')

with open('data/idx_name.obj', 'r') as f:
    IDX_NAME = pickle.load(f)

with open('data/name_idx.obj', 'r') as g:
    NAME_IDX = pickle.load(g)

with open('data/w2v_matrix_wgt.obj', 'r') as h:
    BIG_MATRIX = pickle.load(h)

app = Flask(__name__)


@app.route('/')
def restaurant_query():
    return render_template('videotron.html')


@app.route('/krdooley')
def krdooley():
    return render_template('krdooley.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/get_similarities', methods=['POST'])
def get_similarities():

    local_loc = request.form['dist']
    parsed = local_loc.split(",")
    local_search = int(parsed[0])
    current_location = tuple(map(float, parsed[1:]))

    if local_search == 0:
        target_distance = 500
    elif local_search == 1:
        target_distance = 7
    else:
        target_distance = 2

    search_string = request.form['user_input'].lower()
    print search_string
    target = request.form['state']
    name = '+'.join(search_string.split(',')[0].split()).encode('utf8')

    try:
        temp = [item.strip().upper() for item in search_string.split(',')[-3:-1]]
        temp[0] += ","
        temp = ["+".join(item.split()) for item in temp]
        city = "+".join(temp).encode('utf8')
    except:
        return redirect(url_for('restaurant_query'))

    params = {'location': city,
              'term': name,
              'category_filter': 'restaurants',
              'limit': 2,
              'offset': 0}

    d = yelp.request(params, KEY, SECRET_KEY, TOKEN, SECRET_TOKEN)

    if len(d['businesses']) == 0:
        print "Not in Yelp"
        return redirect(url_for('restaurant_query'))
    elif len(d['businesses']) < 2:
        biz = d['businesses'][0]['name'].lower().encode('utf8')
        long_review = 0
    else:
        long_review = np.argmax([item['review_count'] for item
                                in d['businesses']])
        biz1 = d['businesses'][0]['name'].lower().encode('utf8')
        biz2 = d['businesses'][1]['name'].lower().encode('utf8')

        len_test = min(len(biz1), len(biz2))

        if biz1[:len_test] == biz2[:len_test]:
            biz = biz2 if long_review else biz1
        else:
            biz = biz1

    biz = biz.replace('é', 'e')

    if " ".join(name.split('+'))[:2].lower() != biz[:2]:
        print "Wrong restaurant"
        return redirect(url_for('restaurant_query'))

    try:
        ids = NAME_IDX[biz.decode('utf8')]
        vec = BIG_MATRIX[ids]
    except:
        vec = scrape.get_search_vector(d['businesses'][long_review],
                                       BIGRAMS, DICTIONARY, TFIDF, WORD2VEC)
        if len(vec) == 0:
            print "Could not find reviews"
            return redirect(url_for('restaurant_query'))

    sims = INDEX[vec]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])

    output = {}
    counter = 0
    len_targs = len(biz)
    if local_search == 0:
        for idx, sim in enumerate(sims):
            if IDX_NAME[sim[0]]['location'] == target and \
            IDX_NAME[sim[0]]['name'].lower()[:len_targs].encode('utf8') != biz:
                output[counter] = IDX_NAME[sim[0]]
                output[counter]["sim_score"] = "{0:.3f}".format(round(sim[1], 3))
                counter += 1
                if counter >= 20:
                    break
    else:
        for idx, sim in enumerate(sims):
            try:
                geo_location = (IDX_NAME[sim[0]]['coordinate']['latitude'],
                                IDX_NAME[sim[0]]['coordinate']['longitude'])
            except:
                continue
            if IDX_NAME[sim[0]]['location'] == target and \
            IDX_NAME[sim[0]]['name'].lower()[:len_targs].encode('utf8') != biz \
            and vincenty(current_location, geo_location).miles < target_distance:
                output[counter] = IDX_NAME[sim[0]]
                output[counter]["sim_score"] = "{0:.3f}".format(round(sim[1], 3))
                counter += 1
                if counter >= 20:
                    break

    if len(output) == 0:
        print "No results found"
        return redirect(url_for('restaurant_query'))

    return render_template('top_restaurants.html',
                           top_restaurants=json.dumps(output))

app.debug = True
app.run()
