#!/usr/bin/python
# -*- coding: utf8 -*-

from flask import Flask, url_for, redirect
from flask import render_template
from flask import request
from gensim import similarities
from geopy.distance import vincenty
import libs.yelp.search as yelp
import json
import pickle

KEY = "YtYo3-wypceY7A63JaJPAw"
SECRET_KEY = "xLX6MySk_gFK3cGI_FzwethvqDU"
TOKEN = "BdQOXSrzwX6SO6vuAuzHLCUNOmGEMxCj"
SECRET_TOKEN = "1Xcrnf4VVGWQYk-bOWGacaipNtc"

TARGET_DISTANCE = 2

app = Flask(__name__)

INDEX = similarities.MatrixSimilarity.load('data/w2v_matrix_wgt.index')
with open('data/idx_name.obj', 'r') as f:
    IDX_NAME = pickle.load(f)

with open('data/name_idx.obj', 'r') as g:
    NAME_IDX = pickle.load(g)

with open('data/w2v_matrix_wgt.obj', 'r') as h:
    BIG_MATRIX = pickle.load(h)

@app.route('/hello/')
@app.route('/hello/<name>')
def hello(name=None):
    return render_template('hello.html', name=name)

@app.route('/restaurant_query')
def restaurant_query():
    return render_template('videotron.html')

@app.route('/get_similarities', methods=['POST'] )
def get_similarities():
    # get data from request form, the key is the name you set in your form

    local_loc = request.form['dist']
    parsed = local_loc.split(",")
    local_search = int(parsed[0])
    current_location = tuple(map(float,parsed[1:]))

    search_string = request.form['user_input'].lower()
    target = request.form['state']
    name = '+'.join(search_string.split(',')[0].split()).encode('utf8')

    try:
        state = search_string.split(',')[-2].upper().encode('utf8').strip()
    except:
        return redirect(url_for('restaurant_query'))

    if state == 'CA':
        city = "SAN+FRANCISCO"
    elif state == 'NY':
        city = "NEW+YORK,+NY"
    else:
        return redirect(url_for('restaurant_query'))

    params = {'location': city,
                  'term': name,
                  'category_filter': 'restaurants',
                  'limit': 2,
                  'offset': 0}


    d = yelp.request(params, KEY, SECRET_KEY, TOKEN, SECRET_TOKEN)

    print d

    print len(d['businesses'])
    if len(d['businesses']) < 2:
        biz = d['businesses'][0]['name'].lower().encode('utf8')
    else:
        biz1 = d['businesses'][0]['name'].lower().encode('utf8')
        biz2 = d['businesses'][1]['name'].lower().encode('utf8')

        len_test = min(len(biz1), len(biz2))

        if biz1[:len_test] == biz2[:len_test]:
            biz = biz1[:len_test]#.encode('utf8')
        else:
            biz = biz1#.encode('utf8')


    biz = biz.replace('é','e')
    print biz
    print " ".join(name.split('+'))
    state = 'CA'

    if " ".join(name.split('+'))[:2].lower() != biz[:2]:
        return redirect(url_for('restaurant_query'))

    try:
        ids = NAME_IDX[biz.decode('utf8')]
    except:
        return redirect(url_for('restaurant_query'))

    vec = BIG_MATRIX[ids]
    sims = INDEX[vec]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])

    output = {}
    counter = 0
    len_targs = len(biz)
    if local_search == 0:
        for idx, sim in enumerate(sims):
            if IDX_NAME[sim[0]]['location'] == target and IDX_NAME[sim[0]]['name'].lower()[:len_targs].encode('utf8') != biz:
                #output.append((sim[1], IDX_NAME[sim[0]]['name'].encode('utf8')))
                # output.append((IDX_NAME[sim[0]]['name'].encode('utf8'),
                #                IDX_NAME[sim[0]]['rating'],
                #                IDX_NAME[sim[0]]['snippet'].encode('utf8'),
                #                IDX_NAME[sim[0]]['category'],
                #                IDX_NAME[sim[0]]['url'],
                #                IDX_NAME[sim[0]]['coordinate']['latitude'],
                #                IDX_NAME[sim[0]]['coordinate']['longitude'],
                #                IDX_NAME[sim[0]]['address']))
                output[counter] = IDX_NAME[sim[0]]
                output[counter]["sim_score"] = "{0:.3f}".format(round(sim[1],3))
                counter += 1
                if len(output) == 20:
                    break
    else:
        print "XXXXXXXXXXX"
        for idx, sim in enumerate(sims):
            try:
                geo_location = (IDX_NAME[sim[0]]['coordinate']['latitude'], IDX_NAME[sim[0]]['coordinate']['longitude'])
            except:
                continue
            if IDX_NAME[sim[0]]['location'] == target and IDX_NAME[sim[0]]['name'].lower()[:len_targs].encode('utf8') != biz \
            and vincenty(current_location, geo_location).miles < TARGET_DISTANCE:
                #output.append((sim[1], IDX_NAME[sim[0]]['name'].encode('utf8')))
                # output.append((IDX_NAME[sim[0]]['name'].encode('utf8'),
                #                IDX_NAME[sim[0]]['rating'],
                #                IDX_NAME[sim[0]]['snippet'].encode('utf8'),
                #                IDX_NAME[sim[0]]['category'],
                #                IDX_NAME[sim[0]]['url'],
                #                IDX_NAME[sim[0]]['coordinate']['latitude'],
                #                IDX_NAME[sim[0]]['coordinate']['longitude'],
                #                IDX_NAME[sim[0]]['address']))
                output[counter] = IDX_NAME[sim[0]]
                output[counter]["sim_score"] = "{0:.3f}".format(round(sim[1],3))
                counter += 1
                if len(output) == 15:
                    break

    if len(output) == 0:
        return "No results found"

    output_len = len(output)
    output[output_len] = current_location

    #print output[0]
    #return ''.join(['%.2f <br>' % item[5]['longitude'] for item in output])
    #.decode('utf8')
    #print output
    # fake = {
    #     "name": "per se",
    #     "lat": 100,
    #     "long": 1000,
    #     "url": "www.yahoo.com"
    #     }
    #return fake
    return render_template('top_restaurants.html', top_restaurants=json.dumps(output))
    # # convert data from unicode to string
    # data = str(data)
    
    # # run a simple program that counts all the words
    # dict_counter = {}
    # for word in data.lower().split():
    #     if word not in dict_counter:
    #         dict_counter[word] = 1
    #     else:
    #         dict_counter[word] += 1
    # total_words = len(dict_counter)
    
    # # now return your results 
    # return 'Total words is %i, <br> dict_counter is: %s' % (total_words, dict_counter)
@app.route('/geo')
def geo():
    return render_template('geo.html')
app.debug = True
app.run()