from flask import Flask, url_for
from flask import render_template
from flask import request
from gensim import similarities
import yelp.search as yelp
import json
import pickle

KEY = "YtYo3-wypceY7A63JaJPAw"
SECRET_KEY = "xLX6MySk_gFK3cGI_FzwethvqDU"
TOKEN = "BdQOXSrzwX6SO6vuAuzHLCUNOmGEMxCj"
SECRET_TOKEN = "1Xcrnf4VVGWQYk-bOWGacaipNtc"

app = Flask(__name__)

INDEX = similarities.MatrixSimilarity.load('data/big_matrix.index')
with open('data/idx_name.obj', 'r') as f:
    IDX_NAME = pickle.load(f)

with open('data/name_idx.obj', 'r') as g:
    NAME_IDX = pickle.load(g)

with open('data/big_matrix.obj', 'r') as h:
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
    name = request.form['user_input'].lower()
    print name.split(',')[0]
    state ='CA'
    ids = NAME_IDX[name]

    vec = BIG_MATRIX[ids]
    sims = INDEX[vec]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])

    output = {}
    counter = 0
    for idx, sim in enumerate(sims):
        if IDX_NAME[sim[0]]['location'] == state:
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
    #print output[0]
    #return ''.join(['%.2f <br>' % item[5]['longitude'] for item in output])
    #.decode('utf8')
    print output
    fake = {
        "name": "per se",
        "lat": 100,
        "long": 1000,
        "url": "www.yahoo.com"
        }
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

app.debug = True
app.run()