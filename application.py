#!/usr/bin/env python

from flask import Flask, request, render_template, url_for, Response, json
from yhat import Yhat
import os, sys
import pandas as pd
lib_path = os.path.abspath("/Users/chlee021690/Desktop/Programming/Python/Recommender System/recommendation engine/recommender scripts")
sys.path.append(lib_path)
import preprocessing as preproc
reload(preproc)

# create the flask app as well as the SQLalchemy database associated with the app
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        yh = Yhat("chlee021690@gmail.com", "b36b987283a83e5e4d2814af6ef0eda9", "http://cloud.yhathq.com/")
        recommender_name = "Final_Recommender"
        data = {"user" : request.json['user'], "products" : request.json['products'], "n": request.json['n']}
        pred = yh.predict(recommender_name, data)
        return Response(json.dumps(pred), mimetype='application/json')
    else:
        # if it is GET method, you just need to render the homepage part
        # defines the jQuery pages in order to render the page in home.html template
        engine = preproc.get_db_engine(dialect_driver = 'mysql', dbname = 'recommender')
        sql_command = 'SELECT product_id FROM bestbuy_data'
        aData = pd.read_sql(sql=sql_command, con=engine)
        aData.to_csv("./static/js/products_data.csv", index = False)
        
        css_url = url_for('static', filename='css/main.css')
        jquery_url = url_for('static', filename='js/jquery-1.11.1.js')
        products_url = url_for('static', filename='js/products.js')
        highlight_url = url_for('static', filename='js/highlight.js')
        js_url = url_for('static', filename='js/main.js')
        return render_template('home.html', css_url=css_url,jquery_url=jquery_url, products_url=products_url,
            js_url=js_url, highlight_url=highlight_url)

if __name__ == '__main__':
    app.run(debug=True) # must be turned off for the production mode