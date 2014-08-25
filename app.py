#!/usr/bin/env python

from flask import Flask, request, render_template, url_for, Response, json
from yhat import Yhat
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
        css_url = url_for('static', filename='css/main.css')
        jquery_url = url_for('static', filename='js/jquery-1.10.2.min.js')
        products_url = url_for('static', filename='js/products.js')
        highlight_url = url_for('static', filename='js/highlight.js')
        js_url = url_for('static', filename='js/main.js')
        return render_template('home.html', css_url=css_url,jquery_url=jquery_url, products_url=products_url,
            js_url=js_url, highlight_url=highlight_url)

if __name__ == '__main__':
    app.run(debug=True) # must be turned off for the production mode