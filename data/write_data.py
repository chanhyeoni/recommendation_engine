#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
write_data.py

A backend script to wrtie the data and load the data into the database
There are some costumized functions to deal with a different kind of dataset. Here
The defined functions are used for munging the best buy dataset.
"""
import pandas as pd
import preprocessing as preproc
reload(preproc)

def write_bestbuy_review_data(engine, api = 'http://api.remix.bestbuy.com/v1/reviews?format=json&pageSize=100&apiKey=q3yfbu6smh6bzydeqbjv9kas'):
    """ preprocess the "NEW" Best Buy review data frame """
    # delete the unnecesssary records
    reviews_data = preproc.getData_API(api, 'reviews')
    if('aboutMe' in reviews_data.columns):
        del reviews_data['aboutMe']
        
    users = reviews_data['reviewer']
    
    # change the reviewer's elements into strings    
    for i in range(0, len(users)):
        name = users[i][0]['name']
        reviews_data['reviewer'].iloc[i] = name    
        
    reviews_data.rename(columns = {'id': 'product_id'}, inplace = True)
    reviews_data['submissionTime'] = reviews_data['submissionTime'].apply(lambda x: x.replace('T', ' '))
    reviews_data['rating']= reviews_data['rating'].where(pd.notnull(reviews_data['rating']), 0)
    reviews_data['rating'] = preproc.change_type(reviews_data['rating'], float)
    reviews_data.to_sql('bestbuy_data', con = engine, index = False, if_exists = 'append')

def write_bestbuy_product_data(engine, api = 'http://api.remix.bestbuy.com/v1/products?format=json&pageSize=100&apiKey=q3yfbu6smh6bzydeqbjv9kas'):
    """ preprocess the "NEW" Best Buy review data frame """
    # delete the unnecesssary records
    products_data = preproc.getData_API(api, 'products')
    del products_data['videoChapters']
    del products_data['videoLanguages']
    products_data = products_data.where(pd.notnull(products_data), None)
    print products_data
    print products_data.columns
    # products_data.to_sql('bestbuy_products_data', con = engine, index = False, if_exists = 'replace')

def obtain_USAToday_APIs(api="http://api.usatoday.com/open/reviews/music?count=1000&api_key=mhph6f4afgvetbqtex4rs22a"):
    """ uses the USA Today API in order to extract the list of the sub-APIs for music reviews """
    import requests
    r = requests.get(api)
    jsonfile = r.json()    
    key = jsonfile.keys()    
    data = jsonfile[key[0]]
    keys = jsonfile[key[0]].keys()    
    api_dict = {}
    nData = '?count=1000&'
    for i in keys:
        ls_str = data[i].split('?')
        new_one = ls_str[0]+nData+ls_str[1]
        if(api_dict.has_key(i)):
            continue
        else:
            api_dict[i] = new_one
    return api_dict

def write_USAData(engine, parameter):
    # USA Today data
    api_list = obtain_USAToday_APIs()
    aData_USA = preproc.getData_API(api_list[parameter], 'MusicReviews')
    aData_USA['Rating']= aData_USA ['Rating'].where(pd.notnull(aData_USA['Rating']), 0)
    aData_USA['Rating'] = preproc.change_type(aData_USA['Rating'], float)
    aData_USA.to_sql(name = 'USA_Today_data', con = engine, index = False, if_exists = 'append')

def update_data():
    # configure the SQL engines
    # to use mysql, you need to go thorugh this grant priveliges steop
    engine = preproc.get_db_engine(dialect_driver = 'mysql', dbname = 'recommender')   
    write_bestbuy_review_data(engine)
    write_USAData(engine, 'List of available reviews from 2007...')
    # write_bestbuy_product_data(engine)    