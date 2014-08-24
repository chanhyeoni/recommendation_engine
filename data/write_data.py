#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
write_data.py

A backend script to wrtie the data and load the data into the database

There are some costumized functions to deal with a different kind of dataset. Here
The defined functions are used for munging the best buy dataset.

"""
import pandas as pd
import os, sys
# to use the preprocessing script
lib_path = os.path.abspath("/Users/chlee021690/Desktop/Programming/Python/Recommender System/recommendation engine/recommender scripts")
sys.path.append(lib_path)
import preprocessing as preproc
reload(preproc)

def write_bestbuyData(reviews_data, engine):
    """ preprocess the "NEW" Best Buy review data frame """
    # delete the unnecesssary records
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
    reviews_data['rating'] = preproc.change_type(reviews_data['rating'], int)
    reviews_data.to_sql('bestbuy_data', con = engine, index = False, if_exists = 'append')


def obtain_USAToday_APIs(api="http://api.usatoday.com/open/reviews/music?count=1000&api_key=mhph6f4afgvetbqtex4rs22a"):
    """ 
        uses the USA Today API in order to extract the list of the sub-APIs for music reviews
        
    """
    import requests

    r = requests.get(api)
    jsonfile = r.json()
    
    key = jsonfile.keys()
    
    data = jsonfile[key[0]]
    keys = jsonfile[key[0]].keys()
    
    api_list = []
    nData = '?count=1000&'
    for i in keys:
        ls_str = data[i].split('?')
        new_one = ls_str[0]+nData+ls_str[1]
        api_list.append(new_one)
    
    return api_list
        

if __name__ == '__main__':
    # configure the SQL engines
    # to use mysql, you need to go thorugh this grant priveliges steop
    engine = preproc.get_db_engine(dialect_driver = 'mysql', dbname = 'recommender')   
    
    # best buy data (it uses the api)
    api='http://api.remix.bestbuy.com/v1/reviews?format=json&pageSize=100&apiKey=q3yfbu6smh6bzydeqbjv9kas'
    new_reviews_BB = preproc.getData_API(api, 'reviews')
    write_bestbuyData(new_reviews_BB, engine)
    
    # USA Today data
    api_list = obtain_USAToday_APIs()
    aData_USA = preproc.getData_API(api_list[4], 'MusicReviews')
    aData_USA.to_sql(name = 'USA_Today_data', con = engine, index = False, if_exists = 'append')
    