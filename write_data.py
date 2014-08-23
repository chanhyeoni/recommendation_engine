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

def bestBuy_preprocess(reviews_data):
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
    reviews_data = reviews_data.where(pd.notnull(reviews_data), None)
    
    #reviews_data.to_csv("./data/best_buy.csv", mode = 'a', index = False)

def get_updated_BBData(new_reviews_data):
    """ get the updated Best Buy Data """
    bestBuy_preprocess(new_reviews_data)
    aData_BB = pd.read_csv("./data/best_buy.csv")
    aData_BB = aData_BB.where(pd.notnull(aData_BB), None)
    
    return aData_BB

if __name__ == '__main__':
    """
    filenames = ['train_0.txt']
    aData_Yahoo = preproc.getMultipleData_Static("./data/yahoo music/", filenames, "\t", 
                colnames = ['user_id', 'product_id', 'ratings'],nRowsToRead=40000)
    """
    aData_Yahoo = pd.read_csv("./data/yahoo_data.csv")    
    # best buy data (it uses the api)
    api='http://api.remix.bestbuy.com/v1/reviews?format=json&pageSize=100&apiKey=q3yfbu6smh6bzydeqbjv9kas'
    new_reviews_BB = preproc.getData_API(api, 'reviews')
    aData_BB = get_updated_BBData(new_reviews_BB)
    # configure the SQL engines
    # to use mysql, you need to go thorugh this grant priveliges steop
    engine = preproc.get_db_engine(dialect_driver = 'mysql', dbname = 'recommender')
    
    # write to the sql file
    ls_data = [aData_BB]
    ls_tablename = ['bestbuy_data']
    preproc.write_sql(engine, ls_data, ls_tablename, exists_command = 'replace')
    
    