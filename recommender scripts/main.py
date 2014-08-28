#!/usr/bin/python
"""
main.py
loads the data into the database and uses that for calling some of the functions
calls the recoomendation functions and deploys the model to yhat
"""
import os, sys
import random
import pandas as pd
import recommender as rec
import preprocessing as preproc
import sentiment_analysis as sentiment
lib_path = os.path.abspath("/Users/chlee021690/Desktop/Programming/Python/Recommender System/recommendation engine/data")
sys.path.append(lib_path)
import write_data as data_write
from yhat import Yhat,  BaseModel
reload(data_write)
reload(preproc)
reload(rec)



""" deploy to yhat """
class Recommender(BaseModel):        
    """ uses the similarity-based recommender model """
    def predict(self, data):
        """ data must be the dictionary, whose key is the user, his favorite products,
            and the number of the top k recommended products. It uses similarity-based approach """
        aData = self.dataset
        user = data.get("user")
        products = data.get("products")
        top_n = data.get("n")
        
        results = rec.recommendations_similarity(aData, user, products, n=top_n)
        
        return {'products' : results}
   
class Graphlab_Recommender(BaseModel):
    """ uses the GraphLab-based recommender model """
    def predict(self, data):
        """ data must be the dictionary, whose key is the user, and the number of the top k 
            recommended products uses GraphLab module """
        aData = self.dataset
        params = self.needed_param
        user = data.get("user")
        top_n = data.get("n")
        
        results, product_id = rec.graphlab_recommendations(aData = aData, user = user, needed_param = params, n=top_n)
        products = results[product_id]

        return {'products': products}

class PySpark_Recommender(BaseModel):
    """ uses the Spark-based recommender model """
    def predict(self, data):
        """ data must be the dictionary, whose key is the user, and the number of the top k 
            recommended products uses Spark module """
        filepath = self.filename
        user = data.get("user")
        products= data.get("products")
        top_n = data.get("n")
        
        results = rec.spark_recommendations(filepath, user, products, n=top_n)
        return {'products' : results}

class Final_Recommender(BaseModel):
    """ initialized with the list of models (supervised and unsupervised)
        and bring a mixture of results that demonstrate the predicted recommendations
        and guided discoveries """
    def predict(self, data):

        results = []
        models = self.models
        
        for aModel in models:
            for aProduct in aModel.predict(data)['products']:
                results.append(aProduct)
                
        random.shuffle(results) 
        # this is pass by reference
        # or rather than randomly shuffling, we can do some ranking mechanism
        
        return results
        
if __name__ == "__main__":
    # update the data
    # data_write.update_data()
    
    # get the engines
    engine = preproc.get_db_engine(dialect_driver = 'mysql', dbname = 'recommender')
    sql_command = 'bestbuy_data'
    aData = pd.read_sql(sql=sql_command, con=engine)
        
    """ BEST BUY """
    bestbuy_data = aData
    data = {'user': ['13579abcd'], 'products':[6955008], 'n':5}
    param = {'comment': 'comment', 'ratings': 'rating', 'user_id':'reviewer', 'product_id':'product_id'}
    
    # textual analytics + CF method
    predicted_ratings_data = sentiment.sentiment_analysis_unsupervised(bestbuy_data, param)    
    predicted_ratings_data = predicted_ratings_data.sort(columns = param['user_id'])
    predicted_ratings_data[param['user_id']] = predicted_ratings_data[param['user_id']].fillna('anonymous')
    textBased_Model = Graphlab_Recommender(dataset = predicted_ratings_data, needed_param = param) 
    
    # ratings-based CF methods
    ratingBased_Model = Graphlab_Recommender(dataset = bestbuy_data, needed_param = param)
    
    # test it
    models = [textBased_Model, ratingBased_Model]
    final_Model = Final_Recommender(models = models)
    print final_Model.predict(data)
    
    '''
    """ YAHOO """
    yahoo_data = aData
    yahoo_data.sort(columns = 'user_id', ascending = True, inplace = True) # no pass by value
    
    # rating-based CF recommendations
    data = {'user': [15], 'products':[123764, 71142],  'n':10}        
    aGraphlab_Model = Graphlab_Recommender(dataset = yahoo_data)
    print aGraphlab_Model.predict(data)
        
    """ USA TODAY """
    # rating-based CF recommendations
    usaToday_data = aData
    param = {'user_id':'Reviewer', 'product_id':'Id', 'ratings': 'Rating'}
    data = {'user': ['Edna Gundersen'], 'products':[123901],  'n':10}  
    aGraphlab_Model = Graphlab_Recommender(dataset = usaToday_data, needed_param = param)
    print aGraphlab_Model.predict(data)
    
    # textual analytics + CF method
    param = {'comment': 'Brief', 'ratings': 'Rating', 'user_id':'Reviewer', 'product_id':'Id'}
    model, ratings_data = rec.sentiment_analysis_regress(usaToday_data, param)
    ratings_data = ratings_data.sort(columns = 'user_id')
    ratings_data['user_id'] = ratings_data['user_id'].fillna('anonymous')
    print ratings_data
    
    aGraphlab_Model = Graphlab_Recommender(dataset = ratings_data)
    data = {'user': ['Edna Gundersen'], 'products':[123901],  'n':10} 
    print aGraphlab_Model.predict(data)
    '''
    
    # deployment
    yh = Yhat("chlee021690@gmail.com", "b36b987283a83e5e4d2814af6ef0eda9", "http://cloud.yhathq.com/")
    yh.deploy("Final_Recommender", Final_Recommender, globals()) 