#!/usr/bin/python
"""
main.py
loads the data into the database and uses that for calling some of the functions
calls the recoomendation functions and deploys the model to yhat
"""
import numpy as np
import pandas as pd
import recommender as rec
import preprocessing as preproc
from yhat import Yhat,  BaseModel
reload(preproc)
reload(rec)


""" deploy to yhat """
class Recommender(BaseModel):        
    """ uses the similarity-based recommender model """
    def predict(self, data):
        """  
            data must be the dictionary, whose key is the user, his favorite products,
            and the number of the top k recommended products
            uses similarity-based approach
        """
        aData = self.dataset
        user = data.get("user")
        products = data.get("products")
        top_n = data.get("n")
        
        products_recommended = rec.recommendations_similarity(aData, user, products, n=top_n)
        
        return products_recommended
   
class Graphlab_Recommender(BaseModel):
    """ uses the GraphLab-based recommender model """
    def predict(self, data):
        """  
            data must be the dictionary, whose key is the user,
            and the number of the top k recommended products
            uses GraphLab module
        """
        aData = self.dataset
        user = data.get("user")
        top_n = data.get("n")
        
        results = rec.graphlab_recommendations(aData, user, n=top_n)
        products = results['product_id']

        return {'products': products}

class PySpark_Recommender(BaseModel):
    """ uses the Spark-based recommender model """
    def predict(self, data):
        """
            data must be the dictionary, whose key is the user,
            and the number of the top k recommended products
            uses Spark module
        """
        aData = self.filename
        user = data.get("user")
        products= data.get("products")
        top_n = data.get("n")
        
        results = rec.spark_recommendations(aData, user, products, n=top_n)
        return results
    


if __name__ == "__main__":
    # text part
    engine = preproc.get_db_engine(dialect_driver = 'mysql', dbname = 'recommender')
    sql_command = 'bestbuy_data'
    aData_BB = pd.read_sql(sql=sql_command, con=engine)
    param = {'comment': 'comment', 'ratings': 'rating', 'user_id':'reviewer', 'product_id':'sku'}
    model, ratings_data = rec.sentiment_analysis(aData_BB, param)
    ratings_data = ratings_data.sort(columns = 'user_id')
    ratings_data['user_id'] = ratings_data['user_id'].fillna('anonymous')

    aGraphlab_Model = Graphlab_Recommender(dataset = ratings_data)
    data = {'user': ['13579abcd'], 'products':[6955008], 'n':5}
    print aGraphlab_Model.predict(data)
    
    engine = preproc.get_db_engine(dialect_driver = 'mysql', dbname = 'recommender')
    sql_command = 'yahoo_data'
    yahoo_data = pd.read_sql(sql=sql_command, con=engine)
    yahoo_data.sort(columns = 'user_id', ascending = True, inplace = True) # no pass by value
    data = {'user': [15], 'products':[123764, 71142],  'n':10}    
    
    aGraphlab_Model = Graphlab_Recommender(dataset = yahoo_data)
    print aGraphlab_Model.predict(data)
    
    aPySpark_Model = PySpark_Recommender(filename='./data/yahoo music/train_0.txt')
    print aPySpark_Model.predict(data)
    
    yh = Yhat("chlee021690@gmail.com", "b36b987283a83e5e4d2814af6ef0eda9", "http://cloud.yhathq.com/")
    # yh.deploy("Recommender", Recommender, globals())
    # yh.deploy("Graphlab_Recommender", Graphlab_Recommender, globals())
    # yh.deploy("PySpark_Recommender", PySpark_Recommender, globals())   