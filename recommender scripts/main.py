#!/usr/bin/python
"""
main.py
loads the data into the database and uses that for calling some of the functions
calls the recoomendation functions and deploys the model to yhat
"""
import random
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
        params = self.needed_param
        user = data.get("user")
        top_n = data.get("n")
        
        results, product_id = rec.graphlab_recommendations(aData = aData, user = user, needed_param = params, n=top_n)
        products = results[product_id]

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
    
class Final_Recommender(BaseModel):
    
    def predict(self, data):
        """ 
            initialized with the list of models and bring a mixture of results
        """
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
    # get the engines
    engine = preproc.get_db_engine(dialect_driver = 'mysql', dbname = 'recommender')
    
    
    """ BEST BUY """
    sql_command = 'bestbuy_data'
    aData_BB = pd.read_sql(sql=sql_command, con=engine)
    data = {'user': ['13579abcd'], 'products':[6955008], 'n':5}
    
    # textual analytics + CF method
    param = {'comment': 'comment', 'ratings': 'rating', 'user_id':'reviewer', 'product_id':'product_id'}
    model, ratings_data = rec.sentiment_analysis_classifier(aData_BB, param)
    ratings_data = ratings_data.sort(columns = 'user_id')
    ratings_data['user_id'] = ratings_data['user_id'].fillna('anonymous')
    cols = ratings_data.columns
    ratings_param = {}
    for i in cols:
        if(ratings_param.has_key(i)):
            continue
        else:
            ratings_param[i]=i
    textBased_Model = Graphlab_Recommender(dataset = ratings_data, needed_param = ratings_param)    
    
    # ratings-based CF methods
    param = {'user_id':'reviewer', 'product_id':'product_id', 'ratings': 'rating'}
    ratingBased_Model = Graphlab_Recommender(dataset = aData_BB, needed_param = param)
    
    models = [textBased_Model, ratingBased_Model]
    
    final_Model = Final_Recommender(models = models)
    print final_Model.predict(data)
    
    ''' 
    """ YAHOO """
    sql_command = 'yahoo_data'
    yahoo_data = pd.read_sql(sql=sql_command, con=engine)    
    yahoo_data.sort(columns = 'user_id', ascending = True, inplace = True) # no pass by value
    
    # rating-based CF recommendations
    data = {'user': [15], 'products':[123764, 71142],  'n':10}        
    aGraphlab_Model = Graphlab_Recommender(dataset = yahoo_data)
    print aGraphlab_Model.predict(data)
    

    """ USA TODAY """
    # rating-based CF recommendations
    sql_command = 'USA_Today_data'
    usaToday_data = pd.read_sql(sql=sql_command, con=engine)
    param = {'user_id':'Reviewer', 'product_id':'Id', 'ratings': 'Rating'}
    data = {'user': ['Edna Gundersen'], 'products':[123901],  'n':10}  
    aGraphlab_Model = Graphlab_Recommender(dataset = usaToday_data, needed_param = param)
    print aGraphlab_Model.predict(data)
    ''' 
    
    """
    # textual analytics + CF method
    param = {'comment': 'Brief', 'ratings': 'Rating', 'user_id':'Reviewer', 'product_id':'Id'}
    model, ratings_data = rec.sentiment_analysis_regress(usaToday_data, param)
    ratings_data = ratings_data.sort(columns = 'user_id')
    ratings_data['user_id'] = ratings_data['user_id'].fillna('anonymous')
    print ratings_data
    
    aGraphlab_Model = Graphlab_Recommender(dataset = ratings_data)
    data = {'user': ['Edna Gundersen'], 'products':[123901],  'n':10} 
    print aGraphlab_Model.predict(data)
    """
    
    # deployment
    yh = Yhat("chlee021690@gmail.com", "b36b987283a83e5e4d2814af6ef0eda9", "http://cloud.yhathq.com/")
    yh.deploy("Final_Recommender", Final_Recommender, globals()) 