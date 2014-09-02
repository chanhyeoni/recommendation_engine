"""
    test.py
    This script tests the recommender system using the best buy datase

"""
import pandas as pd
import recommender as rec
import my_preprocess as my_preproc
import database as db
from yhat import Yhat, BaseModel, preprocess
reload(my_preproc)
reload(db)
reload(rec)

def test_demo(data):
    """ make the data frame set for the collaborative filtering part """
    # split the dataset for the graphlab
        # get the engines
    engine = preproc.get_db_engine(dialect_driver = 'mysql', dbname = 'recommender')
    sql_command = 'bestbuy_data'
    aData = pd.read_sql(sql=sql_command, con=engine)
    
    """ BEST BUY """
    bestbuy_data = aData
    param = {'comment': 'comment', 'ratings': 'rating', 'user_id':'reviewer', 'item_id':'product_id'}
    
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

 if __name__ == "__name__":
    data = {'user': '13579abcd', 'products':6955008, 'n':5}
 	test_demo(data)