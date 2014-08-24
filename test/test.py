import pandas as pd
import recommender as rec
import my_preprocess as my_preproc
import database as db
from yhat import Yhat, BaseModel, preprocess
reload(my_preproc)
reload(db)
reload(rec)

def test_demo():
    """ make the data frame set for the collaborative filtering part """
    # split the dataset for the graphlab
    yahoo_train_SFrame, yahoo_test_SFrame = my_preproc.graphlab_split_data(aData_Yahoo, 0.6)
    yahoo_train_SFrame.save("yahoo_train.csv")
    yahoo_test_SFrame.save("yahoo_test.csv")
    
    BBData_train, BBData_test = my_preproc.graphlab_split_data(aData_BB, 0.6)
    
    """ test the matrix factorization recommendation based upon the hyperparameter oprtimization (GraphLab)"""
    import graphlab as gl
    
    columns = yahoo_train_SFrame.column_names()
    user_id = columns[0]
    product_id = columns[1]
    values = columns[2]
    
    env = gl.deploy.environment.Local('local_env')
    standard_param = {'observation_data': yahoo_train_SFrame, 'user_column': user_id, 'item_column': product_id, 'target_column': values}
    hyper_param = {'n_factors':[10,11,12,13,14], 'regularization':[0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
            'nmf':[True, False], 'unobserved_rating_regularization': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
            }
    
    gl.toolkits.model_parameter_search(environment = env, model_factory = gl.recommender.create, 
                                       train_set = 'yahoo_train.csv', save_path = 'hyperparameter_search_result.csv', test_set= 'yahoo_test.csv', 
                                       standard_model_params = standard_param, hyper_params = hyper_param)


    """  test the spark-based recommendations """
    from pyspark import * 
    import pyspark.mllib.recommendation as spark_rec
    filename = "./yahoo music/train_0.txt"
    sc = SparkContext('local', 'spark_rec')
    aData = sc.textFile(filename)
    ratings = aData.map(lambda line: np.array([float(x) for x in line.split('\t')])) 
    # to do this, it assumes that each line of the file consists of [user, product, rating]
    
    rank = 10
    numIterations = 20
    aModel = spark_rec.ALS.train(ratings, rank, numIterations)
    
    """ test the recommendation """
    users_BB =  table_CF_bestbuy.columns
    products_BB =  table_CF_bestbuy.index
    
    users_yahoo =  pd.Series(table_CF_yahoo.columns)
    products_yahoo =  pd.Series(table_CF_yahoo.index)

    # smilarity testing
    print rec.recommendations_similarity(table_CF_bestbuy, users_BB[0], products_BB[range(0,2)], simfunc = rec.sim_cosine)
    print rec.recommendations_similarity(table_CF_yahoo, users_yahoo[0], products_yahoo[range(0,2)], simfunc = rec.sim_cosine)
    
    # matrix factorization using GraphLab
    BB_aModel, BB_results = rec.graphlab_recommendations(BBData_train, [users_BB[1]])
    print BB_results
    print rec.GL_score(BB_aModel, BBData_test)    
    
    yahoo_aModel, results = rec.graphlab_recommendations(yahoo_train_SFrame, [users_yahoo[0]])
    # print yahoo_results
    print rec.graphlab_scores(yahoo_aModel, yahoo_test_SFrame)

    # test yhat demo
    aRecommender = Recommender()
    my_data = {"table": table_CF_yahoo, "user": users_yahoo[2], "products": products_yahoo[range(0,2)], "n": 10}
    products_recommended = aRecommender.recommend_products(my_data)
    print products_recommended
    gl_data = {"table": aData_Yahoo, "user": [users_yahoo[2]], "n": 10}
    gl_products_recommended = aRecommender.graphlab_recommend_products(gl_data)


 if __name__ == "__name__":
 	test_demo()
    print gl_products_recommended