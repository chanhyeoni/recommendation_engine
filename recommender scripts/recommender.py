#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Recommendation Engine module for the Best Buy products data
Created on Wed Aug  6 14:32:57 2014
@author: chlee021690
"""
import os
import sys
import preprocessing as preproc
# for adding the spark module, you need this path
os.environ['SPARK_HOME']="/Users/chlee021690/Desktop/Programming/spark"
sys.path.append("/Users/chlee021690/Desktop/Programming/spark/python/")

import pyspark
import sklearn.metrics as ml_metrics
import graphlab as gl
import pandas as pd
import numpy as np
import recsys


def sim_euclidean(table_CF):
    """ 
        the idea behind this
        use Euclidean distance in order to estiamte the distance between the ratings, from both the person 1 and the person 2
        but the 'similar' data will then have very small distances between each
        so we are going to add 1 and invert the value(reciprocal)
        
        will return m by m data frame (m is the number of users)
    """
    aCopy = table_CF.copy()
    #aCopy = my_preproc.mean_imputation(table_CF)
    aCopy = aCopy.fillna(0)
    euc_dist = ml_metrics.pairwise.euclidean_distances(aCopy)
    euc_dist = pd.DataFrame(euc_dist, columns = aCopy.index)
    euc_dist.index = euc_dist.columns
    return euc_dist

def sim_cosine(table_CF):
    """ 
        the idea behind this 
        measure the similarity of two vectors of an inner product space that measures the cosine of the angle between them.
        The cosine of 0° is 1, and it is less than 1 for any other angle. It is thus a judgement of orientation and not magnitude:
        two vectors with the same orientation have a Cosine similarity of 1, two vectors at 90° have a similarity of 0, and 
        two vectors diametrically opposed have a similarity of -1, independent of their magnitude.
        
        returns the n by n data frame, whose n is the number of products
        (used for item-based collaborative filtering)
    """
    aCopy = table_CF.copy()
    #aCopy = my_preproc.mean_imputation(table_CF)
    aCopy = aCopy.fillna(0)
    cos_dist = ml_metrics.pairwise.cosine_similarity(aCopy)
    cos_dist = pd.DataFrame(cos_dist, columns = aCopy.index)
    cos_dist.index = cos_dist.columns
    return cos_dist

def sim_pearson(table_CF):
    """ 
        Returns the Pearson correlation coefficient for person 1 (p1) and person 2 (p2) 
        The idea behind this:
        two users in the x and y axes, the data points are the ratings for each item (movie)
        so if they are 'similar', it is true that that the points will be fit into a linear line, 
        because they score similarily shows much better score and much more intuitive 
        
        It is more effective to do user-based system than do item-based system
    """
    aCopy = table_CF.copy()
    #aCopy = my_preproc.mean_imputation(table_CF)
    aCopy = aCopy.fillna(0)
    means = np.array(aCopy.apply(lambda x: np.mean(x), axis = 0))
    means = pd.DataFrame(np.tile(means, (len(aCopy.index), 1)))
    diff = np.absolute(np.subtract(aCopy, means))
    numer = diff.dot(diff.T)
    
    sqrt_part = np.sqrt(np.sum(np.square(diff), axis = 1))
    deno = sqrt_part.dot(sqrt_part.T) # THIS IS THE PROBLEM HERE!!!!!!
    
    r = np.divide(numer, deno)
    
    return r

def recommendations_similarity(aData, needed_param, user, products, n = 10, simfunc = sim_cosine):
    """ 
        recommends the top n products based upon the simlarity data frame (sim_measures_table)
        in test demo, products MUST be the products that a particular user has rated the highest   
    """
    table_CF = preproc.make_CF_table(aData, needed_param)
    sim_measures_table = simfunc(table_CF)    
    
    scores = sim_measures_table.dot(table_CF)
    mean_scores = np.array(np.sum(sim_measures_table, axis=1).T)
    mean_scores = pd.DataFrame(np.tile(mean_scores, (scores.shape[1],1))).T
    predicted_ratings = np.divide(scores, np.absolute(mean_scores))
    
    ratings = predicted_ratings[user].order(ascending= False)
    ratings = ratings[0:n]
    
    return (ratings.index[ratings.index.isin(products)==False])

def get_UserMatches(similarity,table_CF, user,n=5):
    """ ranking the users based on how similar they are to the particular person """
    similarity_table = similarity(table_CF)
    similar_users = similarity_table[user]
    # Sort the list so the highest scores appear at the top
    similar_users = similar_users.order(ascending = False)
    return similar_users[0:n]
    
def transformPrefs(prefs):
    ''' transform the original dictionary to the dictionary whose key is now the item, and the value is the dictinary
    of the costumer with the ratings '''
    
    result={}
    for person in prefs:
        for item in prefs[person]:
            result.setdefault(item,{})
            # Flip item and person
            result[item][person]=prefs[person][item]
    return result

def spark_recommendations(filename,  user, products, separator = '\t', n = 10):
    """This method employs the collaborative filtering method from Apache Spark module (pyspark)"""
    """hyperparameter optimizations???? """
    sc = pyspark.SparkContext('loc', 'pyspark_rec')
    aData = sc.textFile(filename)
    data = aData.map(lambda line: np.array([float(x) for x in line.split(separator)])) 
    # to do this, it assumes that each line of the file consists of [user, product, rating]
    
    numIterations = 20
    aModel = pyspark.mllib.recommendation.ALS.train(data, n, numIterations)
    aRDDresults = aModel.predict(user, products)

    return aModel, aRDDresults

def graphlab_recommendations(aData, user, needed_param, n = 10, cv_ratio = 0.7):
    """
        This method uses the recommendation methods from the GraphLab framework.
        For the matrix factorization model, the hyperparameter search method was included, and for doing that, an extra parameter was added
        to the function : the cross-validation ratio. The default value for the CV ratio is 0.7.
    """
    # change the data into SFrame and the user data into SArray
    import preprocessing
    aData = gl.SFrame(aData)
    train, test= preprocessing.graphlab_split_data(aData, cv_ratio)
    user = gl.SArray(user)

    user_id = needed_param['user_id']
    print user_id
    product_id = needed_param['product_id']
    values = needed_param['ratings']
    
    # make models
    methods = ['matrix_factorization', 'linear_model', 'item_similarity', 'popularity', 'item_means']
    sim_type = ['jaccard', 'cosine', 'pearson']
    models = []
    for aMethod in methods:
        if(aMethod != 'item_similarity'):
            model = gl.recommender.create(observation_data = train, user_column = user_id, 
                                          item_column = product_id, target_column = values, method = aMethod)
            models.append(model)
        else:
            for aSim in sim_type:
                sim_model = gl.recommender.create(observation_data = train, user_column = user_id, 
                                          item_column = product_id, target_column = values, method = aMethod, similarity_type = aSim)
                models.append(sim_model)
    
    # generate results for models as well as the rmse results
    recommended = []
    rmse = []
    for model in models:
        aResult = model.recommend(users = user, k = n)
        recommended.append(aResult)
        aRMSE = gl.evaluation.rmse(test[values], model.score(test))
        rmse.append(aRMSE)
        
    # create DataFrame
    df = pd.DataFrame({'models':models, 'recommended':recommended, 'rmse':rmse})
    # find the model that gives k least square errors
    df = df.sort('rmse', ascending = True).iloc[0:2]
    df.index = range(0,2)
    
    colnames = df['recommended'].loc[0].column_names()
    results = pd.DataFrame(columns = colnames)
    
    for aResult in df['recommended']:
        aResult = aResult.to_dataframe()
        results = results.append(aResult)
    
    results = results.sort('score', ascending = False)

    return results.sort('score', ascending=False), product_id