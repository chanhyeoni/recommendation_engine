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
import sklearn.feature_extraction as ml_feature_extract
import sklearn.linear_model as lm
import sklearn.tree as tree
import sklearn.svm as svm
import sklearn.naive_bayes as bayes
import sklearn.isotonic as isotonic
import sklearn.grid_search as gs
import sklearn.cross_validation as cv
import sklearn.ensemble as ensemble
import nltk.classify as nltk_classify
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

def recommendations_similarity(table_CF, user, products, n = 10, simfunc = sim_cosine):
    """ 
        recommends the top n products based upon the simlarity data frame (sim_measures_table)
        in test demo, products MUST be the products that a particular user has rated the highest   
    """
    sim_measures_table = simfunc(table_CF)    
    
    scores = sim_measures_table.dot(table_CF)
    mean_scores = np.array(np.sum(sim_measures_table, axis=1).T)
    mean_scores = pd.DataFrame(np.tile(mean_scores, (scores.shape[1],1))).T
    predicted_ratings = np.divide(scores, np.absolute(mean_scores))
    
    ratings = predicted_ratings[user].order(ascending= False)
    ratings = ratings[0:n]
    
    return (ratings.index[ratings.index.isin(products)==False])

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

def graphlab_recommendations(aData, user, n = 10, cv_ratio = 0.7):
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
    # get the column names
    data_columns = aData.column_names()
    user_id = data_columns[0]
    product_id = data_columns[1]
    values = data_columns[2]
    
    """
    # hyperparameter tuning
    train.save('./data/graphlab_train')
    test.save('./data/graphlab_test')
    env = gl.deploy.environment.Local('b')
    # NEED TO WORK ON THE HYPERPARAMETER OPTIMIZATIONS!!!
    standard_param = {'user_column': user_id, 'item_column': product_id, 'target_column': values}
    hyper_param = {'n_factors':range(10,20), 'regularization':[0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
                   'nmf':[True, False], 'unobserved_rating_regularization': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]}
    aJob = gl.toolkits.model_parameter_search(environment = env, model_factory = gl.recommender.create, 
    train_set = '/data/graphlab_train', save_path ='/data/hyperparameter_search_result', 
    test_set = '/data/graphlab_test', standard_model_params = standard_param, hyper_params = hyper_param)
    results = gl.load_sframe('hyperparameter_search_result')
    """

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
        aRMSE = gl.evaluation.rmse(test['ratings'], model.score(test))
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

    return results.sort('score', ascending=False)
    
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

""" ######################### TEXT ANALYTICS (SENTIMENT ANALYSIS) ######################### """ 

def return_tfidf(text_data):
    """ 
        runs the sklearn tf-idf vecotrizer method to compute the table of the word vs. frequency 
        the data should be such that it has text attributes plus the associated 
    """
    aTFIDF_model = ml_feature_extract.text.TfidfVectorizer()
    text_data = text_data.apply(lambda x: x.lower())
    text_data = text_data.apply(lambda x: preproc.clean(x))
    aTFIDF_model.fit(text_data)
    text_data_tfidf = aTFIDF_model.transform(text_data)
    words = aTFIDF_model.get_feature_names()

    return text_data_tfidf, words


def sentiment_analysis(aData, needed_param):
    
    text_data = aData[needed_param['comment']]
    ratings = aData[needed_param['ratings']]  
    
    # run the tf_idf functions
    text_data, words = return_tfidf(text_data)
    # you may want to add some parsing methods here
    

    # cross-validation
    text_train, text_test, ratings_train, ratings_test = cv.train_test_split(text_data, ratings, test_size = 0.3)
    
    
    
    # Naive Bayes
    param_NB = {'alpha': [0.1, 0.11, 0.15, 0.17, 0.2], 'fit_prior': [True, False]}
    nb_Model = bayes.MultinomialNB()
    grid_NB = gs.GridSearchCV(nb_Model, param_NB)
    grid_NB.fit(text_train.toarray(), ratings_train)
    best_NB_model = grid_NB.best_estimator_
    
    
    # Logistic Regression (can run maximum entropy classifier if multiclass issue)
    param_LogReg = {'penalty':['l1', 'l2'], 'C': [0.0001, 0.001, 0.01, 0.1, 1],
                    'fit_intercept':[True, False]}
    logReg_Model = lm.LogisticRegression()
    grid_LogReg = gs.GridSearchCV(logReg_Model, param_LogReg)
    grid_LogReg.fit(text_train.toarray(), ratings_train)
    best_LogReg_model = grid_LogReg.best_estimator_
    
    # Decision Trees
    param_Tree = {'criterion':['gini', 'entropy'], 'splitter':['best', 'random'], 
    'max_features':['auto', 'sqrt', 'log2']}
    tree_Model = tree.DecisionTreeClassifier()
    grid_Tree = gs.GridSearchCV(tree_Model, param_Tree)
    grid_Tree.fit(text_train.toarray(), ratings_train)
    best_Tree_model = grid_Tree.best_estimator_
    
    '''
    # Support Vector Machines
    # extremely slow --> you have to implement this in other framework
    # adding degrees will extremely slow down 
    param_SVC = {'C': [1], 'kernel':['poly', 'rbf']}
    svc_Model = svm.SVC()
    grid_SVC = gs.RamdomizedSearchCV(svc_Model, param_SVC, n_jobs = 2)
    grid_SVC.fit(text_train.toarray(), ratings_train)
    best_SVC_model = grid_SVC.best_estimator_
    '''
    
    """
    # SGDClassifier, No!, since this is a linear classifier
    # took out the hinge part since they are used for SVM
    # don't use this!
    param_SGD = {'loss':['log', 'modified_huber'],
                 'penalty':['l1', 'l2', 'elasticnet'], 'learning_rate':['constant', 'optimal'], 'eta0':[ 0.1, 1.0]}
    sgd_Model = lm.SGDClassifier()
    grid_sgd = gs.GridSearchCV(sgd_Model, param_SGD)
    grid_sgd.fit(text_train.toarray(), ratings_train)
    best_SGD_model = grid_sgd.best_estimator_
    """
    
    # Perceptron
    # RidgeCV
    # latent Semantic Analysis
    
    # estimate the accuracy using the test set
    nb_scores = grid_NB.score(text_test.toarray(), ratings_test)
    logReg_scores = grid_LogReg.score(text_test.toarray(), ratings_test)
    tree_scores = grid_Tree.score(text_test.toarray(), ratings_test)
    # svc_scores = grid_SVC.score(pd.DataFrame(text_test.toarray()), ratings_test)
    # sgd_scores = grid_sgd.score(text_test.toarray(), ratings_test)    
    
    # find the best model
    models = [ best_NB_model, best_Tree_model]
    scores = [ nb_scores, tree_scores]
    best_estimator_AdaBoost = models[scores.index(max(scores))]
    
    # AdaBoost (base_estimator = Tree algorithm) 
    param_adaBoost = {'algorithm':['SAMME', 'SAMME.R']}
    adaBoost_model = ensemble.AdaBoostClassifier(best_estimator_AdaBoost)
    grid_AdaBoost = gs.GridSearchCV(adaBoost_model, param_adaBoost)
    grid_AdaBoost.fit(text_train.toarray(), ratings_train)
    best_AdaBoost_model = grid_AdaBoost.best_estimator_
    adaBoost_scores = grid_AdaBoost.score(text_test.toarray(), ratings_test)
    
    # IMPORTANT : WRITE THIS IN OBJECT-ORIENTED or DYNAMIC PROGRAMMING WAY
    
    # Random Forest
    param_RandForest = {'n_estimators':[10, 20, 30], 'criterion':['gini', 'entropy'],
                        'max_features':['auto', 'sqrt', 'log2']}
    randForest_Model = ensemble.RandomForestClassifier()
    grid_randForest = gs.GridSearchCV(randForest_Model, param_RandForest)
    grid_randForest.fit(text_train.toarray(), ratings_train)
    best_randForest_model = grid_randForest.best_estimator_
    randForest_scores = grid_randForest.score(text_test.toarray(), ratings_test)
    
    bestModel = None
    if(adaBoost_scores > randForest_scores):
        bestModel = best_AdaBoost_model
        print "AdaBoost is the best model!"
    else:
        bestModel = best_randForest_model
        print "Random Forest is the best model!"
    predicted_ratings = bestModel.predict(text_data.toarray())
    
    print "SCORES!!!"
    print (float(sum(ratings==predicted_ratings))/float(len(predicted_ratings)))
    
    user_id = needed_param['user_id']
    product_id = needed_param['product_id']
    rec_data = aData[[user_id, product_id]]
    rec_data['ratings']=pd.Series(predicted_ratings)    
    rec_data.rename(columns = {user_id: 'user_id', product_id: 'product_id'}, inplace = True)
    
    return bestModel, rec_data