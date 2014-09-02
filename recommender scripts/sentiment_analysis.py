# -*- coding: utf-8 -*-
"""
sentiment_analysis.py
the module for the sentiment analysis

1. supervised sentiment analysis
    - create the TF-IDF table with bi and tri unigrams
    - run all the possible machine learning models with the grid search
    - create ensemble methods in order to boost the accuracy over each model
    - provide the result of the model with the best score tested on the testing set

2. unsupervised sentiment analysis
    - currently using APIs from other free sources
    - still in development version

"""
import pandas as pd
import sklearn.feature_extraction as ml_feature_extract
import sklearn.linear_model as lm
import sklearn.tree as tree
import sklearn.svm as svm
import sklearn.naive_bayes as bayes
import sklearn.grid_search as gs
import sklearn.cross_validation as cv
import sklearn.ensemble as ensemble
import textblob as tb
import preprocessing as preproc

def return_tfidf(text_data):
    """ runs the sklearn tf-idf vecotrizer method to compute the table of the word vs. frequency 
        the data should be such that it has text attributes plus the associated """
        
    aTFIDF_model = ml_feature_extract.text.TfidfVectorizer(analyzer = 'word', ngram_range = (2,3))
    text_data = text_data.apply(lambda x: x.lower())
    text_data = text_data.apply(lambda x: preproc.clean(x))
    aTFIDF_model.fit(text_data)
    text_data_tfidf = aTFIDF_model.transform(text_data)
    words = aTFIDF_model.get_feature_names()
    print aTFIDF_model.get_feature_names()

    return text_data_tfidf, words
    
def generateModelNScores(model, param, data_train, data_test, target_train, target_test):
    """ initializes the models using the GridSearchCV part and returns the best model 
        as well as the scores of the model using the test dataset """
        
    gridModel = gs.GridSearchCV(model, param)
    gridModel.fit(data_train, target_train)
    best_model = gridModel.best_estimator_
    scores = gridModel.score(data_test, target_test)
    
    return best_model, scores

def sentiment_analysis_supervised(aData, needed_param):
    """ supervised sentiment analysis combines every possible classification method 
        to predict ratings. The method compares the scores achived by the models testing 
        on the test set and selects the best one. The classic classification methods such as
        naive Bayes and logistic regressions are compared first, and the one with the best score
        is put into the AdaBoost model in order to boost the accuracy """
        
    text_data = aData[needed_param['comment']]
    ratings = aData[needed_param['ratings']]
    
    # run the tf_idf functions
    text_data, words = return_tfidf(text_data)
    # you may want to add some parsing methods here
    
    # cross-validation
    text_train, text_test, ratings_train, ratings_test = cv.train_test_split(text_data, ratings, test_size = 0.3)
    
    # create emptly list of models and scores
    models = []
    scores = []    
    
    # Naive Bayes
    param_NB = {'alpha': [0.1, 0.11, 0.15, 0.17, 0.2], 'fit_prior': [True, False]}
    nb_Model = bayes.MultinomialNB()
    best_NB, scores_NB = generateModelNScores(nb_Model, param_NB, 
                                              text_train.toarray(), text_test.toarray(),
                                            ratings_train, ratings_test)
    models.append(best_NB)
    scores.append(scores_NB)
    print "finished NB"
    
    # Decision Trees
    param_Tree = {'criterion':['gini', 'entropy'], 'splitter':['best', 'random'],
                  'max_features':['auto', 'sqrt', 'log2']}
    tree_Model = tree.DecisionTreeClassifier()
    best_tree, scores_tree = generateModelNScores(tree_Model, param_Tree, 
                                              text_train.toarray(), text_test.toarray(),
                                            ratings_train, ratings_test)
    models.append(best_tree)
    scores.append(scores_tree)    
    print "finished tree"
    
    # Support Vector Machines
    # extremely slow --> you have to implement this in other framework
    # adding degrees will extremely slow down 
    param_SVC = {'kernel':['rbf']}
    svc_Model = svm.SVC()
    best_SVC, scores_SVC = generateModelNScores(svc_Model, param_SVC, 
                                    text_train.toarray(), text_test.toarray(),
                                    ratings_train, ratings_test)
    models.append(best_SVC)
    scores.append(scores_SVC)    
    print "finished svm"
    
    # find the best model among the AdaBoost-compatible models
    best_estimator_AdaBoost = models[scores.index(max(scores))]
    print best_estimator_AdaBoost
    
    # AdaBoost (base_estimator = Tree algorithm) 
    param_adaBoost = {'algorithm':['SAMME', 'SAMME.R']}
    adaBoost_model = ensemble.AdaBoostClassifier(best_estimator_AdaBoost)
    best_AdaBoost, scores_AdaBoost = generateModelNScores(adaBoost_model, param_adaBoost, 
                                              text_train.toarray(), text_test.toarray(),
                                            ratings_train, ratings_test)
    models.append(best_AdaBoost)
    scores.append(scores_AdaBoost)
    print "finished boosting"
    
    # Logistic Regression (can run maximum entropy classifier if multiclass issue)
    param_LogReg = {'penalty':['l1', 'l2'], 'C': [0.0001, 0.001, 0.01, 0.1, 1],
                    'fit_intercept':[True, False]}
    logReg_Model = lm.LogisticRegression()
    best_logReg, scores_logReg = generateModelNScores(logReg_Model, param_LogReg, 
                                              text_train.toarray(), text_test.toarray(),
                                            ratings_train, ratings_test)
    models.append(best_logReg)
    scores.append(scores_logReg)
    print "finished logistic"

    # Random Forest
    param_RandForest = {'n_estimators':[10, 20, 30], 'criterion':['gini', 'entropy'],
                        'max_features':['auto', 'sqrt', 'log2']}
    randForest_Model = ensemble.RandomForestClassifier()
    best_randForest, scores_randForest = generateModelNScores(randForest_Model, param_RandForest, 
                                                              text_train.toarray(), text_test.toarray(),
                                                              ratings_train, ratings_test)
    models.append(best_randForest)
    scores.append(scores_randForest)
    print "finished random forest"
    
    bestModel =  models[scores.index(max(scores))]
    predicted_ratings = bestModel.predict(text_data.toarray())
    print "Models!"
    print bestModel
    
    print "SCORES!!!"
    print (float(sum(ratings==predicted_ratings))/float(len(predicted_ratings)))
    
    user_id = needed_param['user_id']
    product_id = needed_param['product_id']
    rec_data = aData[[user_id, product_id]]
    rec_data['ratings']=pd.Series(predicted_ratings)    
    rec_data.rename(columns = {user_id: 'user_id', product_id: 'product_id'}, inplace = True)
    
    return bestModel, rec_data

def give_scores(aText):
    """ uses the textblob object in order to quantify the polarity of the reviews """
    polarity = tb.TextBlob(aText).sentiment.polarity

    return polarity
    
def sentiment_analysis_unsupervised(aDataset, needed_param):
    """ uses the unsupervised sentiment analysis using the textblob source code """
    text_data = aDataset[needed_param['comment']]
    predicted_scores = text_data.apply(lambda x: give_scores(x))
    aData = aDataset[[needed_param['user_id'], needed_param['item_id']]]
    aData[needed_param['ratings']] = predicted_scores
    
    return aData