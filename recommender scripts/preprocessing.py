
"""
preprocessing.py
The preprocessing module for the recommender system.
"""
import re
import requests
import pandas as pd
from collections import defaultdict
import sklearn.preprocessing as preproc
import graphlab as gl
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL


def getData_API(api, needed_param, colnames = None):
    """takes the api parameter and retruens the data frame that contains the review dataset"""
    r = requests.get(api)
    jsonfile = r.json()
    needed_data = jsonfile[needed_param]
    needed_data  = pd.DataFrame(needed_data)
    if(colnames != None):
        needed_data.columns = colnames
    return needed_data

def getMultipleData_Static(filepath, ls_filenames, separator, colnames = None, nRowsToSkip=0, nRowsToRead = 40000):
     ''' get multiple data from the filepath, append to the filepath, and create the merged version'''
     def makeData(filepath, static_file, separator):
        datapath = filepath+static_file
        aData = pd.read_csv(filepath_or_buffer = datapath, sep = separator, nrows = nRowsToRead, skiprows = nRowsToSkip) 
        return aData

     data = pd.concat([makeData(filepath, static_file, separator) for static_file in ls_filenames])
     if(colnames != None):
        data.columns = colnames    
     return data

def get_db_engine(dialect_driver = 'postgresql+psycopg2',  dbname='database'):
    """ 
        depending on the type of the database, will create engine as well as necessary operations
        check whether the database already exists
    """
    url = URL(drivername=dialect_driver, username='chlee021690',  database = dbname)
    engine = create_engine(url)

    return engine

def write_sql(engine, ls_data, ls_tablename, exists_command = 'append'):
    """ write/update multiple tables at a time. the length of the ls_data and ls_tablename must be the same """
    if(len(ls_data)==len(ls_tablename)):
        for i in range(0, len(ls_data)):
            ls_data[i].to_sql(name = ls_tablename[i], con = engine, if_exists = exists_command, index = False)
            # maybe you need something to prevent the duplicates of data from being written
    else:
        return None

def removeDuplicates(aSeries):
    ''' takes a Series object and returns the list with the duplicates being removed '''
    data = defaultdict(list)
    for aData in list(aSeries):
        data[aData].append(True)
    
    return data.keys()

def make_CF_table(aData, index_name, col_name):
    ''' make an appropriate table for the collaborative filtering and insert the ratings value in the table '''

    # make the empty collaborative filtering table
    products = removeDuplicates(aData[index_name])
    usernames = removeDuplicates(aData[col_name])
    table_CF =  pd.DataFrame(index = products, columns = usernames)

    # fill in the values
    for product in table_CF.index:
        for user in table_CF.columns:
            if (len(aData[(aData['reviewer']==user) & (aData['sku']==product)].rating)>0):
                table_CF.loc[product, user] = aData[(aData['reviewer']==user) & (aData['sku']==product)].rating.values[0]
            else:
                continue
                
    return table_CF

def mean_imputation(table_CF):
    aCopy = table_CF.copy()
    imp = preproc.Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(aCopy)
    aCopy = pd.DataFrame(imp.transform(aCopy), columns = aCopy.columns)
    return aCopy

def graphlab_processTable(aData, needed_param):
    """ returns the unpirvoted version of the data for the GraphLab from the DataFrame"""
    user_id = needed_param['user_id']
    product_id = needed_param['product_id']
    ratings = needed_param['ratings']
    
    needed_data = aData[[user_id, product_id,ratings]]
    needed_data = needed_data.sort([user_id], ascending = True)
    
    aData = gl.SFrame(needed_data)
    return aData
    
def graphlab_split_data(aData, frac):
    """splits the data using the random_split method from the graphLab"""
    aData = gl.SFrame(aData)
    train, test = aData.random_split(fraction = frac)
    return train, test

def inspect(afilter='classifier', parameter='sample_weight'):
    """ helps you inspect some of the parameters and some options you may want  to choose"""
    import inspect
    from sklearn.utils.testing import all_estimators
    for name, clf in all_estimators(type_filter=afilter):
        if parameter in inspect.getargspec(clf().fit)[0]:
            print name

def clean(s):
    '''clean some of the text that is unrecognized by the python console '''
    try:
        return " ".join(re.findall(r'\w+', s, flags = re.UNICODE | re.LOCALE)).lower()
    except:
        return " ".join(re.findall(r'\w+', "no_text", flags = re.UNICODE | re.LOCALE)).lower()

def change_type(aData, typeChangeTo=int):
    """ 
        change the type of the data into appropariate one
        when retrieving data from sql, some integer data is found out to be str
        to retain the type, it is necessary to have this kind of function
    
    """
    aData = aData.apply(lambda x: typeChangeTo(x))
    return aData