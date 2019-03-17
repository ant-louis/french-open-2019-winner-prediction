import os
import time
import datetime
from contextlib import contextmanager
import random
import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegressionCV

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve, auc

from sklearn.externals import joblib

from matplotlib.legend_handler import HandlerLine2D
from matplotlib import pyplot as plt
@contextmanager
def measure_time(label):
    """
    Context manager to measure time of computation.
    """
    start = time.time()
    yield
    end = time.time()
    print('Duration of [{}]: {}'.format(label, datetime.timedelta(seconds=end-start)))

def load_from_csv(path, delimiter=','):
    """
    Load csv file and return a NumPy array of its data
    """
    return pd.read_csv(path, delimiter=delimiter,dtype=float)

def create_estimator():
    """
    """
    # Loading data
    prefix = '../Data_cleaning'
    df = load_from_csv(os.path.join(prefix, 'training_matches_players_diff.csv'))

    y = df['PlayerA Win'].values.squeeze()
    toDrop = ['PlayerA Win', 'ID_PlayerA', 'ID_PlayerB'] #ID's skew results
    train_features = df.drop(columns=toDrop).columns
    X = df.drop(columns=toDrop).values.squeeze() 
    X = preprocessing.scale(X) 

    model = None
    filename = "Logistic_regression.pkl"

    #Training
    print("Training... getting most important features")
    
    with measure_time('Training...getting most important features'):
        model = LogisticRegressionCV(cv=10, multi_class='auto', max_iter=10000)
        #selector = RFE(model, n_features_to_select = 1)
        model.fit(X,y)
        joblib.dump(model, filename) 


    #feature_importances = pd.DataFrame(model.feature_importances_,
    #                                      index = train_features,
    #                                      columns=['importance']).sort_values('importance',ascending=False)

    #print("Most important features")
    #print(model.ranking_)
    # feature_importances[:100].to_csv("feature_importance.csv")
  
    print("Test set accuracy: {}".format(model.score(X,y)))
    print("=================================================================")

def tune_hyperparameter():
    """
    """
    # Loading data
    prefix = '../Data_cleaning'
    df = load_from_csv(os.path.join(prefix, 'training_matches_players_diff.csv'))

    y = df['PlayerA Win'].values.squeeze()
    toDrop = ['PlayerA Win','ID_PlayerA', 'ID_PlayerB'] #ID's skew results
    train_features = df.drop(columns=toDrop).columns
    X = df.drop(columns=toDrop).values.squeeze()
    X = preprocessing.scale(X)

    #Hyperparameter tuning
    number_feature = [int(x) for x in np.linspace(1, 50, num = 20)]

    # Create the random grid
    random_grid = {'n_features_to_select': number_feature}

    model = LogisticRegressionCV(cv=10, multi_class='auto', max_iter=10000)
    selector = RFE(model)
    rf_random = RandomizedSearchCV(estimator = selector, 
                                    param_distributions = random_grid, 
                                    n_iter = 10, 
                                    cv = 3, 
                                    verbose=2, 
                                    random_state=42, 
                                    n_jobs = -1)
 
    rf_random.fit(X, y)

    print("Best parameters", rf_random.best_params_)

if __name__ == "__main__":

    create_estimator()
    #tune_hyperparameter()
