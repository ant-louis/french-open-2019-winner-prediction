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
    >>> with measure_time('Heavy computation'):
    >>>     do_heavy_computation()
    'Duration of [Heavy computation]: 0:04:07.765971'

    Parameters
    ----------
    label: str
        The label by which the computation will be referred
    """
    start = time.time()
    yield
    end = time.time()
    print('Duration of [{}]: {}'.format(label,
                                        datetime.timedelta(seconds=end-start)))

def load_from_csv(path, delimiter=','):
    """
    Load csv file and return a NumPy array of its data

    Parameters
    ----------
    path: str
        The path to the csv file to load
    delimiter: str (default: ',')
        The csv field delimiter

    Return
    ------
    D: array
        The NumPy array of the data contained in the file
    """
    return pd.read_csv(path, delimiter=delimiter,dtype=float)

def create_estimator():

    # Loading data
    prefix = '../Data_cleaning'
    df = load_from_csv(os.path.join(prefix, 'training_matches_players.csv'))

    y = df['PlayerA Win'].values.squeeze()
    toDrop = ['PlayerA Win', 'ID_PlayerA', 'ID_PlayerB'] #ID's skew results
    train_features = df.drop(columns=toDrop).columns
    X = df.drop(columns=toDrop).values.squeeze()  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = None
    filename = "Logistic_regression.pkl"

    #Training
    print("Training... getting most important features")
    
    with measure_time('Training...getting most important features'):
        model = LogisticRegressionCV(cv=10, multi_class='auto', max_iter=10000)
        selector = RFE(model, n_features_to_select = 1)
        selector.fit(X,y)
        joblib.dump(selector, filename) 

    feature_importances = pd.DataFrame(model.feature_importances_,
                                          index = train_features,
                                          columns=['importance']).sort_values('importance',ascending=False)

    print("Most important features")
    print(selector.ranking_)
    # feature_importances[:100].to_csv("feature_importance.csv")
  
    print("Test set accuracy: {}".format(selector.score(X,y)))
    print("=================================================================")

def tune_hyperparameter():
    # Loading data
    prefix = '../Data_cleaning'
    df = load_from_csv(os.path.join(prefix, 'training_matches_players.csv'))

    y = df['PlayerA Win'].values.squeeze()
    toDrop = ['PlayerA Win','ID_PlayerA', 'ID_PlayerB'] #ID's skew results
    train_features = df.drop(columns=toDrop).columns
    X = df.drop(columns=toDrop).values.squeeze()  

    #Hyperparameter tuning
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]

    number_feature = [int(x) for x in np.linspace(1, 50, num = 2)]

    # Create the random grid
    random_grid = {'n_features_to_select': number_feature}

    model = LogisticRegressionCV(cv=10, multi_class='auto', max_iter=10000)
    selector = RFE(model)
    rf_random = RandomizedSearchCV(estimator = selector, 
                                    param_distributions = random_grid, 
                                    n_iter = 100, 
                                    cv = 3, 
                                    verbose=2, 
                                    random_state=42, 
                                    n_jobs = -1)
 
    rf_random.fit(X, y)

    print("Best parameters", rf_random.best_params_)

if __name__ == "__main__":

    #create_estimator()
    tune_hyperparameter()

   
