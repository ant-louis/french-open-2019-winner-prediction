import os
import time
import datetime
from contextlib import contextmanager
import random
import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

@contextmanager
def measure_time(label):
    """
    Context manager to measure time of computation.
    """
    start = time.time()
    yield
    end = time.time()
    print('Duration of [{}]: {}'.format(label,
                                        datetime.timedelta(seconds=end-start)))

def load_from_csv(path, delimiter=','):
    return pd.read_csv(path, delimiter=delimiter,dtype=float)


def create_estimator(testFeatureImportance):
    """
    """
    # Loading data
    df2018 = load_from_csv('../Data_cleaning/training_data/training_matches_players_diff_2018.csv')
    df2017 = load_from_csv('../Data_cleaning/training_data/training_matches_players_diff_2017.csv')
    df = pd.concat([df2018, df2017])

    y = df['PlayerA Win'].values.squeeze()
    toDrop = ['PlayerA Win', 'ID_PlayerA', 'ID_PlayerB'] #ID's skew results
    train_features = df.drop(columns=toDrop).columns
    X = df.drop(columns=toDrop).values.squeeze()  
    
    #Scaling
    X = preprocessing.scale(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = None
    filename = "RandomForest_diff_2018_2017.pkl"

    #Training
    print("Training... getting most important features")
    
    if testFeatureImportance:
        ntry = 10
    else:
        ntry = 1
    feature_importance_total = np.zeros((train_features.shape[0],1))
    for n in range(ntry):
        with measure_time('Training...getting most important features'):
            print("N = ",n)
            model = RandomForestClassifier(n_estimators=1000,
                                            min_samples_split=7,
                                            min_samples_leaf=2,
                                            max_features='auto',
                                            max_depth=10, 
                                            bootstrap=True,
                                            random_state=42,
                                            verbose=1,
                                            n_jobs=-1)
            model.fit(X_train, y_train)
            
            if not testFeatureImportance:
                joblib.dump(model, filename) 
        
        feature_importance_total = np.add(feature_importance_total, np.array(model.feature_importances_).reshape(train_features.shape[0],1))

    if testFeatureImportance:
        feature_importances = pd.DataFrame(feature_importance_total,
                                            index = train_features,
                                            columns=['importance'])

        print("Most important features")
        feature_importances['importance'] = feature_importances['importance'] / ntry
        print(feature_importances[:100].sort_values('importance',ascending=False))
        feature_importances[:100].to_csv("feature_importance.csv")
  
    y_pred = model.predict(X_test)
    print("Test set accuracy: {}".format(accuracy_score(y_test, y_pred)))
    print("=================================================================")

    y_pred = model.predict(X_train)
    print("Training set accuracy: {}".format(accuracy_score(y_train, y_pred)))
    print("=================================================================")


def tune_hyperparameter():
    """
    """
    # Loading data
    df2018 = load_from_csv('../Data_cleaning/training_data/training_matches_players_diff_2018.csv')
    df2017 = load_from_csv('../Data_cleaning/training_data/training_matches_players_diff_2017.csv')
    df = pd.concat([df2018, df2017])
    
    y = df['PlayerA Win'].values.squeeze()
    toDrop = ['PlayerA Win','ID_PlayerA', 'ID_PlayerB'] #ID's skew results
    X = df.drop(columns=toDrop).values.squeeze()  

    # Normalization
    X = preprocessing.scale(X)

    #Hyperparameter tuning
    max_features = ['auto', 'sqrt', 0.3]
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 7]
    min_samples_leaf = [1, 2, 4]

    # Create the random grid
    random_grid = {'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf}

    rf = RandomForestClassifier(n_estimators=2000)
    rf_random = RandomizedSearchCV(estimator = rf, 
                                    param_distributions = random_grid, 
                                    n_iter = 50, 
                                    cv = 3, 
                                    verbose=2, 
                                    random_state=42, 
                                    n_jobs = -2)
 
    rf_random.fit(X, y)

    print("Best parameters", rf_random.best_params_)


if __name__ == "__main__":

    create_estimator(False)
    # tune_hyperparameter()
