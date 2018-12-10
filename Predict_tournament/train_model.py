import os
import time
import datetime
from contextlib import contextmanager
import random
import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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
    prefix = '/home/tom/Documents/Master1_DataScience/1er QUADRI/Big-Data-Project/Data_cleaning/'
    df = load_from_csv(os.path.join(prefix, 'merged_matches_players.csv'))

    y = df['PlayerA Win'].values.squeeze()
    train_features = df.drop(columns=['PlayerA Win']).columns
    X = df.drop(columns=['PlayerA Win']).values.squeeze()  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training...getting most important features")

    with measure_time('Training...getting most important features'):
        model = RandomForestClassifier(n_estimators=1000,max_depth=3, bootstrap=True,n_jobs=-1, random_state=42)
        model.fit(X_train,y_train)
        joblib.dump(model, "estimators/RandomForest_depth3.pkl") 

    feature_importances = pd.DataFrame(model.feature_importances_,
                                        index = train_features,
                                        columns=['importance']).sort_values('importance',ascending=False)

    print("Most important features")
    print(feature_importances[:100])
    most_imp_features = feature_importances[:100].axes[0].tolist()
    
    y_pred = model.predict(X_test)
    print("Test set accuracy: {}".format(accuracy_score(y_test, y_pred)))
    print("=================================================================")

    y_pred = model.predict(X_train)
    print("Training set accuracy: {}".format(accuracy_score(y_train, y_pred)))
    print("=================================================================")



if __name__ == "__main__":
    create_estimator()