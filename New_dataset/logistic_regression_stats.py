import os
import time
import datetime
from contextlib import contextmanager
import random
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

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
    print('Duration of [{}]: {}'.format(label, datetime.timedelta(seconds=end-start)))


def load_data(path, to_split=True, delimiter=',',selected_features = None):
    """
    Load the csv file and returns (X,y).
    """
    df = pd.read_csv(path, delimiter=delimiter)
    y = df['PlayerA Win'].values.squeeze()

    df = df.drop(columns=['PlayerA Win'])

    # If no features selected by the user
    if not selected_features:
        # Take all numerical features
        df = df.iloc[:,8:]
        selected_features = df.columns

    df = df[selected_features]
    X = df.values.squeeze()
    
    if to_split:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        return X_train, X_test, y_train, y_test, selected_features
    else:
        return X, y, selected_features



def train(path, to_split=True):
    """
    Train th model.
    """
    logit_model = None
    filename = "Logistic_regression_stats.pkl"
    
    # Load the training (and testing) set(s)
    if to_split:
        X_train, X_test, y_train, y_test, _ = load_data(path, to_split=to_split)
    else:
        X, y, _ = load_data(path, to_split=to_split)

    # Get the most important features

    with measure_time('Training...'):
        logit_model = LogisticRegression(random_state=42)

        logit_model.fit(X_train, y_train)
        joblib.dump(logit_model, filename) 
        
    y_pred = logit_model.predict(X_train)
    print("=================================================================")
    print("Logistic Regression Training set accuracy: {}".format(accuracy_score(y_train, y_pred)))
    print("=================================================================")
    
    y_pred = logit_model.predict(X_test)
    print("Logistic Regression Test set accuracy: {}".format(accuracy_score(y_test, y_pred)))
    print("=================================================================")
    
if __name__ == "__main__":
    path = 'new_stats_data.csv'
    train(path, to_split=True)
