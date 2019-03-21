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
    all_df = pd.read_csv(path, delimiter=delimiter)
    # Sorting because we don't want to predict the past matches with data about the future
    all_df.sort_values(by=['Year', 'Day'], inplace=True)
    y = all_df['PlayerA Win'].values.squeeze()
    all_df.drop('PlayerA Win', axis=1, inplace=True)

    # If no features selected by the user
    if selected_features is None:
        # Take all numerical features
        selected_features = all_df.iloc[:,8:].columns.tolist()

    X = all_df[selected_features].values.squeeze()

    print("Selected features :", selected_features)
    # Shuffle is False because we don't want to predict the past matches with data about the future
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, shuffle=False)

    if to_split:
        return X_train, X_test, y_train, y_test, np.asarray(selected_features)
    else:
        return X, y, np.asarray(selected_features)



def train(path, nb_features, to_split=True):
    """
    Train the model.
    """
    logit_model = None
    features_df = pd.read_csv('feature_importance.csv', sep=',')
    features_list = features_df.iloc[:nb_features, 0].tolist()

    filename = "Logistic_regression_stats.pkl"
    
    # Load the training (and testing) set(s)
    if to_split:
        X_train, X_test, y_train, y_test, _ = load_data(path, to_split=to_split, selected_features=features_list)
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
    
    if to_split:
        y_pred = logit_model.predict(X_test)
        print("Logistic Regression Test set accuracy: {}".format(accuracy_score(y_test, y_pred)))
        print("=================================================================")
    
if __name__ == "__main__":
    path = 'new_stats_data_diff.csv'
    train(path, nb_features=5, to_split=True)

