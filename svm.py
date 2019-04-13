import os
import time
import datetime
from contextlib import contextmanager
import random
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
from sklearn.externals import joblib


@contextmanager
def measure_time(label):
    """
    Context manager to measure time of computation.
    """
    start = time.time()
    yield
    end = time.time()
    print('Duration of [{}]: {}'.format(label, datetime.timedelta(seconds=end-start)))


def load_data(path, to_split=True, selected_features=None):
    """
    Load the csv file and returns (X,y).
    """
    # Read the csv file
    df = pd.read_csv(path, header=0, index_col=0)

    # Sorting because we want our testing set to be the last matches
    df.sort_values(by=['Year', 'Day'], inplace=True)

    # Get the output values
    y = df['PlayerA_Win'].values.squeeze()
    df.drop('PlayerA_Win', axis=1, inplace=True)

    # If no features selected by the user, take all numerical features
    if selected_features is None:
        selected_features = ['Same_handedness',
                            'age_diff',
                            'rank_diff',
                            'rank_points_diff',
                            'Win%_diff',
                            'bestof_diff',
                            'minutes_diff',
                            'svpt%_diff',
                            '1st_serve%_diff',
                            '1st_serve_won%_diff',
                            '2nd_serve_won%_diff',
                            'ace%_diff',
                            'df%_diff',
                            'bp_faced%_diff',
                            'bp_saved%_diff']
    print("Selected features :", selected_features)

    # Get the input values
    X = df[selected_features].values.squeeze()
    
    # Shuffle is False because we don't want to predict the past matches with data about the future
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    if to_split:
        return X_train, X_test, y_train, y_test, np.asarray(selected_features)
    else:
        return X, y, np.asarray(selected_features)


def train(path, ker, to_split=True, selected_features=None):
    """
    Train the model.
    """
    nb_features = len(selected_features)
    filename = "_Models/SVM_{}_top{}_features.pkl".format(ker, nb_features)
    
    # Load the training (and testing) set(s)
    if to_split:
        X_train, X_test, y_train, y_test, _ = load_data(path, to_split=to_split, selected_features=selected_features)
    else:
        X_train, y_train, _ = load_data(path, to_split=to_split, selected_features=selected_features)

    with measure_time('Training...'):
        svm_model = SVC(kernel=ker, max_iter=100000)
        svm_model.fit(X_train, y_train)
        joblib.dump(svm_model, filename) 
        
    y_pred = svm_model.predict(X_train)
    print("=================================================================")
    print("SVM with {} Training set accuracy: {}".format(ker, accuracy_score(y_train, y_pred)))
    print("=================================================================")
    
    if to_split:
        y_pred = svm_model.predict(X_test)
        print("SVM with {} Test set accuracy: {}".format(ker, accuracy_score(y_test, y_pred)))
        print("=================================================================")


if __name__ == "__main__":
    path = "_Data/Training_dataset/training_data_weight06_+surface_weighting_min20matches.csv"

    selected_features = ['Same_handedness',
                         'age_diff',
                         'rank_diff',
                         'rank_points_diff',
                         'Win%_diff',
                         'bestof_diff',
                         'minutes_diff',
                         'svpt%_diff',
                         '1st_serve%_diff',
                         '1st_serve_won%_diff',
                         '2nd_serve_won%_diff',
                         'ace%_diff',
                         'df%_diff',
                         'bp_faced%_diff',
                         'bp_saved%_diff']
    
    # selected_features = ['age_diff',
    #                      'rank_diff',
    #                      'rank_points_diff',
    #                      'Win%_diff',
    #                      'bestof_diff',
    #                      '1st_serve_won%_diff',
    #                      '2nd_serve_won%_diff',
    #                      'bp_faced%_diff']
    
    # selected_features = ['Same_handedness',
    #                      'age_diff',
    #                      'Win%_diff',
    #                      'bestof_diff',
    #                      'minutes_diff',
    #                      'svpt%_diff',
    #                      '1st_serve%_diff',
    #                      '1st_serve_won%_diff',
    #                      '2nd_serve_won%_diff',
    #                      'ace%_diff',
    #                      'df%_diff',
    #                      'bp_faced%_diff',
    #                      'bp_saved%_diff']
    
    # Kernels : ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’
    train(path, 'rbf', to_split=True, selected_features=selected_features)