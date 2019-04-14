import os
import time
import datetime
from contextlib import contextmanager
import random
import pandas as pd
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
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



def create_estimator(path, to_split=True, selected_features=None):
    """
    Train the model.
    """
    nb_features = len(selected_features)
    filename = "_Models/MLP_top{}_features.pkl".format(nb_features)
    
    # Load the training (and testing) set(s)
    if to_split:
        X_train, X_test, y_train, y_test, _ = load_data(path, to_split=to_split, selected_features=selected_features)
    else:
        X_train, y_train, _ = load_data(path, to_split=to_split, selected_features=selected_features)

    with measure_time('Training...'):
        model = MLPClassifier(solver='sgd', 
                                hidden_layer_sizes = (20,), 
                                early_stopping=True,
                                learning_rate_init= 0.05,
                                learning_rate = 'constant',
                                activation='tanh',
                                momentum=0.6)
        model.fit(X_train, y_train)
        joblib.dump(model, filename) 
        
    y_pred = model.predict(X_train)
    print("=================================================================")
    print("Training set accuracy: {}".format(accuracy_score(y_train, y_pred)))
    print("=================================================================")
    
    if to_split:
        y_pred = model.predict(X_test)
        print("Test set accuracy: {}".format(accuracy_score(y_test, y_pred)))
        print("=================================================================")


def tune_hyperparameter(path, selected_features=None):
    """
    Get the best hyperparameters.
    """
   # Load the training set
    X, y, _ = load_data(path, to_split=False, selected_features=selected_features)
        
    # Create the random grid
    random_grid = {'hidden_layer_sizes': [(20,), (50,), (100,), (150,)],
                    'activation': ['tanh', 'relu', 'logistic', 'identity'],
                    'learning_rate_init': [0.005, 0.01, 0.05, 0.1, 0.2, 0.3],
                    'learning_rate': ['constant','adaptive'],
                    'momentum': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
    
    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    mlp = MLPClassifier(solver='sgd', early_stopping=True)
    # Random search of parameters, using 5 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    mlp_random = RandomizedSearchCV(estimator = mlp,
                                   param_distributions = random_grid,
                                   n_iter = 100,
                                   cv = 5,
                                   verbose=2,
                                   random_state=42,
                                   n_jobs = -1)
    # Fit the random search model
    mlp_random.fit(X, y)

    print("Best parameters", mlp_random.best_params_)
    # Best parameters {'solver': 'sgd', 'learning_rate_init': 0.05, 'learning_rate': 'constant', 'hidden_layer_sizes': (20,), 'activation': 'tanh', 'momentum': 0.6}

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

    #tune_hyperparameter(path, selected_features=selected_features)
    create_estimator(path, to_split=True, selected_features=selected_features)
