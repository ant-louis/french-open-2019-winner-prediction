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

def create_estimator(path, nb_features, to_split=True):
    """
    Train th model.
    """
    model = None
    filename = "MLP_stats.pkl"
    features_df = pd.read_csv('feature_importance.csv', sep=',')
    features_list = features_df.iloc[:nb_features, 0].tolist()

    # Load the training (and testing) set(s)
    if to_split:
        X, X_test, y, y_test, _ = load_data(path, to_split=to_split, selected_features=features_list)
    else:
        X, y, _ = load_data(path, to_split=to_split, selected_features=features_list)

    # Get the most important features
    with measure_time('Training...'):
        model = MLPClassifier(solver='adam', 
                                hidden_layer_sizes = (100,), 
                                early_stopping=True,
                                learning_rate_init= 0.01,
                                learning_rate = 'adaptive',
                                activation='tanh')
        model.fit(X, y)
        joblib.dump(model, filename) 
        
    y_pred = model.predict(X)
    print("=================================================================")
    print("Training set accuracy: {}".format(accuracy_score(y, y_pred)))
    print("=================================================================")
    
    if to_split:
        y_pred = model.predict(X_test)
        print("Test set accuracy: {}".format(accuracy_score(y_test, y_pred)))
        print("=================================================================")


def tune_hyperparameter(path):
    """
    Get the best hyperparameters.
    """
    # Load the training set
    X, y, _ = load_data(path, to_split=False)
        
    # Create the random grid
    random_grid = {'activation': ['tanh', 'relu'],
                    'learning_rate_init': [0.0005, 0.01, 0.03],
                    'learning_rate': ['constant','adaptive']}
    
    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    mlp = MLPClassifier(solver='adam', hidden_layer_sizes = (100,), early_stopping=True)
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    mlp_random = RandomizedSearchCV(estimator = mlp,
                                   param_distributions = random_grid,
                                   n_iter = 100,
                                   cv = 5,
                                   verbose=2,
                                   random_state=42,
                                   n_jobs = -2)
    # Fit the random search model
    mlp_random.fit(X, y)

    print("Best parameters", mlp_random.best_params_)
    # Best parameters {'learning_rate_init': 0.01, 'learning_rate': 'adaptive', 'activation': 'tanh'}
if __name__ == "__main__":
    path = 'new_stats_data_diff.csv'

    # tune_hyperparameter(path)
    create_estimator(path,6, to_split=False)
