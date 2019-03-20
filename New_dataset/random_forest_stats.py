import os
import time
import datetime
from contextlib import contextmanager
import random
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

import warnings


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

    df.drop(columns=['PlayerA Win'], inplace=True)
    df.drop(columns=['bestof_diff'], inplace=True)# Overfitting feature


    # If no features selected by the user
    if not selected_features:
        # Take all numerical features
        df = df.iloc[:,8:]
        selected_features = df.columns

    X = df[selected_features].values.squeeze()
    
    if to_split:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        return X_train, X_test, y_train, y_test, np.asarray(selected_features)
    else:
        return X, y, np.asarray(selected_features)


def train_estimator(path, computeFeatureImportance=False, nb_features=48, to_split=True):
    """
    Train the model.
    """
    model = None
    features_list = None
    # For testing purposes
    # Select the best features (ones with the most importance)
    # If we want to compute the feature importance, we obviously use all of them
    if not computeFeatureImportance:
        features_df = pd.read_csv('feature_importance.csv', sep=',')
        features_list = features_df.iloc[:nb_features, 0].tolist()

    # Load the training (and testing) set(s)
    if to_split:
        X, X_test, y, y_test, train_features = load_data(path, to_split=to_split, selected_features=features_list)
    else:
        X, y, train_features = load_data(path, to_split=to_split, selected_features=features_list)

    # Get the most important features
    if computeFeatureImportance:
        print("Computing feature importance!")
        ntry = 10
    else:
        print("Training the model!")
        ntry = 1
    feature_importance_total = np.zeros((train_features.shape[0],1))
    for n in range(ntry):
        with measure_time('Training {} out of {}'.format(n+1, ntry)):
            model = RandomForestClassifier(n_estimators=2000,
                                            min_samples_split=7,
                                            min_samples_leaf=2,
                                            max_features='auto',
                                            max_depth=10, 
                                            bootstrap=False,
                                            random_state=42,
                                            verbose=1,
                                            n_jobs=-1)
            model.fit(X, y)

        feature_importance_total = np.add(feature_importance_total, np.array(model.feature_importances_).reshape(train_features.shape[0],1))

    # Only compute the features importance if we actually want to compute it
    if computeFeatureImportance:
        feature_importances = pd.DataFrame(feature_importance_total,
                                            index = train_features,
                                            columns=['importance'])

        print("Most important features")
        feature_importances['importance'] = feature_importances['importance'] / ntry
        feature_importances.sort_values('importance',ascending=False, inplace=True)
        feature_importances.to_csv("feature_importance.csv")

    print("Predicting!")
    y_pred = model.predict(X)
    print("=================================================================")
    print("Training set accuracy: {}".format(accuracy_score(y, y_pred)))
    print("=================================================================")
    
    # If we do a train test split, predict on the test set
    if to_split:
        y_pred = model.predict(X_test)
        print("Test set accuracy: {}".format(accuracy_score(y_test, y_pred)))
        print("=================================================================")


def create_estimator(path, nb_features):
    """Output a model .pkl file trained on the whole dataset
    and on a select few features"""

    model = None
    filename = "RandomForest_stats_{}feat.pkl".format(nb_features)
    features_df = pd.read_csv('feature_importance.csv', sep=',')

    # Select the best features (ones with the most importance)
    features_list = features_df.iloc[:nb_features, 0].tolist()

    X, y, _ = load_data(path, to_split=False, selected_features=features_list)

    print("Creating estimator on dataset with {} best features".format(nb_features))
    model = RandomForestClassifier(n_estimators=5000,
                                min_samples_split=7,
                                min_samples_leaf=2,
                                max_features='auto',
                                max_depth=10, 
                                bootstrap=False,
                                random_state=42,
                                verbose=1,
                                n_jobs=-1)
    model.fit(X, y)
    joblib.dump(model, filename)
    print("Saving model as ", filename)

def tune_hyperparameter(path):
    """
    Get the best hyperparameters.
    """
    # Load the training set
    X, y, _ = load_data(path, to_split=False)
    
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    
    # Create the random grid
    random_grid = {'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    
    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestClassifier()
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = rf,
                                   param_distributions = random_grid,
                                   n_iter = 100,
                                   cv = 3,
                                   verbose=2,
                                   random_state=42,
                                   n_jobs = -1)
    # Fit the random search model
    rf_random.fit(X, y)

    print("Best parameters", rf_random.best_params_)
    # Best parameters {'n_estimators': 1400, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'auto', 'max_depth': 40, 'bootstrap': False}

def plot_feature_importance(nb_features, filename):
    """Plot the desired number of features along with their accuracy"""

    data = pd.read_csv('feature_importance.csv', sep=',')

    feature = data.iloc[:nb_features, 0].tolist()
    importance = data.iloc[:nb_features, 1].tolist()

    # Choose the position of each barplots on the x-axis (space=1,4,3,1)
    x_pos = np.arange(0, nb_features, 1)
    
    # Create bars
    plt.figure() 
    plt.gcf().subplots_adjust(bottom=0.30)
    plt.bar(x_pos, importance)
    
    # Create names on the x-axis
    plt.xticks(x_pos, feature, rotation='vertical')

    plt.ylabel('Importance', fontsize=13)
    plt.xlabel('Feature', fontsize=13)

    plt.savefig(filename,format="svg")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)

    path = 'new_stats_data_diff.csv'
    #tune_hyperparameter(path)
    # train_estimator(path, computeFeatureImportance=False, nb_features=14, to_split=True)
    create_estimator(path, 14)
    # create_estimator(path, 6)
    # plot_feature_importance(47, 'all_features_importance.svg')
    # plot_feature_importance(14, '14_features_importance.svg')
    # plot_feature_importance(5, '5_features_importance.svg')
