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

    # Exclude the matches of Roland Garros 2016, 2017, 2018 (as we will test on that later)
    df_2018 = df[(df['Year'] == 2018) & (df['Day'] == 148)]
    df_2017 = df[(df['Year'] == 2017) & (df['Day'] == 149)]
    df_2016 = df[(df['Year'] == 2016) & (df['Day'] == 143)]
    bad_index = df_2018.index.values.tolist() + df_2017.index.values.tolist() + df_2016.index.values.tolist()
    df = df[~df.index.isin(bad_index)]
    df.reset_index(drop=True, inplace=True)

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


def train_estimator(path, computeFeatureImportance=False, to_split=True, selected_features=None):
    """
    Train the model.
    If computeFeatureImportance is True, to_split must be True too. We don't want
    to contaminate our test_set by doing a feature extraction on it and then
    testing on it. 
    """

    # Load the training (and testing) set(s)
    if to_split:
        X, X_test, y, y_test, train_features = load_data(path, to_split=to_split, selected_features=selected_features)
    else:
        X, y, train_features = load_data(path, to_split=to_split, selected_features=selected_features)

    # Get the most important features and/or train
    if computeFeatureImportance:
        print("Computing feature importance!")
        ntry = 10
    else:
        print("Training the model!")
        ntry = 1
    feature_importance_total = np.zeros((train_features.shape[0], 1))
    for n in range(ntry):
        with measure_time('Training {} out of {}'.format(n+1, ntry)):
            model = RandomForestClassifier(n_estimators=2000,
                                            min_samples_split=5,
                                            min_samples_leaf=1,
                                            max_features='sqrt',
                                            max_depth=10, 
                                            bootstrap=True,
                                            random_state=42,
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
        feature_importances.sort_values('importance', ascending=False, inplace=True)
        feature_importances.to_csv("_Data/Training_dataset/feature_importance.csv")

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


def create_estimator(path, selected_features):
    """
    Output a model .pkl file trained on the whole dataset
    and on a select few features
    """
    nb_features = len(selected_features)
    filename = "_Models/RandomForest_top{}_features.pkl".format(nb_features)

    X, y, _ = load_data(path, to_split=False, selected_features=selected_features)

    print("Creating estimator on dataset with {} best features".format(nb_features))
    model = RandomForestClassifier(n_estimators=2000,
                                min_samples_split=5,
                                min_samples_leaf=1,
                                max_features='sqrt',
                                max_depth=10, 
                                bootstrap=True,
                                random_state=42,
                                n_jobs=-1)
    model.fit(X, y)
    joblib.dump(model, filename)
    print("Saving model as ", filename)

def tune_hyperparameter(path, selected_features=None):
    """
    Get the best hyperparameters.
    """
    # Load the training set
    X, y, _ = load_data(path, to_split=False, selected_features=selected_features)
    
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
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
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}
    
    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestClassifier()
    # Random search of parameters, using 10 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = rf,
                                   param_distributions = random_grid,
                                   n_iter = 100,
                                   cv = 5,
                                   verbose=2,
                                   random_state=42,
                                   n_jobs = -1)
    # Fit the random search model
    rf_random.fit(X, y)

    print("Best parameters", rf_random.best_params_)
    # Best parameters weight 0.8 min 20 matches + surface weighting (all features) : {'n_estimators': 1600, 'min_samples_split': 2, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 10, 'bootstrap': True}
    # Best parameters weight 0.8 min 20 matches + surface weighting (without rank and ranking points) : {'n_estimators': 200, 'min_samples_split': 5, 'min_samples_leaf': 4, 'max_features': 'auto', 'max_depth': 10, 'bootstrap': True}
    # Best parameters weight 0.6 min 20 matches + surface weighting (all features) : {'n_estimators': 2000, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 10, 'bootstrap': True}


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
    train_estimator(path, computeFeatureImportance=True, to_split=True, selected_features=selected_features)
    #create_estimator(path, selected_features)
    