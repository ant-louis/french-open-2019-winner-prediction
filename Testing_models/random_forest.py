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


def load_data(path, to_split=True,selected_features=None):
    """
    Load the csv file and returns (X,y).
    """
    df = pd.read_csv(path, header=0, index_col=0)
    # Sorting because we want our testing set to be the last matches
    df.sort_values(by=['Year', 'Day'], inplace=True)
    y = df['PlayerA Win'].values.squeeze()
    df.drop('PlayerA Win', axis=1, inplace=True)
    #df.drop('bestof_diff', axis=1, inplace=True) # Overfitting feature

    # If no features selected by the user, take all numerical features
    if selected_features is None:
        selected_features = df.iloc[:,11:27].columns.tolist()

    X = df[selected_features].values.squeeze()
    print("Selected features :", selected_features)

    # Shuffle is False because we don't want to predict the past matches with data about the future
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    if to_split:
        return X_train, X_test, y_train, y_test, np.asarray(selected_features)
    else:
        return X, y, np.asarray(selected_features)


def train_estimator(path, computeFeatureImportance=False, to_split=True):
    """
    Train the model.
    If computeFeatureImportance is True, to_split must be True too. We don't want
    to contaminate our test_set by doing a feature extraction on it and then
    testing on it. 
    """

    # Load the training (and testing) set(s)
    if to_split:
        X, X_test, y, y_test, train_features = load_data(path, to_split=to_split)
    else:
        X, y, train_features = load_data(path, to_split=to_split)

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
            model = RandomForestClassifier(n_estimators=1800,
                                            min_samples_split=5,
                                            min_samples_leaf=4,
                                            max_features='auto',
                                            max_depth=10, 
                                            bootstrap=False,
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
        feature_importances.to_csv("Data/feature_importance.csv")

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
    filename = "Models/RandomForest_top{}_features.pkl".format(nb_features)
    features_df = pd.read_csv('Data/feature_importance.csv', index_col=0)

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
    # Best parameters 20/03 {'n_estimators': 1400, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'auto', 'max_depth': 40, 'bootstrap': False}
    # Best parameters 30/03 {'n_estimators': 1800, 'min_samples_split': 5, 'min_samples_leaf': 4, 'max_features': 'auto', 'max_depth': 10, 'bootstrap': False}

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

    path = "Data/training_diff_data.csv"

    #tune_hyperparameter(path)
    train_estimator(path, computeFeatureImportance=False, to_split=True)
    
