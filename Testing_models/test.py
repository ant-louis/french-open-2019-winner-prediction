# For testing purposes
    # Select the best features (ones with the most importance)
    # If we want to compute the feature importance, we obviously use all of them
    if not computeFeatureImportance:
        features_df = pd.read_csv('feature_importance.csv', sep=',')
        features_list = features_df.iloc[:nb_features, 0].tolist()




    # train_estimator(path, computeFeatureImportance=False, nb_features=48, to_split=True)
    # train_estimator(path, computeFeatureImportance=False, nb_features=4, to_split=True)
    # train_estimator(path, computeFeatureImportance=False, nb_features=5, to_split=True)
    # train_estimator(path, computeFeatureImportance=False, nb_features=6, to_split=True)
    # train_estimator(path, computeFeatureImportance=False, nb_features=7, to_split=True)
    # train_estimator(path, computeFeatureImportance=False, nb_features=14, to_split=True)
    create_estimator(path, 5)
    # plot_feature_importance(48, 'all_features_importance.svg')
    # plot_feature_importance(14, '14_features_importance.svg')
    # plot_feature_importance(5, '5_features_importance.svg')



    # Sorting because we don't want to predict the past matches with data about the future
    all_df.sort_values(by=['Year', 'Day'], inplace=True)
    # Shuffle is False because we don't want to predict the past matches with data about the future
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, shuffle=False)
