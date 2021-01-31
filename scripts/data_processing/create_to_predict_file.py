import pandas as pd
import numpy as np
import itertools
import os
import sys
from sklearn.externals import joblib
from sklearn import preprocessing


def create_matches_file(players_stats):
    """
    Create the file of all possible matches in order to predict the results.
    """
    # Read the csv files
    df = pd.read_csv(players_stats, header=0)
    df.rename(columns={'Unnamed: 0': 'id'}, inplace=True)
    players_id = df['id']
    df.drop(labels=['id'], axis=1, inplace=True)
    df.insert(1, 'id', players_id)

    # Generate all possible matches between players
    combination = itertools.combinations(df['Name'].values, 2)
    to_predict_df = pd.DataFrame([c for c in combination], columns=['PlayerA_name', 'PlayerB_name'])

    # Add new columns
    cols = ['PlayerA_name',
            'PlayerB_name',
            'PlayerA_id',
            'PlayerA_FR',
            'PlayerA_righthanded', 
            'PlayerA_age', 
            'PlayerA_rank', 
            'PlayerA_rank_points', 
            'PlayerA_Win%', 
            'PlayerA_bestof', 
            'PlayerA_minutes', 
            'PlayerA_svpt%', 
            'PlayerA_1st_serve%', 
            'PlayerA_1st_serve_won%', 
            'PlayerA_2nd_serve_won%', 
            'PlayerA_ace%', 
            'PlayerA_df%', 
            'PlayerA_bp_faced%', 
            'PlayerA_bp_saved%',
            'PlayerB_id',
            'PlayerB_FR',
            'PlayerB_righthanded', 
            'PlayerB_age', 
            'PlayerB_rank', 
            'PlayerB_rank_points', 
            'PlayerB_Win%', 
            'PlayerB_bestof', 
            'PlayerB_minutes', 
            'PlayerB_svpt%', 
            'PlayerB_1st_serve%', 
            'PlayerB_1st_serve_won%', 
            'PlayerB_2nd_serve_won%', 
            'PlayerB_ace%', 
            'PlayerB_df%', 
            'PlayerB_bp_faced%', 
            'PlayerB_bp_saved%']
    to_predict_df = to_predict_df.reindex(columns=cols)
    
    # Fill in the stats of each player for each possible match
    for i, match in to_predict_df.iterrows():
        name_A = match['PlayerA_name']
        name_B = match['PlayerB_name']

        # Put stats of PlayerA
        tmp_df = df.loc[df['Name'] == name_A]
        tmp_df.columns = ['Name',
                    'PlayerA_id',
                    'PlayerA_FR',
                    'PlayerA_righthanded', 
                    'PlayerA_age', 
                    'PlayerA_rank', 
                    'PlayerA_rank_points', 
                    'PlayerA_Win%', 
                    'PlayerA_bestof', 
                    'PlayerA_minutes', 
                    'PlayerA_svpt%', 
                    'PlayerA_1st_serve%', 
                    'PlayerA_1st_serve_won%', 
                    'PlayerA_2nd_serve_won%', 
                    'PlayerA_ace%', 
                    'PlayerA_df%', 
                    'PlayerA_bp_faced%', 
                    'PlayerA_bp_saved%']
        to_predict_df.at[i, 2:19] = tmp_df.iloc[0, 1:]

        # Put stats of Player B
        tmp_df = df.loc[df['Name'] == name_B]
        tmp_df.columns = ['Name',
            'PlayerB_id',
            'PlayerB_FR',
            'PlayerB_righthanded', 
            'PlayerB_age', 
            'PlayerB_rank', 
            'PlayerB_rank_points', 
            'PlayerB_Win%', 
            'PlayerB_bestof', 
            'PlayerB_minutes', 
            'PlayerB_svpt%', 
            'PlayerB_1st_serve%', 
            'PlayerB_1st_serve_won%', 
            'PlayerB_2nd_serve_won%', 
            'PlayerB_ace%', 
            'PlayerB_df%', 
            'PlayerB_bp_faced%', 
            'PlayerB_bp_saved%']
        to_predict_df.at[i, 19:] = tmp_df.iloc[0, 1:]
        
    # New feature "Same_handedness"
    to_predict_df['Same_handedness'] = np.where(to_predict_df.PlayerA_righthanded == to_predict_df.PlayerB_righthanded, 1, 0)
    to_predict_df.drop(columns=['PlayerA_righthanded'], inplace = True)
    to_predict_df.drop(columns=['PlayerB_righthanded'], inplace = True)
    
    # Difference in stats between PlayerA and playerB
    to_diffA = ['PlayerA_age',
    'PlayerA_rank',
    'PlayerA_rank_points',
    'PlayerA_Win%',
    'PlayerA_bestof',
    'PlayerA_minutes',
    'PlayerA_svpt%',
    'PlayerA_1st_serve%',
    'PlayerA_1st_serve_won%',
    'PlayerA_2nd_serve_won%',
    'PlayerA_ace%',
    'PlayerA_df%',
    'PlayerA_bp_faced%',
    'PlayerA_bp_saved%']

    to_diffB = ['PlayerB_age',
    'PlayerB_rank',
    'PlayerB_rank_points',
    'PlayerB_Win%',
    'PlayerB_bestof',
    'PlayerB_minutes',
    'PlayerB_svpt%',
    'PlayerB_1st_serve%',
    'PlayerB_1st_serve_won%',
    'PlayerB_2nd_serve_won%',
    'PlayerB_ace%',
    'PlayerB_df%',
    'PlayerB_bp_faced%',
    'PlayerB_bp_saved%']
    playerA_df = to_predict_df.loc[:, to_diffA]
    playerB_df = to_predict_df.loc[:, to_diffB]
    players_diff = pd.DataFrame()
    playerB_df.columns = list(playerA_df.columns) #Names of columns must be the same when subtracting
    players_diff[playerB_df.columns] = playerA_df.sub(playerB_df, axis = 'columns')

    #Updating column names
    column_names_diff = [s[8:] +'_diff' for s in list(playerA_df.columns)]
    players_diff.columns = column_names_diff
    
    # Concatenate differences with previous dataframe
    diff_df = pd.concat([to_predict_df.iloc[:,0:2],
                         to_predict_df['PlayerA_id'],
                         to_predict_df['PlayerB_id'],
                         to_predict_df['PlayerA_FR'],
                         to_predict_df['PlayerB_FR'],
                         to_predict_df['Same_handedness'],
                         players_diff], axis=1)

    # Add extra features
    diff_df['best_of'] = 5.0
    diff_df['draw_size'] = 128.0
    diff_df['surface_Carpet'] = 0.0
    diff_df['surface_Clay'] = 1.0
    diff_df['surface_Grass'] = 0.0
    diff_df['surface_Hard'] = 0.0
    
    # Standardize the data
    scaler = preprocessing.StandardScaler()
    diff_df.iloc[:,7:21] = scaler.fit_transform(diff_df.iloc[:,7:21])

    # Save dataset
    diff_df.to_csv('../../data/predictions/to_predict_data_French_Open_2019.csv', sep=',', encoding='utf-8', float_format='%.10f', decimal='.')


def predict_all_matches(matches_to_predict, model_path, selected_features=None):
    """
    Create a csv file with the results of all possibles matches of French Open.
    """
    df = pd.read_csv(matches_to_predict, delimiter=',', encoding='utf-8', index_col=0)
    
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
                            'bp_saved%_diff',
                            'best_of',
                            'draw_size',
                            'surface_Carpet',
                            'surface_Clay',
                            'surface_Grass',
                            'surface_Hard']
    print("Selected features :", selected_features)
    X = df[selected_features].values.squeeze()
    
    # Import model, predict output, extract probabilities
    if(os.path.isfile(model_path)):
        print("Predicting from {}... using the model {}".format(matches_to_predict, model_path))
        model = joblib.load(model_path)
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)
    else:
        print("No estimator found. Exiting...")
        exit()

    # Creating the prediction result
    prediction = pd.DataFrame()
    prediction['PlayerA_name'] = df['PlayerA_name']
    prediction['PlayerB_name'] = df['PlayerB_name']
    prediction['PlayerA_id'] = df['PlayerA_id']
    prediction['PlayerB_id'] = df['PlayerB_id']
    prediction['PlayerA_win'] = y_pred
    prediction['PlayerA_winning_proba'] = y_pred_proba[:,1]
    prediction['PlayerB_winning_proba'] = y_pred_proba[:,0]

    prediction.to_csv('../../data/predictions/predictions_2019_matches.csv', index=False)


if __name__ == "__main__":
    # Files
    players_stats = '../../data/predictions/stats_players_2019.csv'
    matches_to_predict = '../../data/predictions/to_predict_data_French_Open_2019.csv'
    model_path = '_Models/RandomForest_top15_features.pkl'
    
    # Create the file containing all possible matches to predict
    #create_matches_file(players_stats)
    
    # Predict the results of all these matches
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
    predict_all_matches(matches_to_predict, model_path, selected_features=selected_features)
