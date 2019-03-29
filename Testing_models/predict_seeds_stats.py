
import pandas as pd
import itertools
import os
import sys
import argparse

from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier


def parse_arguments():
    parser = argparse.ArgumentParser(
        "Tennis Prediction - Create seed matchups and compute win probabilities")

    parser.add_argument(
        "--inputFile",
        type=str,
        nargs='?',
        const=True,
        help="Input file. If createFile is 'y', (seeds_20XX.csv, seeds_player_stats20XX.csv) is an example for  \
                an input and output file pair. \n \
                Else, the pair  (seeds_player_stats20XX.csv, matches_20XX.csv) is recommended.")
    parser.add_argument(
        "--outputFile",
        type=str,
        default='matches_prediction.csv',
        nargs='?',
        const=True,
        help="Output file. If createFile is 'y', (seeds_20XX.csv, seeds_player_stats20XX.csv) is an example for  \
                an input and output file pair. \n \
                Else, the pair  (seeds_player_stats20XX.csv, matches_20XX.csv) is recommended.")
    parser.add_argument(
        "--model",
        type=str,
        nargs='?',
        const=True,
        help="Name of '.pkl' ML model")

    arguments, _ = parser.parse_known_args()
    return arguments

def createFile(input_file, seeds_file):
    data = pd.read_csv(input_file, encoding='utf-8')
    seeds_df = pd.read_csv(seeds_file, encoding='utf-8')
    playerA_df = pd.DataFrame()
    playerB_df = pd.DataFrame()

    # Take matches from 2018 before the 27th May 2018
    condition2017 = (data['Year'] == 2017) & (data['Day'] >= 199)
    condition2018 = (data['Year'] == 2018) & (data['Day'] <= 177)
    matches_2018 = data[condition2018]
    matches_2017 = data[condition2017]
    matches_before_RG = pd.concat([matches_2017, matches_2018])

    # Upper case and normalize names
    from unidecode import unidecode
    seeds_df['Name'] = [unidecode(x) for x in seeds_df['Name'].values]

    for name in seeds_df['Name'].values:
        # All matches from player 'name'
        PlayerA_matches = matches_before_RG[matches_before_RG['PlayerA_name'] == name]
        PlayerB_matches = matches_before_RG[matches_before_RG['PlayerB_name'] == name]

        # Take the match closest to Roland Garros 2018
        PlayerA_matches.sort_values(by=['Year', 'Day'], inplace=True)
        PlayerB_matches.sort_values(by=['Year', 'Day'], inplace=True)

        # Check if the player is player A or B in the most recent game
        if not PlayerA_matches.empty and not PlayerB_matches.empty:
            if PlayerA_matches.iloc[-1, :]['Day'] > PlayerB_matches.iloc[-1, :]['Day']:
                closest = PlayerA_matches.iloc[-1, :]
                player_columns = [
                    'PlayerA_name',
                    'PlayerA_id',
                    'PlayerA_Win%',
                    'PlayerA_height',
                    'PlayerA_age',
                    'PlayerA_rank',
                    'PlayerA_rank_points',
                    'PlayerA_ace',
                    'PlayerA_df',
                    'PlayerA_svpt',
                    'PlayerA_1stIn',
                    'PlayerA_1stWon',
                    'PlayerA_2ndWon',
                    'PlayerA_SvGms',
                    'PlayerA_bpSaved',
                    'PlayerA_bpFaced',
                    'PlayerA_minutes',
                    'PlayerA_bestof',
                    'PlayerA_hand_L',
                    'PlayerA_hand_R'
                ]
                playerA_df = playerA_df.append(closest[player_columns], ignore_index=True)

            else:
                closest = PlayerB_matches.iloc[-1, :]

                player_columns = [
                    'PlayerB_name',
                    'PlayerB_id',
                    'PlayerB_Win%',
                    'PlayerB_height',
                    'PlayerB_age',
                    'PlayerB_rank',
                    'PlayerB_rank_points',
                    'PlayerB_ace',
                    'PlayerB_df',
                    'PlayerB_svpt',
                    'PlayerB_1stIn',
                    'PlayerB_1stWon',
                    'PlayerB_2ndWon',
                    'PlayerB_SvGms',
                    'PlayerB_bpSaved',
                    'PlayerB_bpFaced',
                    'PlayerB_minutes',
                    'PlayerB_bestof',
                    'PlayerB_hand_L',
                    'PlayerB_hand_R',
                ]
                playerB_df = playerB_df.append(closest[player_columns], ignore_index=True)

        elif not PlayerB_matches.empty:
            closest = PlayerB_matches.iloc[-1, :]

            player_columns = [
                'PlayerB_name',
                'PlayerB_id',
                'PlayerB_Win%',
                'PlayerB_height',
                'PlayerB_age',
                'PlayerB_rank',
                'PlayerB_rank_points',
                'PlayerB_ace',
                'PlayerB_df',
                'PlayerB_svpt',
                'PlayerB_1stIn',
                'PlayerB_1stWon',
                'PlayerB_2ndWon',
                'PlayerB_SvGms',
                'PlayerB_bpSaved',
                'PlayerB_bpFaced',
                'PlayerB_minutes',
                'PlayerB_bestof',
                'PlayerB_hand_L',
                'PlayerB_hand_R',
            ]
            playerB_df = playerB_df.append(closest[player_columns], ignore_index=True)
        elif not PlayerA_matches.empty:
            closest = PlayerA_matches.iloc[-1, :]
            player_columns = [
                'PlayerA_name',
                'PlayerA_id',
                'PlayerA_Win%',
                'PlayerA_height',
                'PlayerA_age',
                'PlayerA_rank',
                'PlayerA_rank_points',
                'PlayerA_ace',
                'PlayerA_df',
                'PlayerA_svpt',
                'PlayerA_1stIn',
                'PlayerA_1stWon',
                'PlayerA_2ndWon',
                'PlayerA_SvGms',
                'PlayerA_bpSaved',
                'PlayerA_bpFaced',
                'PlayerA_minutes',
                'PlayerA_bestof',
                'PlayerA_hand_L',
                'PlayerA_hand_R'
            ]
            playerA_df = playerA_df.append(closest[player_columns], ignore_index=True)
        else:
            print(name)


    # Remove A and B from playerA.. and playerB columns
    playerA_df.rename(columns=lambda x: str(x[:6] + x[7:]), inplace=True)
    playerB_df.rename(columns=lambda x: str(x[:6] + x[7:]), inplace=True)

    # Concatenate into one dataframe
    seeds_player_stats = pd.concat([playerA_df, playerB_df])

    # Reorder columns
    cols = ['Player_name',
            'Player_id',
            'Player_height',
            'Player_age',
            'Player_rank',
            'Player_rank_points',
            'Player_bestof',
            'Player_Win%',
            'Player_ace',
            'Player_df',
            'Player_svpt',
            'Player_1stIn',
            'Player_1stWon',
            'Player_2ndWon',
            'Player_SvGms',
            'Player_minutes',
            'Player_bpSaved',
            'Player_bpFaced',
            'Player_hand_L',
            'Player_hand_R',
    ]
    seeds_player_stats = seeds_player_stats[cols]
    seeds_player_stats.drop(columns=['Player_id'], inplace=True)

    # Add the seed rank to the stats
    seeds_player_stats = pd.merge(seeds_df, seeds_player_stats, left_on="Name", right_on="Player_name")
    seeds_player_stats.drop(columns=['Name'], inplace=True)

    seeds_player_stats.to_csv("seeds_player_stats.csv", index=False)

    # Generate all combinations
    combination = itertools.combinations(range(1, 33), 2)
    to_predict = pd.DataFrame([c for c in combination], columns=['Rank_A', 'Rank_B'])

    # Merge the players to the combinations based on their seed rank
    to_predict = pd.merge(to_predict, seeds_player_stats,left_on="Rank_A", right_on="Ranking")
    to_predict = pd.merge(to_predict, seeds_player_stats, left_on="Rank_B",right_on="Ranking", suffixes=['_PlayerA', '_PlayerB'])
    # Duplicates due to merge
    to_predict.drop(columns=['Ranking_PlayerA','Ranking_PlayerB'], inplace=True)

    to_rename = {
        'Player_name_PlayerA': 'PlayerA_name',
        'Player_height_PlayerA': 'PlayerA_height',
        'Player_age_PlayerA': 'PlayerA_age',
        'Player_rank_PlayerA': 'PlayerA_rank',
        'Player_rank_points_PlayerA': 'PlayerA_rank_points',
        'Player_bestof_PlayerA' : 'PlayerA_bestof',
        'Player_Win%_PlayerA': 'PlayerA_Win%',
        'Player_ace_PlayerA': 'PlayerA_ace',
        'Player_df_PlayerA': 'PlayerA_df',
        'Player_svpt_PlayerA': 'PlayerA_svpt',
        'Player_1stIn_PlayerA': 'PlayerA_1stIn',
        'Player_1stWon_PlayerA': 'PlayerA_1stWon',
        'Player_2ndWon_PlayerA': 'PlayerA_2ndWon',
        'Player_SvGms_PlayerA': 'PlayerA_SvGms',
        'Player_minutes_PlayerA' : 'PlayerA_minutes',
        'Player_bpSaved_PlayerA': 'PlayerA_bpSaved',
        'Player_bpFaced_PlayerA': 'PlayerA_bpFaced',
        'Player_hand_L_PlayerA': 'PlayerA_hand_L',
        'Player_hand_R_PlayerA': 'PlayerA_hand_R',
        'Player_name_PlayerB': 'PlayerB_name',
        'Player_height_PlayerB': 'PlayerB_height',
        'Player_age_PlayerB': 'PlayerB_age',
        'Player_rank_PlayerB': 'PlayerB_rank',
        'Player_rank_points_PlayerB': 'PlayerB_rank_points',
        'Player_bestof_PlayerB' : 'PlayerB_bestof',
        'Player_Win%_PlayerB': 'PlayerB_Win%',
        'Player_ace_PlayerB': 'PlayerB_ace',
        'Player_df_PlayerB': 'PlayerB_df',
        'Player_svpt_PlayerB': 'PlayerB_svpt',
        'Player_1stIn_PlayerB': 'PlayerB_1stIn',
        'Player_1stWon_PlayerB': 'PlayerB_1stWon',
        'Player_2ndWon_PlayerB': 'PlayerB_2ndWon',
        'Player_SvGms_PlayerB': 'PlayerB_SvGms',
        'Player_minutes_PlayerB' : 'PlayerB_minutes',
        'Player_bpSaved_PlayerB': 'PlayerB_bpSaved',
        'Player_bpFaced_PlayerB': 'PlayerB_bpFaced',
        'Player_hand_L_PlayerB': 'PlayerB_hand_L',
        'Player_hand_R_PlayerB': 'PlayerB_hand_R',

    }
    to_predict.rename(columns=to_rename, inplace=True)

    # Reorder columns
    cols = ['Rank_A',
            'Rank_B',
            'PlayerA_name',
            'PlayerB_name',
            'PlayerA_Win%',
            'PlayerA_height',
            'PlayerA_age',
            'PlayerA_rank',
            'PlayerA_rank_points',
            'PlayerA_ace',
            'PlayerA_df',
            'PlayerA_svpt',
            'PlayerA_1stIn',
            'PlayerA_1stWon',
            'PlayerA_2ndWon',
            'PlayerA_SvGms',
            'PlayerA_bpSaved',
            'PlayerA_bpFaced',
            'PlayerA_minutes',
            'PlayerA_bestof',
            'PlayerA_hand_L',
            'PlayerA_hand_R',
            'PlayerB_Win%',
            'PlayerB_height',
            'PlayerB_age',
            'PlayerB_rank',
            'PlayerB_rank_points',
            'PlayerB_ace',
            'PlayerB_df',
            'PlayerB_svpt',
            'PlayerB_1stIn',
            'PlayerB_1stWon',
            'PlayerB_2ndWon',
            'PlayerB_SvGms',
            'PlayerB_bpSaved',
            'PlayerB_bpFaced',
            'PlayerB_minutes',
            'PlayerB_bestof',
            'PlayerB_hand_L',
            'PlayerB_hand_R',
   ]

    to_predict = to_predict[cols]
    # Sort values based on seed rank
    to_predict.sort_values(by = ['Rank_A','Rank_B'],inplace=True)

    # Difference in stats between PlayerA and playerB
    playerA_df = to_predict.iloc[:,4:22]
    playerB_df = to_predict.iloc[:,22:40]
    players_diff = pd.DataFrame()
    playerB_df.columns = list(playerA_df.columns) #Names of columns must be the same when subtracting
    players_diff[playerB_df.columns] = playerA_df.sub(playerB_df, axis = 'columns')

    #Updating column names
    column_names_diff = [s[8:] +'_diff' for s in list(playerA_df.columns)]
    players_diff.columns = column_names_diff

    # Concatenate into one dataframe
    to_predict = pd.concat([to_predict.iloc[:,0:4],players_diff,to_predict.iloc[:,40:]],axis=1)
    
    # Adding all the match data, removing unnecessary columns for prediction
    data_diff = pd.read_csv('new_stats_data_diff.csv', encoding='utf-8')
    missing_columns = [x for x in data_diff.columns if x not in to_predict.columns]
    for col in missing_columns:
        to_predict[col] = 0

    to_drop = [
        'Unnamed: 0',
        'Year',
        'Day', 
        'PlayerA_id', 
        'PlayerB_id', 
        'PlayerA Win']

    to_predict.drop(columns=to_drop, inplace=True)
    to_predict['best_of'] = 5
    to_predict['surface_Clay'] = 1
    to_predict['draw_size_128.0'] = 1
    to_predict['tourney_level_A'] = 1
    to_predict['round_R32'] = 1

    # Checking if all the colums are present (for testing purposes only)
    # print([x for x in data_diff.columns if x not in to_predict.columns])
    print([x for x in data_diff.columns if x not in to_predict.columns])
    # Should print ['Unnamed: 0', 'Year', 'Day', 'PlayerA_id', 'PlayerB_id', 'PlayerA Win']

    to_predict.to_csv(output_file, index=False, sep=',', encoding='utf-8', float_format='%.6f', decimal='.')


def predictSeeds(input_file, output_file, modelName, selected_features=None):

    df = pd.read_csv(input_file, delimiter=',', encoding='utf-8')

    # If no features selected by the user
    if not selected_features:
        # Take all numerical features
        df = df.iloc[:,4:]
        selected_features = df.columns

    X = df[selected_features].values.squeeze()
    
    y_pred_proba = None

    # Import model, predict output, extract probabilities
    if(os.path.isfile(modelName)):
        print("Predicting from {}... using the model {}".format(input_file, modelName))
        model = joblib.load(modelName)
        y_pred_proba = model.predict_proba(X)
    else:
        print("No estimator found. Exiting...")
        exit()

    # Creating the prediction result
    print("Assembling prediction and saving to {}...".format(output_file))
    prediction = pd.DataFrame()
    prediction['Rank_A'] = df['Rank_A']
    prediction['Rank_B'] = df['Rank_B']

    prediction['WinnerA_proba'] = y_pred_proba[:,0]
    prediction['WinnerB_proba'] = y_pred_proba[:,1]

    prediction.to_csv(output_file,index=False)
    print("Success !")

if __name__=='__main__':

    # args = parse_arguments()
    
    # input_file = args.inputFile
    # output_file = args.outputFile
    # model_name = args.model

    input_file = 'to_predict.csv'
    output_file = 'RandomForest_5_pred.csv'
    # model_name = 'RandomForest_stats_5feat.pkl'
    model_name = 'RandomForest_stats_5feat.pkl'
    seeds = 'seeds_2018.csv'
    # createFile('new_stats_data_standard.csv', seeds)

    # Select the best features (ones with the most importance)
    NBFEATURES = 5
    features_df = pd.read_csv('feature_importance.csv', sep=',')
    features_list = features_df.iloc[:NBFEATURES, 0].tolist()

    predictSeeds(input_file, output_file , model_name, features_list)
