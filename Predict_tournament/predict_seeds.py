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
        "--createFile",
        type=str,
        default='n',
        nargs='?',
        const=True,
        help="Create new seeds file before preditcting - y/n")
    parser.add_argument(
        "--inputFile",
        type=str,
        nargs='?',
        const=True,
        help="Input file. If createFile is 'y', (seeds_20XX.csv, to_predict20XX.csv) is an example for  \
                an input and output file pair. \n \
                Else, the pair  (to_predict20XX.csv, matches_20XX.csv) is recommended.")
    parser.add_argument(
        "--outputFile",
        type=str,
        default='matches_prediction.csv',
        nargs='?',
        const=True,
        help="Output file. If createFile is 'y', (seeds_20XX.csv, to_predict20XX.csv) is an example for  \
                an input and output file pair. \n \
                Else, the pair  (to_predict20XX.csv, matches_20XX.csv) is recommended.")
    parser.add_argument(
        "--model",
        type=str,
        default="RandomForest_diff.pkl",
        nargs='?',
        const=True,
        help="Name of '.pkl' ML model")

    arguments, _ = parser.parse_known_args()
    return arguments


def createToPredictFile(seed_file, output_file):

    """Create to_predict csv file from seeds"""
    
    print("Creating prediction data from  {} and saving to {}...".format(seed_file, output_file))

    players = pd.read_csv('../Data_cleaning/training_data/training_players.csv', dtype=object, encoding='utf-8')
    players.drop(players.columns[0], axis=1, inplace=True)

    #--------------------------------MATCHES ---------------------------------------------------------#

    seeds = pd.read_csv(seed_file, encoding='utf-8')

    #Normalize and uppercase names
    seeds['Name'] = seeds['Name'].str.normalize('NFD').str.upper()

    #Generate all combinations
    combination = itertools.combinations(list(range(1,33)),2)
    df_combination = pd.DataFrame([c for c in combination], columns=['ID_PlayerA','ID_PlayerB'])

    # Match the seed number to the player and merge the two
    to_predict = pd.merge(df_combination,seeds,left_on="ID_PlayerA",right_on="Ranking")
    to_predict = pd.merge(to_predict,seeds,left_on="ID_PlayerB",right_on="Ranking", suffixes=['_PlayerA','_PlayerB'])

    to_predict.drop(columns= ['Ranking_PlayerA','Ranking_PlayerB'], inplace=True)
    to_predict = pd.merge(to_predict, players, left_on="Name_PlayerA", right_on="Name",suffixes=['_PlayerA','_PlayerB'])
    to_predict = pd.merge(to_predict, players, left_on="Name_PlayerB", right_on="Name",suffixes=['_PlayerA','_PlayerB'])

    cols = list(to_predict.columns.values)


    # -----------------------------------DIFFERENCE -------------------------
    # Separating numeric columns from non numeric columns

    non_numeric_cols = [
    'ID_PlayerA',
    'ID_PlayerB',

    'Favorite Surface_All-Rounder_PlayerA',
    'Favorite Surface_Carpet_PlayerA',
    'Favorite Surface_Clay_PlayerA',
    'Favorite Surface_Fast_PlayerA',
    'Favorite Surface_Fastest_PlayerA',
    'Favorite Surface_Firm_PlayerA',
    'Favorite Surface_Grass_PlayerA',
    'Favorite Surface_Hard_PlayerA',
    'Favorite Surface_Non-Carpet_PlayerA',
    'Favorite Surface_Non-Grass_PlayerA',
    'Favorite Surface_Non-Hard_PlayerA',
    'Favorite Surface_None_PlayerA',
    'Favorite Surface_Slow_PlayerA',
    'Favorite Surface_Soft_PlayerA',
    'Plays_0_PlayerA',

    'Favorite Surface_All-Rounder_PlayerB',
    'Favorite Surface_Carpet_PlayerB',
    'Favorite Surface_Clay_PlayerB',
    'Favorite Surface_Fast_PlayerB',
    'Favorite Surface_Fastest_PlayerB',
    'Favorite Surface_Firm_PlayerB',
    'Favorite Surface_Grass_PlayerB',
    'Favorite Surface_Hard_PlayerB',
    'Favorite Surface_Non-Carpet_PlayerB',
    'Favorite Surface_Non-Grass_PlayerB',
    'Favorite Surface_Non-Hard_PlayerB',
    'Favorite Surface_None_PlayerB',
    'Favorite Surface_Slow_PlayerB',
    'Favorite Surface_Soft_PlayerB',
    'Plays_0_PlayerB',

    ]

    numeric_cols = [col for col in cols if col not in non_numeric_cols]

    #Drop redundant hand play variable (already in variable 'Plays')
    hands = ['Plays_Left-handed_PlayerA','Plays_Right-handed_PlayerA','Plays_Left-handed_PlayerB','Plays_Right-handed_PlayerB']
    numeric_cols = [col for col in numeric_cols if col not in hands]

    #Get numeric columns for each player separately and create dataframes
    PlayerA_numeric_cols = numeric_cols[numeric_cols.index('Current Rank_PlayerA'):numeric_cols.index('Current Rank_PlayerB')-1]
    PlayerB_numeric_cols = numeric_cols[numeric_cols.index('Current Rank_PlayerB'):]
    playerA_df = to_predict[PlayerA_numeric_cols]
    playerB_df = to_predict[PlayerB_numeric_cols]

    playerA_df = playerA_df.apply(pd.to_numeric)
    playerB_df = playerB_df.apply(pd.to_numeric)

    # Difference in stats between PlayerA and playerB
    players_diff = pd.DataFrame()
    playerB_df.columns = PlayerA_numeric_cols #Names of columns must be the same when subtracting
    players_diff[PlayerA_numeric_cols] = playerA_df.sub(playerB_df, axis = 'columns')

    #Updating column names (remove suffix _PlayerA)
    column_names_diff = [s[:-8] +'_diff' for s in PlayerA_numeric_cols]
    players_diff.columns = column_names_diff 


    # Concatenating into new dataframe
    to_predict = pd.concat([to_predict[non_numeric_cols[:-1]], 
                                        players_diff, 
                                        to_predict[non_numeric_cols[-1]]], axis=1)

    #Add all columns that were in the training set
    cols_to_add = ['Nb sets max',
                'Court_Indoor',
                'Court_Outdoor',
                'Series_ATP250',
                'Series_ATP500',
                'Series_Grand Slam',
                'Series_International',
                'Series_International Gold',
                'Series_Masters',
                'Series_Masters 1000',
                'Series_Masters Cup',
                'Surface_Carpet',
                'Surface_Clay',
                'Surface_Grass',
                'Surface_Hard']
    for col in cols_to_add:
        to_predict[col] = 0

    # We play at Roland Garros
    to_predict['Nb sets max'] = 5
    to_predict['Surface_Clay'] = 1
    to_predict['Court_Outdoor'] = 1
    to_predict['Court_Indoor'] = 0

    # Convert ID's to numeric values and sort them
    to_predict = to_predict.apply(pd.to_numeric)
    to_predict.sort_values(by = ['ID_PlayerA','ID_PlayerB'],inplace=True)

    # Export to csv
    to_predict.to_csv(output_file,index=False)
    print("Success !")


#------------------------------------------------------Predicting--------------------------------------------

def predictSeeds(input_file, output_file, modelName):
    to_predict = pd.read_csv(input_file, dtype=float, encoding='utf-8')
    
    to_predict_without_id = to_predict.drop(columns=['ID_PlayerA', 'ID_PlayerB'],inplace=False)
    y_pred_proba = None

    #Import estimator
    estimator_file = modelName

    # Predicting the output, extracting probabilities
    if(os.path.isfile(estimator_file)):
        print("Predicting from {}... using the model {}".format(input_file, modelName))
        model = joblib.load(estimator_file)
        y_pred_proba = model.predict_proba(to_predict_without_id)
    else:
        print("No estimator found. Exiting...")
        exit()

    # Creating the prediction result
    print("Assembling prediction and saving to {}...".format(output_file))
    prediction = pd.DataFrame()
    prediction['ID_PlayerA'] = to_predict['ID_PlayerA']
    prediction['ID_PlayerB'] = to_predict['ID_PlayerB']

    prediction['WinnerA_proba'] = y_pred_proba[:,0]
    prediction['WinnerB_proba'] = y_pred_proba[:,1]
    # for i,y in enumerate(y_pred):
    #     if y == 0:
    #         prediction.iloc[i,2] = to_predict.iloc[i ,1]
    #     elif y == 1:
    #         prediction.iloc[i,2] = to_predict.iloc[i, 0]

    prediction.to_csv(output_file,index=False)
    print("Success !")


if __name__=='__main__':

    args = parse_arguments()
    
    create_file = args.createFile
    input_file = args.inputFile
    output_file = args.outputFile
    model_name = args.model
    
    if create_file == 'y':
        createToPredictFile(input_file, output_file)
    else:
        predictSeeds(input_file, output_file, model_name)
