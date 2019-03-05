import pandas as pd
import itertools
import os
import sys
import argparse

from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier

PREFIX = '/home/tom/Documents/Uliege/Big-Data-Project'

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
        default="RandomForest_opt.pkl",
        nargs='?',
        const=True,
        help="Name of '.pkl' ML model")

    arguments, _ = parser.parse_known_args()
    return arguments


def createToPredictFile(seed_file, output_file):

    """Create to_predict csv file from seeds"""
    
    print("Creating prediction data from  {} and saving to {}...".format(seed_file, output_file))

    players = pd.read_csv(os.path.join(PREFIX, 'Data_cleaning/training_players.csv'), dtype=object, encoding='utf-8')
    players.drop(players.columns[0], axis=1, inplace=True)

    #--------------------------------MATCHES ---------------------------------------------------------#

    seeds = pd.read_csv(seed_file, encoding='utf-8')
    #Normalize and uppercase names
    seeds['Name'] = seeds['Name'].str.normalize('NFD').str.upper()

    #Generate all combinations
    combination = itertools.combinations(list(range(1,33)),2)
    df = pd.DataFrame([c for c in combination], columns=['ID_PlayerA','ID_PlayerB'])

    # Match the seed number to the player and merge the two
    to_predict = pd.merge(df,seeds,left_on="ID_PlayerA",right_on="Ranking")
    to_predict = pd.merge(to_predict,seeds,left_on="ID_PlayerB",right_on="Ranking", suffixes=['_PlayerA','_PlayerB'])

    to_predict.drop(columns= ['Ranking_PlayerA','Ranking_PlayerB'], inplace=True)
    to_predict = pd.merge(to_predict, players, left_on="Name_PlayerA", right_on="Name",suffixes=['_PlayerA','_PlayerB'])
    to_predict = pd.merge(to_predict, players, left_on="Name_PlayerB", right_on="Name",suffixes=['_PlayerA','_PlayerB'])

    #Add all columns that were in the training set
    merged = pd.read_csv(os.path.join(PREFIX, 'Data_cleaning/training_matches_players.csv'), dtype=object, encoding='utf-8')
    for col in merged.columns.values:
        if col not in to_predict.columns:
            to_predict[col] = 0

    #Drop unncessary columns
    to_drop = ['Name_PlayerA',
                'Name_PlayerB',
                'PlayerA Win',
                'Plays_Left-handed_PlayerA',
                'Plays_Right-handed_PlayerA',
                'Plays_Left-handed_PlayerB',
                'Plays_Right-handed_PlayerB'
                ]

    to_predict.drop(columns=to_drop,inplace=True)

    # We play at Roland Garros
    to_predict['Nb sets max'] = 5
    to_predict['Surface_Clay'] = 1

    # Update the rankings
    to_predict['PlayerA ranking'] = to_predict['ID_PlayerA'].values
    to_predict['PlayerB ranking'] = to_predict['ID_PlayerB'].values


    # Convert ID's to numeric values and sort them
    to_predict = to_predict.apply(pd.to_numeric)
    to_predict.sort_values(by = ['ID_PlayerA','ID_PlayerB'],inplace=True)

    # Export to csv
    to_predict.to_csv(os.path.join(PREFIX, 'Predict_tournament', output_file),index=False)


#------------------------------------------------------Predicting--------------------------------------------

def predictSeeds(input_file, output_file, modelName):
    to_predict = pd.read_csv(os.path.join(PREFIX, 'Predict_tournament/', input_file), dtype=float, encoding='utf-8')
    
    to_predict_without_id = to_predict.drop(columns=['ID_PlayerA', 'ID_PlayerB'],inplace=False)
    y_pred_proba = None

    #Import estimator
    estimator_file = modelName
    path = os.path.join(PREFIX,'Predict_tournament/', estimator_file)

    # Predicting the output, extracting probabilities
    if(os.path.isfile(path)):
        print("Predicting from {}... using the model {}".format(input_file, modelName))
        model = joblib.load(path)
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

    prediction.to_csv(os.path.join(PREFIX, os.path.join('Predict_tournament/', output_file)),index=False)
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
