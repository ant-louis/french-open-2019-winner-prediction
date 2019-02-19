import pandas as pd
import itertools
import os
import sys

from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier

PREFIX = '/home/tom/Documents/Uliege/Big-Data-Project'


def createToPredictFile(seed_file):


    """Create to_predict csv file from seeds"""
    matches = pd.read_csv(os.path.join(PREFIX, 'Data_cleaning/training_matches.csv'), dtype=object, encoding='utf-8')
    players = pd.read_csv(os.path.join(PREFIX, 'Data_cleaning/training_players.csv'), dtype=object, encoding='utf-8')
    players.drop(players.columns[0], axis=1, inplace=True)

    #--------------------------------MATCHES ---------------------------------------------------------#

    seeds = pd.read_csv(seed_file, encoding='utf-8')
    #Normalize and uppercase names
    seeds['Name'] = seeds['Name'].str.normalize('NFD').str.upper()

    #Generate all combinations
    combination = itertools.combinations(list(range(1,33)),2)
    df = pd.DataFrame([c for c in combination], columns=['ID_PlayerA','ID_PlayerB'])
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
                'PlayerA Win'
                ]

    to_predict.drop(columns=to_drop,inplace=True)

    # We play at Roland Garros
    to_predict['Nb sets max'] = 3
    to_predict['Surface_Clay'] = 1

    # Update the rankings
    to_predict['PlayerA ranking'] = to_predict['ID_PlayerA'].values
    to_predict['PlayerB ranking'] = to_predict['ID_PlayerB'].values



    # Convert ID's to numeric values and sort them
    to_predict = to_predict.apply(pd.to_numeric)
    to_predict.sort_values(by = ['ID_PlayerA','ID_PlayerB'],inplace=True)

    # Export to csv
    to_predict.to_csv(os.path.join(PREFIX, 'Predict_tournament/Test_data/to_predict.csv'),index=False)


#------------------------------------------------------Predicting--------------------------------------------

def predictSeeds(output_file):
    to_predict = pd.read_csv(os.path.join(PREFIX, 'Predict_tournament/Test_data/to_predict.csv'), dtype=float, encoding='utf-8')
    
    to_predict_without_id = to_predict.drop(columns=['ID_PlayerA', 'ID_PlayerB'],inplace=False)

    #Import estimator
    estimator_file = "RandomForest_opt.pkl"
    if(os.path.isfile(estimator_file)):
        model = joblib.load(estimator_file)
        y_pred_proba = model.predict_proba(to_predict_without_id)

    print(y_pred_proba)

    # Creating the prediction result
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

    prediction.to_csv(os.path.join(PREFIX, os.path.join('Predict_tournament/Test_data/', output_file)),index=False)


if __name__=='__main__':

    if(len(sys.argv) != 3):
        print("Call with \"python predict_seeds.py predict example_output.csv\"")
        print(" or \"python predict_seeds.py createFile example_input.csv\"")
        exit()
    
    if(sys.argv[1] == 'predict'):
        predictSeeds(sys.argv[2])
    elif(sys.argv[1] == 'createFile'):
        createToPredictFile(sys.argv[2])

    else:
        print("Call with \"python predict_seeds.py predict example_output.csv\"")
        print(" or \"python predict_seeds.py createFile example_input.csv\"")
        exit()
