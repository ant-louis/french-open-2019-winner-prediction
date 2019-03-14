
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
        nargs='?',
        const=True,
        help="Name of '.pkl' ML model")

    arguments, _ = parser.parse_known_args()
    return arguments


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
    
    input_file = args.inputFile
    output_file = args.outputFile
    model_name = args.model
    
    predictSeeds(input_file, output_file, model_name)
