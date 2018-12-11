import pandas as pd
import itertools
import os

from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier

"""Create to_predict csv file from seeds"""
prefix = '/home/tom/Documents/Master1_DataScience/1er QUADRI/Big-Data-Project/Data_cleaning/'
matches = pd.read_csv(os.path.join(prefix, 'training_matches.csv'), dtype=object, encoding='utf-8')
players = pd.read_csv(os.path.join(prefix, 'training_players.csv'), dtype=object, encoding='utf-8')
players.drop(players.columns[0], axis=1, inplace=True)

#--------------------------------MATCHES ---------------------------------------------------------#

seeds = pd.read_csv('seeds_2018.csv', encoding='utf-8')
seeds['Name'] = seeds['Name'].str.upper()
#Generate all combinations
combination = itertools.combinations(list(range(1,33)),2)
df = pd.DataFrame([c for c in combination], columns=['ID_PlayerA','ID_PlayerB'])
to_predict = pd.merge(df,seeds,left_on="ID_PlayerA",right_on="Ranking")
to_predict = pd.merge(to_predict,seeds,left_on="ID_PlayerB",right_on="Ranking", suffixes=['_PlayerA','_PlayerB'])

to_predict.drop(columns= ['Ranking_PlayerA','Ranking_PlayerB'], inplace=True)
to_predict = pd.merge(to_predict, players, left_on="Name_PlayerA", right_on="Name",suffixes=['_PlayerA','_PlayerB'])
to_predict = pd.merge(to_predict, players, left_on="Name_PlayerB", right_on="Name",suffixes=['_PlayerA','_PlayerB'])

#Add all columns that were in the training set
merged = pd.read_csv(os.path.join(prefix, 'training_matches_players.csv'), dtype=object, encoding='utf-8')
for col in merged.columns.values:
    if col not in to_predict.columns:
        to_predict[col] = 0

#Drop unncessary columns
to_drop = ['Name_PlayerA',
            'Name_PlayerB',
            'PlayerA Win'
            ]

to_predict.drop(columns=to_drop,inplace=True)

#We play at Roland Garros
to_predict['Nb sets max'] = 3
to_predict['Surface_Clay'] = 1

#Update the rankings
to_predict['PlayerA ranking'] = to_predict['ID_PlayerA'].values
to_predict['PlayerB ranking'] = to_predict['ID_PlayerB'].values



#Convert ID's to numeric values and sort them
to_predict = to_predict.apply(pd.to_numeric)
to_predict.sort_values(by = ['ID_PlayerA','ID_PlayerB'],inplace=True)

#Export to csv
prefix = '/home/tom/Documents/Master1_DataScience/1er QUADRI/Big-Data-Project/Predict_tournament/'
to_predict.to_csv(os.path.join(prefix, 'to_predict.csv'),index=False)


#------------------------------------------------------Predicting--------------------------------------------
to_predict = pd.read_csv(os.path.join(prefix, 'to_predict.csv'), dtype=float, encoding='utf-8')

#Import estimator
filename = "RandomForest_depth3.pkl"
if(os.path.isfile(filename)):
    model = joblib.load(filename)
    y_pred = model.predict(to_predict)

# Creating the prediction result
prediction = pd.DataFrame()
prediction['ID_PlayerA'] = to_predict['ID_PlayerA']
prediction['ID_PlayerB'] = to_predict['ID_PlayerB']
prediction['Winner'] = 0
for i,y in enumerate(y_pred):
    if y == 0:
        prediction.iloc[i,2] = to_predict.iloc[i ,1]
    else:
        prediction.iloc[i,2] = to_predict.iloc[i, 0]

prediction.to_csv(os.path.join(prefix, 'matches.csv'),index=False)


