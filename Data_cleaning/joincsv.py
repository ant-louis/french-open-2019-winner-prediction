import pandas as pd 
import os

matches = pd.read_csv('cleaned_matches_data.csv', dtype=object, encoding='utf-8')
players = pd.read_csv('cleaned_players_data.csv', dtype=object, encoding='utf-8')
#--------------------------------MATCHES ---------------------------------------------------------#

#Rename index as ID
players.rename(columns={players.columns[0]:"ID"}, inplace=True)

#Drop index column an match features, as we will not have them
#when predicting the winner
#Drop non numeric columns
to_drop_matches = [
                  'Games won by winner in set 1' ,
                  'Games won by loser in set 1' ,
                  'Games won by winner in set 2' ,
                  'Games won by loser in set 2' ,
                  'Games won by winner in set 3' ,
                  'Games won by loser in set 3' ,
                  'Games won by winner in set 4' ,
                  'Games won by loser in set 4' ,
                  'Games won by winner in set 5' ,
                  'Games won by loser in set 5' ,
                  'Winner sets',
                  'Loser sets',
                  'Tournament',
                  'Completed or retired',
                  'Month',
                  'Year',
                  'Round'
                  ]
                  
matches.drop(columns=to_drop_matches, inplace=True)
matches.drop(matches.columns[0], axis=1, inplace=True)

#Reclean invalid values that are not numeric in a given column
matches.replace(to_replace='NR',value=0,inplace=True)
matches.replace(to_replace=' ',value=0,inplace=True)
matches.replace(to_replace='`1',value=0,inplace=True)


#Rename winner and loser to player A and B
to_rename={'Winner':'PlayerA',
            'Loser'	:'PlayerB',
            'Winner ranking' :'PlayerA ranking',
            'Loser ranking':'PlayerB ranking'
            }
matches.rename(columns=to_rename, inplace=True)

#Classification output winner
matches["PlayerA Win"] = '1'

#Randomly swap playerA and playerB values
#so that the classification doesn't depend on the feature Winner only
to_swapA = ['PlayerA', 'PlayerA ranking']
to_swapB = ['PlayerB','PlayerB ranking']
index = matches.sample(frac=0.5,random_state=42).index
tmp = matches.loc[index,to_swapA].values
matches.loc[index,to_swapA] = matches.loc[index,to_swapB].values
matches.loc[index,to_swapB] = tmp
matches.loc[index,'PlayerA Win'] = 0


#One hot encoding because Decision tree work with valuesnot strings
 
hot_encode_matches = ['Country',
                      'Court',
                      'Series',
                      'Surface',
                    ]
matches = pd.get_dummies(matches, columns = hot_encode_matches) 



#-------------------------------------------PLAYERS-------------------------#
to_drop_players = ['Active','Retired']

players.drop(columns=to_drop_players, inplace=True)
#Reclean invalid values that are not numeric in a given column
players.replace(to_replace='Infinity',value=100,inplace=True)

# Split the match time
players["Match Time"] = players["Match Time"].astype(str).str.split(":").apply(lambda x: int(x[0]) * 60 + int(x[1]))
players.rename(columns={"Match Time":"Match Time Average"}, inplace=True)

# Remove percentages
players['After Losing 1st Set'] = players['After Losing 1st Set'].str.rstrip('%').astype('float') / 100.0
players['After Winning 1st Set'] = players['After Winning 1st Set'].str.rstrip('%').astype('float') / 100.0

hot_encode_player = ['Plays', 'Favorite Surface','Country']
players = pd.get_dummies(players, columns = hot_encode_player)


#-------------------------------------------MERGING----------------------------------------------------------------------------------------------------#
#Merge players and matches, adding suffixes if necessary
players_and_matches = pd.merge(matches,players,left_on="PlayerA",right_on="Name",suffixes=['_Match','_PlayerA'])
players_and_matches = pd.merge(players_and_matches,players,left_on="PlayerB",right_on="Name",suffixes=['_PlayerA','_PlayerB'])


#Drop columns from merged
todrop_merged = ['Name_PlayerA', 'Name_PlayerB', 'PlayerA','PlayerB','Country_0']
players_and_matches.drop(columns=todrop_merged,inplace=True)


#Move player columns to the front
cols = list(players_and_matches.columns.values)
cols.pop(cols.index('ID_PlayerA')) 
cols.pop(cols.index('ID_PlayerB')) 
cols.pop(cols.index('PlayerA Win')) 
players_and_matches = players_and_matches[['ID_PlayerA', 'ID_PlayerB'] + cols + ['PlayerA Win']] #Create new dataframe with columns in the order you want

#Convert ID's to numeric values and sort them
players_and_matches = players_and_matches.apply(pd.to_numeric)
players_and_matches.sort_values(by = ['ID_PlayerA','ID_PlayerB'],inplace=True)



#Write to csv
players_and_matches.to_csv("training_matches_players.csv", index=False)
matches.to_csv("training_matches.csv", index=False)
players.to_csv("training_players.csv", index=False)
