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
                  'Round',
                  'Country'
                  ]
                  
matches.drop(columns=to_drop_matches, inplace=True)
matches.drop(matches.columns[0], axis=1, inplace=True)

#Reclean invalid values that are not numeric in a given column
matches.replace(to_replace='NR',value=0,inplace=True)
matches.replace(to_replace=' ',value=0,inplace=True)
matches.replace(to_replace='`1',value=0,inplace=True)

#Convert names to uppercase 
matches['Winner'] = matches['Winner'].str.upper()
matches['Loser'] = matches['Loser'].str.upper()


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


#One hot encoding 
hot_encode_matches = ['Court',
                      'Series',
                      'Surface',
                    ]
matches = pd.get_dummies(matches, columns = hot_encode_matches) 



#-------------------------------------------PLAYERS-------------------------------------------#
to_drop_players = ['Active',
                  'Retired',
                  'Country']

players.drop(columns=to_drop_players, inplace=True)

#Reclean invalid values that are not numeric in a given column
players.replace(to_replace='Infinity',value=100,inplace=True)

#Upper case names
players['Name'] = players['Name'].str.upper()

# Split the match time
players["Match Time"] = players["Match Time"].astype(str).str.split(":").apply(lambda x: int(x[0]) * 60 + int(x[1]))
players.rename(columns={"Match Time":"Match Time Average"}, inplace=True)

# Remove percentages
players['After Losing 1st Set'] = players['After Losing 1st Set'].str.rstrip('%').astype('float') / 100.0
players['After Winning 1st Set'] = players['After Winning 1st Set'].str.rstrip('%').astype('float') / 100.0

hot_encode_player = ['Plays', 'Favorite Surface']
players = pd.get_dummies(players, columns = hot_encode_player)


#-------------------------------------------MERGING----------------------------------------------------------------------------------------------------#
#Merge players and matches, adding suffixes if necessary
players_and_matches = pd.merge(matches,players,left_on="PlayerA",right_on="Name",suffixes=['_Match','_PlayerA'])
players_and_matches = pd.merge(players_and_matches,players,left_on="PlayerB",right_on="Name",suffixes=['_PlayerA','_PlayerB'])

#Drop columns from merged
todrop_merged = ['Name_PlayerA', 'Name_PlayerB', 'PlayerA', 'PlayerB']
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

# -----------------------------------------------DIFFERENCE---------------------------------------

import pprint
pp = pprint.PrettyPrinter(indent=4)
# #BEFORE PRINT
# pp.pprint(cols)

# Separating numeric columns from non numeric columns
non_numeric_cols = [
  'ID_PlayerA',
  'ID_PlayerB',
  'Nb sets max',
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
  'Surface_Hard',

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

  'PlayerA Win'
  
]

numeric_cols = [col for col in cols if col not in non_numeric_cols]
print(numeric_cols)
#Drop redundant hand play variable (already in variable 'Plays')
hands = ['Plays_Left-handed_PlayerA','Plays_Right-handed_PlayerA','Plays_Left-handed_PlayerB','Plays_Right-handed_PlayerB']
numeric_cols = [col for col in numeric_cols if col not in hands]

#Get numeric columns for each player separately and create dataframes
PlayerA_numeric_cols = numeric_cols[numeric_cols.index('Current Rank_PlayerA'):numeric_cols.index('Current Rank_PlayerB')]
PlayerB_numeric_cols = numeric_cols[numeric_cols.index('Current Rank_PlayerB'):]
playerA_df = players_and_matches[PlayerA_numeric_cols]
playerB_df = players_and_matches[PlayerB_numeric_cols]


# Difference in stats between PlayerA and playerB
players_diff = pd.DataFrame()
playerB_df.columns = PlayerA_numeric_cols #Names of columns must be the same when subtracting
players_diff[PlayerA_numeric_cols] = playerA_df.sub(playerB_df, axis = 'columns')

#Updating column names
column_names_diff = [s[:-8] +'_diff' for s in PlayerA_numeric_cols]
players_diff.columns = column_names_diff 

# Concatenating into new dataframe
players_and_matches_diff = pd.concat([players_and_matches[non_numeric_cols[:-1]], 
                                      players_diff, 
                                      players_and_matches[non_numeric_cols[-1]]], axis=1)

players_and_matches_diff.to_csv("training_matches_players_diff.csv", index=False)


# AFTER PRINT
all_cols= list(players_and_matches_diff.columns.values)

pp.pprint(all_cols)
