import pandas as pd
import numpy as np

# Set tables reading options
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

# Read csv
df = pd.read_csv('../../data/original_dataset/preprocessed_data.csv', header=0, index_col=0)
players_df = pd.read_csv('../../data/predictions/players_2016.csv', header=0, index_col=0)

#-------------------------------------------------------------------------------------------
# GET LATEST AGE, RANK, RANKING POINTS OF EACH PLAYER
#-------------------------------------------------------------------------------------------

# Limit date before Roland Garros
#curr_year = 2018 
#max_day = 148
#curr_year = 2017
#max_day = 149
curr_year = 2016
max_day = 143

# Create new dataframe to contain info about the 128 players
index = np.array(np.arange(1,129))
columns = ['PlayerA_FR',
          'PlayerA_righthanded',
           'PlayerA_age',
           'PlayerA_rank',
           'PlayerA_rank_points']
new_df = pd.DataFrame(index=index, columns=columns)

# Columns of the players' stats
playerA_cols = [2,3,10,12,14,15,16]
playerB_cols = [2,3,11,13,25,26,27]

for i, player in players_df.iterrows():
    name = player['PlayerA_Name']

    # Get ranking, rank points and age of player just before Roland Garros
    playerA_rows = df.index[(df['PlayerA_name'] == name) & (df['Year'] + df['Day']/365 < curr_year + max_day/365)].tolist()
    playerB_rows = df.index[(df['PlayerB_name'] == name) & (df['Year'] + df['Day']/365 < curr_year + max_day/365)].tolist()
    playerA_df = df.iloc[playerA_rows, playerA_cols]
    playerB_df = df.iloc[playerB_rows, playerB_cols]
    playerB_df.columns = list(playerA_df)
    tmp_df = pd.concat([playerA_df, playerB_df], ignore_index=True)
    if tmp_df.empty:
        continue
        
    # Sort by latest date
    tmp_df.sort_values(by=['Year', 'Day'], ascending=[False, False], inplace=True)

    # Add it in the new df
    new_df.at[i,:] = tmp_df.iloc[0,:]

# Fill last missing values by median
new_df.fillna(new_df.median(), inplace=True)

#-------------------------------------------------------------------------------------------
# COMPUTE LATEST STATS OF EACH PLAYER
#-------------------------------------------------------------------------------------------

# Weights of surface weighting
corr_df = pd.read_csv('../../data/new_stats_dataset/correlation_between_surfaces.csv', header=0, index_col=0)
weight_carpet = corr_df.loc['Clay','Carpet']
weight_grass = corr_df.loc['Clay','Grass']
weight_hard = corr_df.loc['Clay','Hard']
weight_clay = corr_df.loc['Clay','Clay']

# Compute the stats of the players
new_columns = ['PlayerA_Win%',
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
new_df = new_df.reindex(columns=[*new_df.columns.tolist(), *new_columns])

# Columns of the players' stats
playerA_cols = [2,3,4,7] + list(range(17,25)) + [36,37,38,39]
playerB_cols = [2,3,4,7] + list(range(28,36)) + [36,37,38,39]

for i, player in players_df.iterrows():
    name = player['PlayerA_Name']
    
    # Take all past matches of PLAYER 1 and look for same id in playerA and playerB
    playerA_rows = df.index[(df['PlayerA_name'] == name) & (df['Year'] + df['Day']/365 < curr_year + max_day/365)].tolist()
    playerB_rows = df.index[(df['PlayerB_name'] == name) & (df['Year'] + df['Day']/365 < curr_year + max_day/365)].tolist()
    playerA_df = df.iloc[playerA_rows, playerA_cols]
    playerA_df['Win'] = 1
    playerB_df = df.iloc[playerB_rows, playerB_cols]
    playerB_df['Win'] = 0
    playerB_df.columns = list(playerA_df)
    tmp_df = pd.concat([playerA_df, playerB_df], ignore_index=True)
    if tmp_df.empty:
        continue
    
    # Compute a weight for each past match of the player
    tmp_df['elapsing_time'] = (curr_year + max_day/365) - (tmp_df['Year'] + tmp_df['Day']/365)
    tmp_df['weight'] = tmp_df['elapsing_time'].apply(lambda t: 0.6**t)
    tmp_df.loc[tmp_df['elapsing_time'] <= 1, 'weight'] = 1
    tmp_df['weight'] = (0.95 * tmp_df['weight']) + (0.05 * (weight_carpet*tmp_df['surface_Carpet'] + weight_clay*tmp_df['surface_Clay'] + weight_grass*tmp_df['surface_Grass'] + weight_hard*tmp_df['surface_Hard']))
    tmp_df.drop(columns=['Year', 'Day', 'elapsing_time', 'surface_Carpet', 'surface_Clay', 'surface_Grass', 'surface_Hard'], inplace = True)

    # Compute the weighted average of the player
    weighted_means = np.average(tmp_df, weights=tmp_df['weight'],axis=0)
    weighted_df = pd.DataFrame(weighted_means.reshape(-1, len(weighted_means)), columns=list(tmp_df.columns))
    weighted_df = weighted_df.drop('weight', axis=1)
    
    # Add stats of the player in new dataframe
    new_df.at[i, 8:] = weighted_df.iloc[0, 2:10]
    new_df.at[i, 'PlayerA_bestof'] = weighted_df['best_of']
    new_df.at[i, 'PlayerA_minutes'] = weighted_df['minutes']
    new_df.at[i, 'PlayerA_Win%'] = weighted_df['Win']
    
# Concat and updating columns names
new_df = pd.concat([players_df, new_df], axis=1)
column_names = [s[8:] for s in list(new_df.columns)]
new_df.columns = column_names

# Fill last missing values by median
new_df.fillna(new_df.median(), inplace=True)

# Save dataset
new_df.to_csv('../../data/predictions/stats_players_2016_weight06.csv', sep=',', encoding='utf-8', float_format='%.10f', decimal='.')
