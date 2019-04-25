import pandas as pd
import numpy as np

# Set tables reading options
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

# Read csv
df = pd.read_csv('_Data/Original_dataset/preprocessed_data_with_2019_matches.csv', header=0, index_col=0)
players_df = pd.read_csv('_Data/Predictions/players_2019.csv', header=0, index_col=0)

# Convert all numerical values to int
players_df.iloc[:, 1:] = players_df.iloc[:, 1:].apply(pd.to_numeric, downcast='float')

#-------------------------------------------------------------------------------------------
# COMPUTE LATEST STATS OF EACH PLAYER
#-------------------------------------------------------------------------------------------

# Starting date of Roland-Garros 2019
curr_year = 2019
max_day = 146

# Weights of surface weighting
corr_df = pd.read_csv('_Data/New_stats_dataset/correlation_between_surfaces.csv', header=0, index_col=0)
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
players_df = players_df.reindex(columns=[*players_df.columns.tolist(), *new_columns])

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
    players_df.at[i, 9:] = weighted_df.iloc[0, 2:10]
    players_df.at[i, 'PlayerA_bestof'] = weighted_df['best_of']
    players_df.at[i, 'PlayerA_minutes'] = weighted_df['minutes']
    players_df.at[i, 'PlayerA_Win%'] = weighted_df['Win']
    
# Updating columns names
column_names = [s[8:] for s in list(players_df.columns)]
players_df.columns = column_names

# Fill last missing values by median
players_df.fillna(players_df.median(), inplace=True)

# Save dataset
players_df.to_csv('_Data/Predictions/stats_players_2019.csv', sep=',', encoding='utf-8', float_format='%.10f', decimal='.')
