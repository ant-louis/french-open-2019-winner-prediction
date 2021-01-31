import pandas as pd
import numpy as np


# Read the csv file
df = pd.read_csv('../../data/original_dataset/preprocessed_data.csv', header=0, index_col=0)
df.reset_index(drop=True,inplace=True)

# Set tables reading options
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

# Create a copy of dataframe
new_df = df.copy(deep=True)

# Create new features "best_of" and "minutes" for players
new_df['PlayerA_bestof'] = np.nan
new_df['PlayerB_bestof'] = np.nan
new_df['PlayerA_minutes'] = np.nan
new_df['PlayerB_minutes'] = np.nan

# Create new features for percentage of winning
new_df['PlayerA_Win%'] = np.nan
new_df['PlayerB_Win%'] = np.nan

# Weights of surface weighting
corr_df = pd.read_csv('../../data/new_stats_dataset/correlation_between_surfaces.csv', header=0, index_col=0)

# Columns of the players' stats
playerA_cols = [2,3,4,7] + list(range(17,25)) + [36,37,38,39]
playerB_cols = [2,3,4,7] + list(range(28,36)) + [36,37,38,39]

# FOR EACH MATCH OF DATAFRAME
for i, match in df.iterrows():
    print(i)
    
    # Get the current date of the match
    curr_year = match['Year']
    curr_day = match['Day']
    
    # Get the surfaces weights depending on the surface of the match
    if (match['surface_Carpet'] == 1):
        weight_carpet = corr_df.loc['Carpet','Carpet']
        weight_grass = corr_df.loc['Carpet','Grass']
        weight_hard = corr_df.loc['Carpet','Hard']
        weight_clay = corr_df.loc['Carpet','Clay']
    elif (match['surface_Clay'] == 1):
        weight_carpet = corr_df.loc['Clay','Carpet']
        weight_grass = corr_df.loc['Clay','Grass']
        weight_hard = corr_df.loc['Clay','Hard']
        weight_clay = corr_df.loc['Clay','Clay']
    elif (match['surface_Grass'] == 1):
        weight_carpet = corr_df.loc['Grass','Carpet']
        weight_grass = corr_df.loc['Grass','Grass']
        weight_hard = corr_df.loc['Grass','Hard']
        weight_clay = corr_df.loc['Grass','Clay']
    elif (match['surface_Hard'] == 1):
        weight_carpet = corr_df.loc['Hard','Carpet']
        weight_grass = corr_df.loc['Hard','Grass']
        weight_hard = corr_df.loc['Hard','Hard']
        weight_clay = corr_df.loc['Hard','Clay']

    # Take all past matches of PLAYER 1 and look for same id in playerA and playerB
    id_1 = match['PlayerA_id']
    p1_playerA_rows = df.index[(df['PlayerA_id'] == id_1) & (df['Year'] + df['Day']/365 < curr_year + curr_day/365)].tolist()
    p1_playerB_rows = df.index[(df['PlayerB_id'] == id_1) & (df['Year'] + df['Day']/365 < curr_year + curr_day/365)].tolist()
    p1_playerA_df = df.iloc[p1_playerA_rows, playerA_cols]
    p1_playerA_df['Win'] = 1
    p1_playerB_df = df.iloc[p1_playerB_rows, playerB_cols]
    p1_playerB_df['Win'] = 0
    p1_playerB_df.columns = list(p1_playerA_df)
    tmp1_df = pd.concat([p1_playerA_df,p1_playerB_df], ignore_index=True)
    
    # Take all past matches of PLAYER 2 and look for same id in playerA and playerB
    id_2 = match['PlayerB_id']
    p2_playerA_rows = df.index[(df['PlayerA_id'] == id_2) & (df['Year'] + df['Day']/365 < curr_year + curr_day/365)].tolist()
    p2_playerB_rows = df.index[(df['PlayerB_id'] == id_2) & (df['Year'] + df['Day']/365 < curr_year + curr_day/365)].tolist()
    p2_playerA_df = df.iloc[p2_playerA_rows, playerA_cols]
    p2_playerA_df['Win'] = 1
    p2_playerB_df = df.iloc[p2_playerB_rows, playerB_cols]
    p2_playerB_df['Win'] = 0
    p2_playerA_df.columns = list(p2_playerB_df)
    tmp2_df = pd.concat([p2_playerA_df,p2_playerB_df], ignore_index=True)
    
    # Check if players have played enough matches to compute representative stats
    #if (tmp1_df.empty) | (tmp2_df.empty):
    if (len(tmp1_df) < 20) | (len(tmp2_df) < 20):
        continue

    # Compute a weight for each past match of PLAYER 1
    tmp1_df['elapsing_time'] = (curr_year + curr_day/365) - (tmp1_df['Year'] + tmp1_df['Day']/365)
    tmp1_df['weight'] = tmp1_df['elapsing_time'].apply(lambda t: 0.6**t)
    tmp1_df.loc[tmp1_df['elapsing_time'] <= 1, 'weight'] = 1
    tmp1_df['weight'] = (0.95 * tmp1_df['weight']) + (0.05 * (weight_carpet*tmp1_df['surface_Carpet'] + weight_clay*tmp1_df['surface_Clay'] + weight_grass*tmp1_df['surface_Grass'] + weight_hard*tmp1_df['surface_Hard']))
    tmp1_df.drop(columns=['Year', 'Day', 'elapsing_time', 'surface_Carpet', 'surface_Clay', 'surface_Grass', 'surface_Hard'], inplace = True)

    # Compute a weight for each past match of PLAYER 2
    tmp2_df['elapsing_time'] = (curr_year + curr_day/365) - (tmp2_df['Year'] + tmp2_df['Day']/365)
    tmp2_df['weight'] = tmp2_df['elapsing_time'].apply(lambda t: 0.6**t)
    tmp2_df.loc[tmp2_df['elapsing_time'] <= 1, 'weight'] = 1
    tmp2_df['weight'] = (0.95 * tmp2_df['weight']) + (0.05 * (weight_carpet*tmp2_df['surface_Carpet'] + weight_clay*tmp2_df['surface_Clay'] + weight_grass*tmp2_df['surface_Grass'] + weight_hard*tmp2_df['surface_Hard']))
    tmp2_df.drop(columns=['Year', 'Day', 'elapsing_time', 'surface_Carpet', 'surface_Clay', 'surface_Grass', 'surface_Hard'], inplace = True)

    # Compute the weighted average of PLAYER 1
    weighted_means = np.average(tmp1_df, weights=tmp1_df['weight'],axis=0)
    weighted1_df = pd.DataFrame(weighted_means.reshape(-1, len(weighted_means)), columns=list(tmp1_df.columns))
    weighted1_df = weighted1_df.drop('weight', axis=1)

    # Compute the weighted average of PLAYER 2
    weighted_means = np.average(tmp2_df, weights=tmp2_df['weight'],axis=0)
    weighted2_df = pd.DataFrame(weighted_means.reshape(-1, len(weighted_means)), columns=list(tmp2_df.columns))
    weighted2_df = weighted2_df.drop('weight', axis=1)

    # Add stats of of PLAYER 1 in new dataframe
    new_df.at[i, 17:25] = weighted1_df.iloc[0, 2:10]
    new_df.at[i, 'PlayerA_bestof'] = weighted1_df['best_of']
    new_df.at[i, 'PlayerA_minutes'] = weighted1_df['minutes']
    new_df.at[i, 'PlayerA_Win%'] = weighted1_df['Win']

    # Add stats of of PLAYER 1 in new dataframe
    new_df.at[i, 28:36] = weighted2_df.iloc[0, 2:10]
    new_df.at[i, 'PlayerB_bestof'] = weighted2_df['best_of']
    new_df.at[i, 'PlayerB_minutes'] = weighted2_df['minutes']
    new_df.at[i, 'PlayerB_Win%'] = weighted2_df['Win']

# Drop matches with too less stats
new_df = new_df.dropna()
new_df.reset_index(drop=True, inplace=True)

# Drop "minutes" feature (won't know that for predicting matches)
new_df.drop(columns=['minutes'], inplace=True)

# Reorder columns and save
cols = ['PlayerA_name',
        'PlayerB_name',
        'Year',
        'Day',
        'best_of',
        'draw_size',
        'round',
        'PlayerA_id',
        'PlayerB_id',
        'PlayerA_FR',
        'PlayerB_FR',
        'PlayerA_righthanded',
        'PlayerB_righthanded',
         'PlayerA_age',
         'PlayerA_rank',
         'PlayerA_rank_points',
        'PlayerA_Win%',
        'PlayerA_bestof',
        'PlayerA_minutes',
        'PlayerA_svpt%',
        'PlayerA_1st_serve%',
        'PlayerA_1st_serve_won%',
        'PlayerA_2nd_serve_won%',
        'PlayerA_ace%',
        'PlayerA_df%',
        'PlayerA_bp_faced%',
        'PlayerA_bp_saved%',
         'PlayerB_age',
         'PlayerB_rank',
         'PlayerB_rank_points',
        'PlayerB_Win%',
        'PlayerB_bestof',
        'PlayerB_minutes',
         'PlayerB_svpt%',
        'PlayerB_1st_serve%',
        'PlayerB_1st_serve_won%',
        'PlayerB_2nd_serve_won%',
        'PlayerB_ace%',
        'PlayerB_df%',
        'PlayerB_bp_faced%',
        'PlayerB_bp_saved%',
        'surface_Carpet',
        'surface_Clay',
        'surface_Grass',
        'surface_Hard',
        'PlayerA_Win']
new_df = new_df[cols]

# Save dataset
new_df.to_csv('../../data/new_stats_dataset/new_stats_data_weight06_+surface_weighting_min20matches.csv', sep=',', encoding='utf-8', float_format='%.10f', decimal='.')
