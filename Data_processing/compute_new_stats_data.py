import pandas as pd
import numpy as np


# Read the csv file
df = pd.read_csv('Data/preprocessed_data.csv', header=0, index_col=0)

# Set tables reading options
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 1000)

# Create a copy of dataframe
new_df = df.copy(deep=True)

# Create new features "best_of" and "minutes" for players
new_df['PlayerA_bestof'] = 0.0
new_df['PlayerB_bestof'] = 0.0
new_df['PlayerA_minutes'] = 0.0
new_df['PlayerB_minutes'] = 0.0
# Create new features for percentage of winning
new_df['PlayerA_Win%'] = 0.0
new_df['PlayerB_Win%'] = 0.0

# Columns of the players' stats
playerA_cols = [2,3,4,7] + list(range(18,26))
playerB_cols = [2,3,4,7] + list(range(30,38))


# FOR EACH MATCH OF DATAFRAME
for i, match in df.iterrows():
    print(i)
    
    # Get the current date of the match
    curr_year = match['Year']
    curr_day = match['Day']

    # COMPUTE STATS OF PLAYER 1
    id_1 = match['PlayerA_id']

    # Take all past matches of that player looking for the id in playerA and playerB
    p1_playerA_rows = df.index[(df['PlayerA_id'] == id_1) & (df['Year'] + df['Day']/365 < curr_year + curr_day/365)].tolist()
    p1_playerB_rows = df.index[(df['PlayerB_id'] == id_1) & (df['Year'] + df['Day']/365 < curr_year + curr_day/365)].tolist()
    p1_playerA_df = df.iloc[p1_playerA_rows, playerA_cols]
    p1_playerA_df['Win'] = 1
    p1_playerB_df = df.iloc[p1_playerB_rows, playerB_cols]
    p1_playerB_df['Win'] = 0
    p1_playerB_df.columns = list(p1_playerA_df)
    tmp1_df = pd.concat([p1_playerA_df,p1_playerB_df], ignore_index=True)
    # If it is empty, continue
    if tmp1_df.empty:
        continue

    # Compute a weight for each match
    tmp1_df['elapsing_time'] = (curr_year + curr_day/365) - (tmp1_df['Year'] + tmp1_df['Day']/365)
    tmp1_df['weight'] = tmp1_df['elapsing_time'].apply(lambda t: 0.8**t)
    tmp1_df.loc[tmp1_df['elapsing_time'] <= 0.5, 'weight'] = 1
    tmp1_df.drop(columns=['Year', 'Day', 'elapsing_time'], inplace = True)

    # Compute the weighted average
    weighted_means = np.average(tmp1_df, weights=tmp1_df['weight'],axis=0)
    weighted1_df = pd.DataFrame(weighted_means.reshape(-1, len(weighted_means)), columns=list(tmp1_df.columns))
    weighted1_df = weighted1_df.drop('weight', axis=1)

    # Add stats in new dataframe
    new_df.at[i, 18:26] = weighted1_df.iloc[0, 2:10]
    new_df.at[i, 'PlayerA_bestof'] = weighted1_df['best_of']
    new_df.at[i, 'PlayerA_minutes'] = weighted1_df['minutes']
    new_df.at[i, 'PlayerA_Win%'] = weighted1_df['Win']


    # COMPUTE STATS OF PLAYER 2
    id_2 = match['PlayerB_id']

    # Take all past matches of that player looking for the id in playerA and playerB
    p2_playerA_rows = df.index[(df['PlayerA_id'] == id_2) & (df['Year'] + df['Day']/365 < curr_year + curr_day/365)].tolist()
    p2_playerB_rows = df.index[(df['PlayerB_id'] == id_2) & (df['Year'] + df['Day']/365 < curr_year + curr_day/365)].tolist()
    p2_playerA_df = df.iloc[p2_playerA_rows, playerA_cols]
    p2_playerA_df['Win'] = 1
    p2_playerB_df = df.iloc[p2_playerB_rows, playerB_cols]
    p2_playerB_df['Win'] = 0
    p2_playerA_df.columns = list(p2_playerB_df)
    tmp2_df = pd.concat([p2_playerA_df,p2_playerB_df], ignore_index=True)
    # If it is empty, continue
    if tmp2_df.empty:
        continue

    # Compute a weight for each match
    tmp2_df['elapsing_time'] = (curr_year + curr_day/365) - (tmp2_df['Year'] + tmp2_df['Day']/365)
    tmp2_df['weight'] = tmp2_df['elapsing_time'].apply(lambda t: 0.8**t)
    tmp2_df.loc[tmp2_df['elapsing_time'] <= 0.5, 'weight'] = 1
    tmp2_df.drop(columns=['Year', 'Day', 'elapsing_time'], inplace = True)

    # Compute the weighted average
    weighted_means = np.average(tmp2_df, weights=tmp2_df['weight'],axis=0)
    weighted2_df = pd.DataFrame(weighted_means.reshape(-1, len(weighted_means)), columns=list(tmp2_df.columns))
    weighted2_df = weighted2_df.drop('weight', axis=1)

    # Add stats in new dataframe
    new_df.at[i, 30:38] = weighted2_df.iloc[0, 2:10]
    new_df.at[i, 'PlayerB_bestof'] = weighted2_df['best_of']
    new_df.at[i, 'PlayerB_minutes'] = weighted2_df['minutes']
    new_df.at[i, 'PlayerB_Win%'] = weighted2_df['Win']


# Keep only matches where enough data was considered for computing stats
new_df = new_df[(new_df['PlayerA_Win%'] != 0) & (new_df['PlayerB_Win%'] != 0)]
new_df = new_df[(new_df['PlayerA_Win%'] != 1) & (new_df['PlayerB_Win%'] != 1)]

# Drop "minutes" feature (won't know that for predicting matches)
new_df.drop(columns=['minutes'], inplace = True)

# Drop "best_of" feature (bestof diff gives more info)
new_df.drop(columns=['best_of'], inplace = True)

# Reorder columns
cols = ['PlayerA_name',
        'PlayerB_name',
        'Year',
        'Day',
        'draw_size',
        'round',
        'PlayerA_id',
        'PlayerB_id',
        'PlayerA_FR',
        'PlayerB_FR',
        'PlayerA_righthanded',
        'PlayerB_righthanded',
         'PlayerA_height',
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
         'PlayerB_height',
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
        'PlayerA_ace%',
        'PlayerB_df%',
        'PlayerB_bp_faced%',
        'PlayerB_bp_saved%',
        'surface_Carpet',
        'surface_Clay',
        'surface_Grass',
        'surface_Hard',
        'PlayerA Win']
new_df = new_df[cols]

# Save dataset
new_df.to_csv('Data/new_stats_data_all_matches.csv', sep=',', encoding='utf-8', float_format='%.10f', decimal='.')
