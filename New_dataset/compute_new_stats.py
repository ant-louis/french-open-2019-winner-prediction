import pandas as pd
import numpy as np
from sklearn import preprocessing


# Read the csv file
df = pd.read_csv('Data/cleaned_data.csv', header=0)

# Set tables reading options
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

# Drop useless index
df = df.drop('Unnamed: 0', axis=1)

# Convert all numerical values to int
df.iloc[:, 2:68] = df.iloc[:, 2:68].apply(pd.to_numeric, downcast='integer')

# Split the date and compute the day number
df.rename(columns={'tourney_date': 'Year'}, inplace=True)
df['Month'] = df["Year"].astype(str).str[4:6]
df['Day'] = df["Year"].astype(str).str[6:8]
df['Year'] = df["Year"].astype(str).str[0:4]
df['Year'] = df['Year'].astype(int)
df['Month'] = df['Month'].astype(int)
df['Day'] = df['Day'].astype(int)
df['Month'] *= 30
df['Day'] = df['Month'] + df['Day']
df.drop(columns=['Month'], inplace = True)

# Reorder columns
cols = ['Year',
        'Day',
        'PlayerA_name',
         'PlayerB_name',
         'PlayerA_id',
         'PlayerA_height',
         'PlayerA_age',
         'PlayerA_rank',
         'PlayerA_rank_points',
         'PlayerB_id',
         'PlayerB_height',
         'PlayerB_age',
         'PlayerB_rank',
         'PlayerB_rank_points',
         'best_of',
         'minutes',
         'PlayerA_ace',
         'PlayerA_df',
         'PlayerA_svpt',
         'PlayerA_1stIn',
         'PlayerA_1stWon',
         'PlayerA_2ndWon',
         'PlayerA_SvGms',
         'PlayerA_bpSaved',
         'PlayerA_bpFaced',
         'PlayerB_ace',
         'PlayerB_df',
         'PlayerB_svpt',
         'PlayerB_1stIn',
         'PlayerB_1stWon',
         'PlayerB_2ndWon',
         'PlayerB_SvGms',
         'PlayerB_bpSaved',
         'PlayerB_bpFaced',
         'surface_Carpet',
         'surface_Clay',
         'surface_Grass',
         'surface_Hard',
         'draw_size_4.0',
         'draw_size_8.0',
         'draw_size_9.0',
         'draw_size_10.0',
         'draw_size_16.0',
         'draw_size_28.0',
         'draw_size_32.0',
         'draw_size_48.0',
         'draw_size_56.0',
         'draw_size_64.0',
         'draw_size_96.0',
         'draw_size_128.0',
         'tourney_level_A',
         'tourney_level_D',
         'tourney_level_F',
         'tourney_level_G',
         'tourney_level_M',
         'PlayerA_hand_L',
         'PlayerA_hand_R',
         'PlayerB_hand_L',
         'PlayerB_hand_R',
         'round_BR',
         'round_F',
         'round_QF',
         'round_R128',
         'round_R16',
         'round_R32',
         'round_R64',
         'round_RR',
         'round_SF',
         'PlayerA Win']
df = df[cols]

# Create a copy of dataframe
new_df = df.copy(deep=True)

# Divide "best_of" and "minutes" variables
new_df['PlayerA_bestof'] = 0.0
new_df['PlayerB_bestof'] = 0.0
new_df['PlayerA_minutes'] = 0.0
new_df['PlayerB_minutes'] = 0.0
# New variables for percentage of winning
new_df['PlayerA_Win%'] = 0.0
new_df['PlayerB_Win%'] = 0.0

# Columns of the players' stats
playerA_cols = [0,1] + list(range(14,25))
playerB_cols = [0,1] + [14,15] + list(range(25,34))


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
    new_df.at[i, 16:25] = weighted1_df.iloc[0, 2:11]
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
    p2_playerB_df.columns = list(p2_playerA_df)
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
    cols = ['best_of',
            'minutes',
            'PlayerB_ace',
            'PlayerB_df',
            'PlayerB_svpt',
            'PlayerB_1stIn',
            'PlayerB_1stWon',
            'PlayerB_2ndWon',
            'PlayerB_SvGms',
            'PlayerB_bpSaved',
            'PlayerB_bpFaced',
            'Win']
    weighted2_df.columns = cols
    new_df.at[i, 25:34] = weighted2_df.iloc[0, 2:11]
    new_df.at[i, 'PlayerB_bestof'] = weighted2_df['best_of']
    new_df.at[i, 'PlayerB_minutes'] = weighted2_df['minutes']
    new_df.at[i, 'PlayerB_Win%'] = weighted1_df['Win']

# Reorder columns
cols = ['PlayerA_name',
        'PlayerB_name',
        'Year',
        'Day',
        'PlayerA_id',
        'PlayerB_id',
        'PlayerA_Win%',
        'PlayerA_height',
        'PlayerA_age',
        'PlayerA_rank',
        'PlayerA_rank_points',
        'PlayerA_ace',
        'PlayerA_df',
        'PlayerA_svpt',
        'PlayerA_1stIn',
        'PlayerA_1stWon',
        'PlayerA_2ndWon',
        'PlayerA_SvGms',
        'PlayerA_bpSaved',
        'PlayerA_bpFaced',
        'PlayerA_minutes',
        'PlayerA_bestof',
        'PlayerA_hand_L',
        'PlayerA_hand_R',
        'PlayerB_Win%',
        'PlayerB_height',
        'PlayerB_age',
        'PlayerB_rank',
        'PlayerB_rank_points',
        'PlayerB_ace',
        'PlayerB_df',
        'PlayerB_svpt',
        'PlayerB_1stIn',
        'PlayerB_1stWon',
        'PlayerB_2ndWon',
        'PlayerB_SvGms',
        'PlayerB_bpSaved',
        'PlayerB_bpFaced',
        'PlayerB_minutes',
        'PlayerB_bestof',
        'PlayerB_hand_L',
        'PlayerB_hand_R',
        'best_of',
        'minutes',
        'surface_Carpet',
        'surface_Clay',
        'surface_Grass',
        'surface_Hard',
        'draw_size_4.0',
        'draw_size_8.0',
        'draw_size_9.0',
        'draw_size_10.0',
        'draw_size_16.0',
        'draw_size_28.0',
        'draw_size_32.0',
        'draw_size_48.0',
        'draw_size_56.0',
        'draw_size_64.0',
        'draw_size_96.0',
        'draw_size_128.0',
        'tourney_level_A',
        'tourney_level_D',
        'tourney_level_F',
        'tourney_level_G',
        'tourney_level_M',
        'round_BR',
        'round_F',
        'round_QF',
        'round_R128',
        'round_R16',
        'round_R32',
        'round_R64',
        'round_RR',
        'round_SF',
        'PlayerA Win']
new_df = new_df[cols]

# Drop "minutes" feature (won't know that for predicting matches)
new_df.drop(columns=['minutes'], inplace = True)

# Standardize the data
scaler = preprocessing.StandardScaler()
new_df.iloc[:,7:21] = scaler.fit_transform(new_df.iloc[:,7:21]) # Data of playerA
new_df.iloc[:,25:39] = scaler.fit_transform(new_df.iloc[:,25:39]) # Data of playerA

# Create new dataframe where we swap PlayerA and PlayerB and then merge the swapped set
# and the original set to get a a symmetric dataset
to_swapA = ['PlayerA_name',
            'PlayerA_id',
            'PlayerA_Win%',
            'PlayerA_height',
            'PlayerA_age',
            'PlayerA_rank',
            'PlayerA_rank_points',
            'PlayerA_ace',
            'PlayerA_df',
            'PlayerA_svpt',
            'PlayerA_1stIn',
            'PlayerA_1stWon',
            'PlayerA_2ndWon',
            'PlayerA_SvGms',
            'PlayerA_bpSaved',
            'PlayerA_bpFaced',
            'PlayerA_minutes',
            'PlayerA_bestof',
            'PlayerA_hand_L',
            'PlayerA_hand_R']

to_swapB = ['PlayerB_name',
            'PlayerB_id',
            'PlayerB_Win%',
            'PlayerB_height',
            'PlayerB_age',
            'PlayerB_rank',
            'PlayerB_rank_points',
            'PlayerB_ace',
            'PlayerB_df',
            'PlayerB_svpt',
            'PlayerB_1stIn',
            'PlayerB_1stWon',
            'PlayerB_2ndWon',
            'PlayerB_SvGms',
            'PlayerB_bpSaved',
            'PlayerB_bpFaced',
            'PlayerB_minutes',
            'PlayerB_bestof',
            'PlayerB_hand_L',
            'PlayerB_hand_R']

idx = new_df.sample(frac=0.5, replace=False).index
tmp = new_df.loc[idx, to_swapA]
new_df.loc[idx, to_swapA] = new_df.loc[idx, to_swapB]
new_df.loc[idx, to_swapB] = tmp
new_df.loc[idx, 'PlayerA Win'] = 0

# Difference in stats between PlayerA and playerB
playerA_df = new_df.iloc[:,6:22]
playerB_df = new_df.iloc[:,24:40]
players_diff = pd.DataFrame()
playerB_df.columns = list(playerA_df.columns) #Names of columns must be the same when subtracting
players_diff[playerB_df.columns] = playerA_df.sub(playerB_df, axis = 'columns')

#Updating column names
column_names_diff = [s[8:] +'_diff' for s in list(playerA_df.columns)]
players_diff.columns = column_names_diff

# Concatenate differences with previous dataframe
final_df = pd.concat([new_df.iloc[:,0:6],players_diff,new_df.iloc[:,40:]],axis=1)

# Save dataset
final_df.to_csv('New_dataset/new_stats_data.csv', sep=',', encoding='utf-8', float_format='%.6f', decimal='.')
