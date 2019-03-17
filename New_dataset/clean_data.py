import pandas as pd


# Read csv file
df = pd.read_csv('all_games.csv',sep=',')

# Keep only matches between the best 32 players at the moment
df = df[df['winner_rank'] <= 32]
df = df[df['loser_rank'] <= 32]

# Drop columns
to_drop = ['winner_seed',
            'winner_entry',
            'loser_seed',
            'loser_entry',
            'match_num',
            'score',
            'tourney_id',
            'tourney_name',
            'winner_ioc',
            'loser_ioc']
df.drop(columns=to_drop, inplace = True)

# Rename columns
to_rename={'winner_name': 'PlayerA_name',
            'loser_name': 'PlayerB_name',
            'winner_id': 'PlayerA_id',
            'winner_ht': 'PlayerA_height',
            'winner_age': 'PlayerA_age',
            'winner_rank': 'PlayerA_rank',
            'winner_rank_points':'PlayerA_rank_points',
            'loser_id': 'PlayerB_id',
            'loser_ht': 'PlayerB_height',
            'loser_age': 'PlayerB_age',
            'loser_rank': 'PlayerB_rank',
            'loser_rank_points':'PlayerB_rank_points',
            'w_ace':'PlayerA_ace',
            'w_df':'PlayerA_df',
            'w_svpt':'PlayerA_svpt',
            'w_1stIn':'PlayerA_1stIn',
            'w_1stWon':'PlayerA_1stWon',
            'w_2ndWon':'PlayerA_2ndWon',
            'w_SvGms':'PlayerA_SvGms',
            'w_bpSaved':'PlayerA_bpSaved',
            'w_bpFaced':'PlayerA_bpFaced',
            'l_ace':'PlayerB_ace',
            'l_df':'PlayerB_df',
            'l_svpt':'PlayerB_svpt',
            'l_1stIn':'PlayerB_1stIn',
            'l_1stWon':'PlayerB_1stWon',
            'l_2ndWon':'PlayerB_2ndWon',
            'l_SvGms':'PlayerB_SvGms',
            'l_bpSaved':'PlayerB_bpSaved',
            'l_bpFaced':'PlayerB_bpFaced',
            'winner_hand':'PlayerA_hand',
            'loser_hand':'PlayerB_hand'
          }
df.rename(columns=to_rename, inplace=True)

# Omit matches where a stat is missing
df = df.dropna()

#Upper case and normalize names
df['PlayerA_name'] = df['PlayerA_name'].str.normalize('NFD').str.upper()
df['PlayerB_name'] = df['PlayerB_name'].str.normalize('NFD').str.upper()

#Move player names to the front
cols = list(df.columns.values)
cols.pop(cols.index('PlayerA_name')) 
cols.pop(cols.index('PlayerB_name')) 
#Create new dataframe with columns in the order you want
df = df[['PlayerA_name', 'PlayerB_name'] + cols] 

to_hot_encode = [
    'surface', 
    'draw_size', 
    'tourney_level',
    'PlayerA_hand',
    'PlayerB_hand',
    'round'
]
# Convert into categorical data
df = pd.get_dummies(df, columns = to_hot_encode)

#Classification output winner
df["PlayerA Win"] = '1'

# Save dataset
df.to_csv('cleaned_data.csv', sep=',', encoding='utf-8', float_format='%.0f', decimal='.')
