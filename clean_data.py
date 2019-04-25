import pandas as pd
import numpy as np


# Read csv file
df = pd.read_csv('_Data/Original_dataset/all_games.csv', sep=',')

# Drop useless columns
to_drop = ['tourney_id',
           'tourney_name',
           'tourney_level',
           'match_num',
           'winner_seed',
           'winner_entry',
           'loser_seed',
           'loser_entry',
           'winner_ht',
           'loser_ht',
           'score']
df.drop(columns=to_drop, inplace = True)

# Rename columns
to_rename={'winner_name': 'PlayerA_name',
            'loser_name': 'PlayerB_name',
            'winner_id': 'PlayerA_id',
            'winner_age': 'PlayerA_age',
            'winner_rank': 'PlayerA_rank',
            'winner_rank_points':'PlayerA_rank_points',
            'loser_id': 'PlayerB_id',
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
            'loser_hand':'PlayerB_hand',
           'winner_ioc' : 'PlayerA_ioc',
           'loser_ioc' : 'PlayerB_ioc',
          }
df.rename(columns=to_rename, inplace=True)

# Classification output winner
df["PlayerA_Win"] = 1.0

# Drop matches with missing values
df = df.dropna()
df = df[(df.PlayerA_svpt != 0) &
         (df.PlayerA_1stIn != 0) &
         (df.PlayerA_1stWon != 0) &
         (df.PlayerA_2ndWon != 0)]
df = df[(df.PlayerB_svpt != 0) &
         (df.PlayerB_1stIn != 0) &
         (df.PlayerB_1stWon != 0) &
         (df.PlayerB_2ndWon != 0)]
df.reset_index(drop=True, inplace=True)

# Split the date and compute the day number
df.rename(columns={'tourney_date': 'Year'}, inplace=True)
df['Month'] = df["Year"].astype(str).str[4:6]
df['Day'] = df["Year"].astype(str).str[6:8]
df['Year'] = df["Year"].astype(str).str[0:4]
df['Year'] = df['Year'].astype(int)
df['Month'] = df['Month'].astype(int)
df['Day'] = df['Day'].astype(int)
df['Month'] = (df['Month'] - 1)*30
df['Day'] = df['Month'] + df['Day']
df.drop(columns=['Month'], inplace = True)

# Convert R and L to numbers
df = df.replace({'PlayerA_hand': 'R', 'PlayerB_hand': 'R'}, 1)
df = df.replace({'PlayerA_hand': 'U', 'PlayerB_hand': 'U'}, 1)
df = df.replace({'PlayerA_hand': np.nan, 'PlayerB_hand': np.nan}, 1)
df = df.replace({'PlayerA_hand': 'L', 'PlayerB_hand': 'L'}, 0)
df.rename(columns={'PlayerA_hand' : 'PlayerA_righthanded', 'PlayerB_hand' : 'PlayerB_righthanded'}, inplace=True)

# Feature for French players (more weight)
df.loc[df.PlayerA_ioc != 'FRA', 'PlayerA_ioc'] = 0
df.loc[df.PlayerB_ioc != 'FRA', 'PlayerB_ioc'] = 0
df.loc[df.PlayerA_ioc == 'FRA', 'PlayerA_ioc'] = 1
df.loc[df.PlayerB_ioc == 'FRA', 'PlayerB_ioc'] = 1
df.rename(columns={'PlayerA_ioc' : 'PlayerA_FR', 'PlayerB_ioc' : 'PlayerB_FR'}, inplace=True)

# Convert surface into categorical data
df = df[df.surface != 'None']
df = pd.get_dummies(df, columns=['surface'])

# Get rid of Round Robin tournaments
df = df[df['round'] != 'RR']
df = df[df['round'] != 'BR']

# Convert round into numerical data
df = df.replace({'round': 'R128'}, 128.)
df = df.replace({'round': 'R64'}, 64.)
df = df.replace({'round': 'R32'}, 32.)
df = df.replace({'round': 'R16'}, 16.)
df = df.replace({'round': 'QF'}, 8.)
df = df.replace({'round': 'SF'}, 4.)
df = df.replace({'round': 'F'}, 2.)

# Reorder columns and save
cols = ['PlayerA_name',
        'PlayerB_name',
        'Year',
        'Day',
        'best_of',
        'draw_size',
        'round',
        'minutes',
        'PlayerA_id',
        'PlayerB_id',
        'PlayerA_FR',
        'PlayerB_FR',
        'PlayerA_righthanded',
        'PlayerB_righthanded',
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
        'PlayerA_Win',
        'surface_Carpet',
        'surface_Clay',
        'surface_Grass',
        'surface_Hard']
df = df[cols]

df.to_csv('_Data/Original_dataset/cleaned_data.csv', sep=',', encoding='utf-8', float_format='%.10f', decimal='.')
