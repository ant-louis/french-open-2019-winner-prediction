import pandas as pd
import numpy as np


# Read the csv file
df = pd.read_csv('Data/cleaned_data.csv', header=0, index_col=0)

# Set tables reading options
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 1000)

# Convert all numerical values to int
df.iloc[:, 7:] = df.iloc[:, 7:].apply(pd.to_numeric, downcast='float')

# Get rid of Round Robin tournaments
df = df[df['round'] != 'RR']
df = df[df['round'] != 'BR']

# New feature "Same_handedness"
df['Same_handedness'] = np.where(df.PlayerA_righthanded == df.PlayerB_righthanded, 1., 0.)
df.drop(columns=['PlayerA_righthanded'], inplace = True)
df.drop(columns=['PlayerB_righthanded'], inplace = True)

# Compute new features for PlayerA
df['PlayerA_svpt%'] = (df.PlayerA_svpt + df.PlayerA_ace + df.PlayerA_df)/((df.PlayerA_svpt + df.PlayerA_ace + df.PlayerA_df) + (df.PlayerB_svpt + df.PlayerB_ace + df.PlayerB_df))
df['PlayerA_1st_serve%'] = df.PlayerA_1stIn/df.PlayerA_svpt
df['PlayerA_1st_serve_won%'] = df.PlayerA_1stWon/df.PlayerA_1stIn
df['PlayerA_2nd_serve_won%'] = df.PlayerA_2ndWon/(df.PlayerA_svpt - df.PlayerA_1stIn)
df['PlayerA_ace%'] = df.PlayerA_ace/(df.PlayerA_svpt + df.PlayerA_ace + df.PlayerA_df)
df['PlayerA_df%'] = df.PlayerA_df/(df.PlayerA_svpt + df.PlayerA_ace + df.PlayerA_df)
df['PlayerA_bp_faced%'] = df.PlayerA_bpFaced/df.PlayerA_svpt
df['PlayerA_bp_saved%'] = df.PlayerA_bpSaved/df.PlayerA_bpFaced

# Compute new features for PlayerB
df['PlayerB_svpt%'] = (df.PlayerB_svpt + df.PlayerB_ace + df.PlayerB_df)/((df.PlayerA_svpt + df.PlayerA_ace + df.PlayerA_df) + (df.PlayerB_svpt + df.PlayerB_ace + df.PlayerB_df))
df['PlayerB_1st_serve%'] = df.PlayerB_1stIn/df.PlayerB_svpt
df['PlayerB_1st_serve_won%'] = df.PlayerB_1stWon/df.PlayerB_1stIn
df['PlayerB_2nd_serve_won%'] = df.PlayerB_2ndWon/(df.PlayerB_svpt - df.PlayerB_1stIn)
df['PlayerB_ace%'] = df.PlayerB_ace/(df.PlayerB_svpt + df.PlayerB_ace + df.PlayerB_df)
df['PlayerB_df%'] = df.PlayerB_df/(df.PlayerB_svpt + df.PlayerB_ace + df.PlayerB_df)
df['PlayerB_bp_faced%'] = df.PlayerB_bpFaced/df.PlayerB_svpt
df['PlayerB_bp_saved%'] = df.PlayerB_bpSaved/df.PlayerB_bpFaced

# Convert NaN because of "bpFaced" that might b equal to 0
df.fillna(0, inplace=True)

# Drop columns
to_dropA = ['PlayerA_ace',
           'PlayerA_df',
           'PlayerA_svpt',
           'PlayerA_1stIn',
           'PlayerA_1stWon',
           'PlayerA_2ndWon',
           'PlayerA_SvGms',
           'PlayerA_bpSaved',
           'PlayerA_bpFaced']
to_dropB = ['PlayerB_ace',
           'PlayerB_df',
           'PlayerB_svpt',
           'PlayerB_1stIn',
           'PlayerB_1stWon',
           'PlayerB_2ndWon',
           'PlayerB_SvGms',
           'PlayerB_bpSaved',
           'PlayerB_bpFaced']
df.drop(columns=to_dropA, inplace = True)
df.drop(columns=to_dropB, inplace = True)

# Convert tound into numerical data
df = df.replace({'round': 'R128'}, 128.)
df = df.replace({'round': 'R64'}, 64.)
df = df.replace({'round': 'R32'}, 32.)
df = df.replace({'round': 'R16'}, 16.)
df = df.replace({'round': 'QF'}, 8.)
df = df.replace({'round': 'SF'}, 4.)
df = df.replace({'round': 'F'}, 2.)

# Reorder columns
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
        'Same_handedness',
         'PlayerA_height',
         'PlayerA_age',
         'PlayerA_rank',
         'PlayerA_rank_points',
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
        'PlayerA Win']
df = df[cols]

# Reset index
df.reset_index(drop=True, inplace=True)

# Save dataset
df.to_csv('Data/preprocessed_data.csv', sep=',', encoding='utf-8', float_format='%.6f', decimal='.')
