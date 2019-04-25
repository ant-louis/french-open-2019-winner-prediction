import pandas as pd
import numpy as np
from sklearn import preprocessing


# Read the csv file
df = pd.read_csv('_Data/New_stats_dataset/new_stats_data_weight06_+surface_weighting_min20matches.csv', header=0, index_col=0)

# Create new dataframe where we swap PlayerA and PlayerB and then merge the swapped set
# and the original set to get a a symmetric dataset
to_swapA = ['PlayerA_name',
'PlayerA_id',
'PlayerA_FR',
'PlayerA_righthanded',
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
'PlayerA_bp_saved%']

to_swapB = ['PlayerB_name',
'PlayerB_id',
'PlayerB_FR',
'PlayerB_righthanded',
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
'PlayerB_bp_saved%']

swapped_df = df.copy(deep=True)
idx = swapped_df.sample(frac=0.5, replace=False).index
tmp = swapped_df.loc[idx, to_swapA].values
swapped_df.loc[idx, to_swapA] = swapped_df.loc[idx, to_swapB].values
swapped_df.loc[idx, to_swapB] = tmp
swapped_df.loc[idx, 'PlayerA_Win'] = 0

# New feature "Same_handedness"
swapped_df['Same_handedness'] = np.where(swapped_df.PlayerA_righthanded == swapped_df.PlayerB_righthanded, 1., 0.)
swapped_df.drop(columns=['PlayerA_righthanded'], inplace = True)
swapped_df.drop(columns=['PlayerB_righthanded'], inplace = True)

# Difference in stats between PlayerA and playerB
to_diffA = ['PlayerA_age',
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
'PlayerA_bp_saved%']

to_diffB = ['PlayerB_age',
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
'PlayerB_bp_saved%']
playerA_df = swapped_df.loc[:, to_diffA]
playerB_df = swapped_df.loc[:, to_diffB]
players_diff = pd.DataFrame()
playerB_df.columns = list(playerA_df.columns) #Names of columns must be the same when subtracting
players_diff[playerB_df.columns] = playerA_df.sub(playerB_df, axis = 'columns')

#Updating column names
column_names_diff = [s[8:] +'_diff' for s in list(playerA_df.columns)]
players_diff.columns = column_names_diff

# Concatenate differences with previous dataframe
diff_df = pd.concat([swapped_df.iloc[:,0:11], players_diff, swapped_df.iloc[:,39:]],axis=1)

# Standardize the data
scaler = preprocessing.StandardScaler()
diff_df.iloc[:,11:25] = scaler.fit_transform(diff_df.iloc[:,11:25])

# Reorder columns and save
cols = ['PlayerA_name', 
        'PlayerB_name',
        'PlayerA_id',
        'PlayerB_id',
        'Year', 
        'Day',
        'PlayerA_FR', 
        'PlayerB_FR',
        'Same_handedness',
        'age_diff', 
        'rank_diff', 
        'rank_points_diff', 
        'Win%_diff', 
        'bestof_diff',
        'minutes_diff', 
        'svpt%_diff', 
        '1st_serve%_diff', 
        '1st_serve_won%_diff',
        '2nd_serve_won%_diff', 
        'ace%_diff', 
        'df%_diff', 
        'bp_faced%_diff',
        'bp_saved%_diff',
        'best_of', 
        'draw_size',
        'surface_Carpet', 
        'surface_Clay',
        'surface_Grass', 
        'surface_Hard',
        'round',
        'PlayerA_Win']
diff_df = diff_df[cols]

diff_df.to_csv('_Data/Training_dataset/training_data_weight06_+surface_weighting_min20matches.csv', sep=',', encoding='utf-8', float_format='%.10f', decimal='.')
