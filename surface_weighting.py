import pandas as pd
import numpy as np

# Read the csv file
df = pd.read_csv('_Data/Original_dataset/preprocessed_data.csv', header=0, index_col=0)
df.reset_index(drop=True,inplace=True)

# Set tables reading options
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

# Get the name of all players in this dataset
playersA_df = pd.DataFrame(data=df.PlayerA_name.unique(), columns=['Name'])
playersB_df = pd.DataFrame(data=df.PlayerB_name.unique(), columns=['Name'])
players = pd.concat([playersA_df, playersB_df], ignore_index=True)
players.drop_duplicates(inplace=True)
players.reset_index(drop=True, inplace=True)

# For each player, compute its percentage of matches won on each surface
players['%_won_carpet'] = 0.0
players['%_won_clay'] = 0.0
players['%_won_grass'] = 0.0
players['%_won_hard'] = 0.0

cols = list(range(36,40))
for i, player in players.iterrows():
    # Get all the matches of the player
    name = player['Name']
    playerA_rows = df.index[(df['PlayerA_name'] == name)].tolist()
    playerB_rows = df.index[(df['PlayerB_name'] == name)].tolist()
    playerA_df = df.iloc[playerA_rows, cols]
    playerA_df['Win'] = 1
    playerB_df = df.iloc[playerB_rows, cols]
    playerB_df['Win'] = 0
    matches_df = pd.concat([playerA_df, playerB_df], ignore_index=True)

    # Compute mean percentage for Carpet
    carpet_df = matches_df[matches_df['surface_Carpet']==1]
    players.at[i,'%_won_carpet'] = carpet_df['Win'].mean()

    # Compute mean percentage for Clay
    clay_df = matches_df[matches_df['surface_Clay']==1]
    players.at[i,'%_won_clay'] = clay_df['Win'].mean()

    # Compute mean percentage for Grass
    grass_df = matches_df[matches_df['surface_Grass']==1]
    players.at[i,'%_won_grass'] = grass_df['Win'].mean()

    # Compute mean percentage for Hard
    hard_df = matches_df[matches_df['surface_Hard']==1]
    players.at[i,'%_won_hard'] = hard_df['Win'].mean()

# Drop the missing values
players.dropna(inplace=True)
players.reset_index(drop=True,inplace=True)

# Mean and std of percentage of matches won on Carpet
mean_carpet = players['%_won_carpet'].mean()
std_carpet = players['%_won_carpet'].std()

# Mean and std of percentage of matches won on Clay
mean_clay = players['%_won_clay'].mean()
std_clay = players['%_won_clay'].std()

# Mean and std of percentage of matches won on Grass
mean_grass = players['%_won_grass'].mean()
std_grass = players['%_won_grass'].std()

# Mean and std of percentage of matches won on Hard
mean_hard = players['%_won_hard'].mean()
std_hard = players['%_won_hard'].std()

# Create new dataframe to contain the correlation coefficients
index = ['Clay', 'Carpet', 'Grass', 'Hard']
columns = ['Clay', 'Carpet', 'Grass', 'Hard']
corr_df = pd.DataFrame(index=index, columns=columns)

# Compute the correlation between Clay and other surfaces
X = (players['%_won_clay'] - mean_clay) * (players['%_won_carpet'] - mean_carpet)
corr_df.at['Clay', 'Carpet'] = X.sum()/(len(X) * std_clay * std_carpet)
X = (players['%_won_clay'] - mean_clay) * (players['%_won_grass'] - mean_grass)
corr_df.at['Clay', 'Grass'] = X.sum()/(len(X) * std_clay * std_grass)
X = (players['%_won_clay'] - mean_clay) * (players['%_won_hard'] - mean_hard)
corr_df.at['Clay', 'Hard'] = X.sum()/(len(X) * std_clay * std_hard)

# Compute the correlation between Hard and other surfaces
X = (players['%_won_hard'] - mean_hard) * (players['%_won_carpet'] - mean_carpet)
corr_df.at['Hard', 'Carpet'] = X.sum()/(len(X) * std_hard * std_carpet)
X = (players['%_won_hard'] - mean_hard) * (players['%_won_grass'] - mean_grass)
corr_df.at['Hard', 'Grass'] = X.sum()/(len(X) * std_hard * std_grass)
X = (players['%_won_hard'] - mean_hard) * (players['%_won_clay'] - mean_clay)
corr_df.at['Hard', 'Clay'] = X.sum()/(len(X) * std_hard * std_clay)

# Compute the correlation between Grass and other surfaces
X = (players['%_won_grass'] - mean_grass) * (players['%_won_carpet'] - mean_carpet)
corr_df.at['Grass', 'Carpet'] = X.sum()/(len(X) * std_grass * std_carpet)
X = (players['%_won_grass'] - mean_grass) * (players['%_won_hard'] - mean_hard)
corr_df.at['Grass', 'Hard'] = X.sum()/(len(X) * std_grass * std_hard)
X = (players['%_won_grass'] - mean_grass) * (players['%_won_clay'] - mean_clay)
corr_df.at['Grass', 'Clay'] = X.sum()/(len(X) * std_grass * std_clay)

# Compute the correlation between Carpet and other surfaces
X = (players['%_won_carpet'] - mean_carpet) * (players['%_won_grass'] - mean_grass)
corr_df.at['Carpet', 'Grass'] = X.sum()/(len(X) * std_carpet * std_grass)
X = (players['%_won_carpet'] - mean_carpet) * (players['%_won_hard'] - mean_hard)
corr_df.at['Carpet', 'Hard'] = X.sum()/(len(X) * std_carpet * std_hard)
X = (players['%_won_carpet'] - mean_carpet) * (players['%_won_clay'] - mean_clay)
corr_df.at['Carpet', 'Clay'] = X.sum()/(len(X) * std_carpet * std_clay)

corr_df.fillna(1, inplace=True)

# Save table
corr_df.to_csv('_Data/New_stats_dataset/correlation_between_surfaces.csv', sep=',', float_format='%.3f', decimal='.')
