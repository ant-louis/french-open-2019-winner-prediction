import pandas as pd

df = pd.read_csv('all_games.csv',sep=',')

# Keep only matches between the best 
# 32 players at the moment
cond1 = df['winner_rank'] < 33
df = df[cond1]

cond2 = df['loser_rank'] < 33
df = df[cond2]

# Drop columns
to_drop = ['winner_seed',
            'winner_entry',
            'loser_seed',
            'loser_entry',
            'match_num',
            'score',]

df.drop(columns=to_drop, inplace = True)

# Omit matches where a stat is missing
df = df.dropna()

# Convert into categorical data
df = pd.get_dummies(df)

# Save dataset
df.to_csv('cleaned_data.csv', sep=',', encoding='utf-8', float_format='%.0f', decimal='.')
