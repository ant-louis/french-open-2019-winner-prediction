import pandas as pd

df = pd.read_csv('all_games.csv',sep=',')

# Keep only matches between the best 
# 32 players at the moment
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
            'winner_name',
            'loser_name',
            'winner_ioc',
            'loser_ioc']

df.drop(columns=to_drop, inplace = True)

# Omit matches where a stat is missing
df = df.dropna()

# Convert into categorical data
df = pd.get_dummies(df)

# Split the date
df["tourney_date"] = df["tourney_date"].astype(str).str.slice(stop=4)

# Save dataset
df.to_csv('cleaned_data.csv', sep=',', encoding='utf-8', float_format='%.0f', decimal='.')
