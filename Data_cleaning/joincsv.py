import pandas as pd 

matches = pd.read_csv('cleaned_matches_data.csv', dtype=object, encoding='utf-8')
players = pd.read_csv('cleaned_players_data.csv', dtype=object, encoding='utf-8')
matches.drop(matches.columns[0], axis=1, inplace=True)
players.drop(players.columns[0], axis=1, inplace=True)


players_and_matches = pd.merge(matches,players,left_on="Winner",right_on="Name",suffixes=['_Match','_Winner'])
players_and_matches = pd.merge(players_and_matches,players,left_on="Loser",right_on="Name",suffixes=['_Winner','_Loser'])

players_and_matches.to_csv("merged_matches_players.csv", index=False)
