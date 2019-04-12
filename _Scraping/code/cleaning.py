import pandas as pd

matches_2019 = pd.read_csv("../Original data/matches_2019.csv")
matches_2018 = pd.read_csv("../Original data/matches_2018.csv")
tournaments_2018 = pd.read_csv('../Original data/tournaments_2018-2018.csv')
tournaments_2019 = pd.read_csv('../Original data/tournaments_2019-2019.csv')
clean_data = pd.read_csv('../../_Data/Original_dataset/cleaned_data.csv')


matches_col_to_keep = [
    'tourney_year',
    'tourney_day',
    'tourney_surface',
    'tourney_singles_draw',
    'winner_name',
    'losers_name',
    'match_duration',

    'winner_aces',
    'winner_double_faults',
    'winner_service_points_total',
    'winner_first_serves_in',
    'winner_first_serve_points_won',
    'winner_second_serve_points_won',
    'winner_service_games_played',
    'winner_break_points_saved',
    'winner_break_points_serve_total',
    'loser_aces',

    'loser_double_faults',
    'loser_service_points_total',
    'loser_first_serves_in',
    'loser_first_serve_points_won',
    'loser_second_serve_points_won',
    'loser_service_games_played',
    'loser_break_points_saved',
    'loser_break_points_serve_total'
]

# Mege with tournament and keep only relevant columns
clean_matches_2019 = pd.merge(matches_2019, tournaments_2019, left_on='tourney_order', right_on='tourney_order')
clean_matches_2019 = clean_matches_2019[matches_col_to_keep]

# Apply 'Title Case' string format to names
clean_matches_2019['winner_name'] = clean_matches_2019['winner_name'].apply(lambda x : x.title())
clean_matches_2019['losers_name'] = clean_matches_2019['losers_name'].apply(lambda x : x.title())
# Replace - with spaces in names
clean_matches_2019['winner_name'] = clean_matches_2019['winner_name'].str.replace('-', ' ')
clean_matches_2019['losers_name'] = clean_matches_2019['losers_name'].str.replace('-', ' ')



to_rename = {
    'tourney_year': 'Year',
    'tourney_day':'Day',
    'tourney_singles_draw':'draw_size',
    'winner_name':'PlayerA_name',
    'losers_name':'PlayerB_name',
    'match_duration':'minutes',

    'winner_aces':'PlayerA_ace',
    'winner_double_faults':'PlayerA_df',
    'winner_service_points_total':'PlayerA_svpt',
    'winner_first_serves_in':'PlayerA_1stIn',
    'winner_first_serve_points_won':'PlayerA_1stWon',
    'winner_second_serve_points_won':'PlayerA_2ndWon',
    'winner_service_games_played':'PlayerA_SvGms',
    'winner_break_points_saved':'PlayerA_bpSaved',
    'winner_break_points_serve_total':'PlayerA_bpFaced',
    
    'loser_aces':'PlayerB_ace',
    'loser_double_faults':'PlayerB_df',
    'loser_service_points_total':'PlayerB_svpt',
    'loser_first_serves_in':'PlayerB_1stIn',
    'loser_first_serve_points_won':'PlayerB_1stWon',
    'loser_second_serve_points_won':'PlayerB_2ndWon',
    'loser_service_games_played':'PlayerB_SvGms',
    'loser_break_points_saved':'PlayerB_bpSaved',
    'loser_break_points_serve_total':'PlayerB_bpFaced'    
}


clean_matches_2019.rename(columns= to_rename, inplace=True)

clean_matches_2019['round'] = 0
clean_matches_2019['best_of'] = 3

clean_matches_2019['PlayerA_rank'] = 0
clean_matches_2019['PlayerA_rank_points'] = 0
clean_matches_2019['PlayerB_rank'] = 0
clean_matches_2019['PlayerB_rank_points'] = 0

clean_matches_2019['PlayerA_id'] = 0
clean_matches_2019['PlayerB_id'] = 0
clean_matches_2019['PlayerA_FR'] = 0
clean_matches_2019['PlayerB_FR'] = 0
clean_matches_2019['PlayerA_height'] = 0
clean_matches_2019['PlayerB_height'] = 0

clean_matches_2019['surface_Clay'] = 0
clean_matches_2019['surface_Carpet'] = 0
clean_matches_2019['surface_Grass'] = 0
clean_matches_2019['surface_Hard'] = 0

clean_matches_2019['PlayerA_Win'] = 1

# Best of of 5 if draw_size is 128 
for index, row in clean_matches_2019.iterrows():
    if row['draw_size'] == 128:
        clean_matches_2019.loc[index,'best_of'] = 5

# One hot encore the surface manually
for index, row in clean_matches_2019.iterrows():
    if row['tourney_surface'] == 'Clay':
        clean_matches_2019.loc[index,'surface_Clay'] = 1
    elif row['tourney_surface'] == 'Carpet':
        clean_matches_2019.loc[index,'surface_Carpet'] = 1
    elif row['tourney_surface'] == 'Grass':
        clean_matches_2019.loc[index,'surface_Grass'] = 1
    elif row['tourney_surface'] == 'Hard':
        clean_matches_2019.loc[index,'surface_Hard'] = 1

clean_matches_2019.drop(columns= ['tourney_surface'], inplace=True)

# Extract only relevant information
player_A_data = clean_data[['PlayerA_name', 'PlayerA_age','PlayerA_righthanded', 'Year', 'Day']]
player_B_data = clean_data[['PlayerB_name', 'PlayerB_age','PlayerB_righthanded', 'Year', 'Day']]

#Rename columns so that the players can be stacked, regardless if A or B
to_rename_A = {'PlayerA_name':'Name', 'PlayerA_age' : 'age', 'PlayerA_righthanded':'righthanded'}
to_rename_B = {'PlayerB_name':'Name', 'PlayerB_age' : 'age', 'PlayerB_righthanded':'righthanded'}
player_A_data.rename(columns=to_rename_A, inplace=True)
player_B_data.rename(columns=to_rename_B, inplace=True)
player_data = pd.concat([player_A_data, player_B_data])

# Sort them and take the most recent match they played to get their age
player_data.sort_values(by=['Year', 'Day'], inplace=True, ascending=False)
player_data.drop(columns=['Year', 'Day'], inplace=True)
player_data = player_data.drop_duplicates('Name', keep='first')

player_ages = player_data[['Name','age']]
player_righthandedness = player_data[['Name','righthanded']]
print("Unique player names in 1993-2018: ", player_data.shape)
print("Clean matches shape before merging :",clean_matches_2019.shape)


# Get unique player names in the 2019 matches
matches_player_names_A = clean_matches_2019.drop_duplicates("PlayerA_name", keep='last')['PlayerA_name']
matches_player_names_A.rename(columns={'PlayerA_name':'Name'}, inplace=True)
matches_player_names_B = clean_matches_2019.drop_duplicates("PlayerB_name", keep='last')['PlayerB_name']
matches_player_names_B.rename(columns={'PlayerB_name':'Name'}, inplace=True)
matches_player_names = pd.concat([matches_player_names_A, matches_player_names_B])
matches_player_names = matches_player_names.drop_duplicates(keep='first')

# Use the names of the players already in our dataset to
# replace the names of the players in the incoming 2019 dataset
name_dict = {}
i = 0
total_length = len(player_ages['Name'].values)
for player_name in player_ages['Name'].values:
    i += 1
    split = player_name.split()
    player_fn = split[0]
    player_ln = split[-1]
    for match_player_name in matches_player_names.values:
        match_split = match_player_name.split()
        match_player_fn = match_split[0]
        match_player_ln = match_split[-1]
        
        # Create dictionnary of the names to replace
        # Compare only the first and last names in the so that the
        # missing or superfluous middle names are added/deleted
        if (player_ln == match_player_ln and player_fn == match_player_fn):
            name_dict[match_player_name] = player_name
clean_matches_2019.replace(to_replace=name_dict, inplace=True)

# Adding age of the players as a feature by merging clean_matches_2019 with players_age and righthandedness
clean_matches_2019 = pd.merge(clean_matches_2019, player_ages, left_on="PlayerA_name", right_on="Name")
clean_matches_2019 = pd.merge(clean_matches_2019, player_ages, left_on="PlayerB_name", right_on="Name")
clean_matches_2019 = pd.merge(clean_matches_2019, player_righthandedness, left_on="PlayerA_name", right_on="Name")
clean_matches_2019 = pd.merge(clean_matches_2019, player_righthandedness, left_on="PlayerB_name", right_on="Name")
to_rename = {
    'age_x':'PlayerA_age', 
    'age_y': 'PlayerB_age',
    'righthanded_x': 'PlayerA_righthanded',
    'righthanded_y': 'PlayerB_righthanded',
    }
clean_matches_2019.rename(columns=to_rename, inplace=True)
clean_matches_2019.drop(columns=['Name_x','Name_y'], inplace=True)

print("Clean matches shape after merging :",clean_matches_2019.shape)

to_rearrange = [
    'PlayerA_name',
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
    'surface_Hard'
]

clean_matches_2019 = clean_matches_2019[to_rearrange]
# numeric_cols = clean_matches_2019.columns.difference(['PlayerA_name', 'PlayerB_name'])
# clean_matches_2019[numeric_cols] = clean_matches_2019[numeric_cols].apply(pd.to_numeric)
clean_matches_2019 = clean_matches_2019.astype('float', errors='ignore')
clean_matches_2019.to_csv("../Clean data/clean_matches_2019.csv", index=False)

clean_data = pd.concat([clean_data,clean_matches_2019], sort=False)
clean_data.drop(columns=['Unnamed: 0'], inplace=True)
print("Merging 1993-2018 with 2019 shape:" ,clean_data.shape)
clean_data.to_csv('../../_Data/Original_dataset/cleaned_with_2019_data.csv')
