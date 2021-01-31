import pandas as pd

matches_2019 = pd.read_csv("../../data/scraping/matches_2019.csv")
matches_2018 = pd.read_csv("../../data/scraping/matches_2018.csv")
tournaments_2018 = pd.read_csv('../../data/scraping/tournaments_2018-2018.csv')
tournaments_2019 = pd.read_csv('../../data/scraping/tournaments_2019-2019.csv')
clean_data = pd.read_csv('../../data/original_dataset/cleaned_data.csv')


# Megrge with tournament
clean_matches_2019 = pd.merge(matches_2019, tournaments_2019, left_on='tourney_order', right_on='tourney_order')
clean_matches_2018 = pd.merge(matches_2018, tournaments_2018, left_on='tourney_order', right_on='tourney_order')

# Correct wrong year for a given date (31 december 2018 is labeled as 2019)
wrong_date = clean_matches_2019['tourney_dates'] == '2018.12.31'
clean_matches_2019.loc[wrong_date, 'tourney_year'] = 2018

# Use month to convert to day
clean_matches_2019['tourney_day'] = clean_matches_2019['tourney_day'].astype(int)
clean_matches_2019['tourney_day'] = (clean_matches_2019['tourney_month'] -1)*30 + clean_matches_2019['tourney_day']
clean_matches_2018['tourney_day'] = clean_matches_2018['tourney_day'].astype(int)
clean_matches_2018['tourney_day'] = (clean_matches_2018['tourney_month'] -1)*30 + clean_matches_2018['tourney_day']

# Keep only relevant columns
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

clean_matches_2019 = clean_matches_2019[matches_col_to_keep]
clean_matches_2018 = clean_matches_2018[matches_col_to_keep]

# From 2018, only use matches after day 257 (we have the rest already)
after_257_days = clean_matches_2018['tourney_day'] > 257
clean_matches_2018.where(after_257_days, inplace=True)
clean_matches_2018.dropna(inplace=True)

# Concatenate 2018 after 257 days and  2019 matches into one
clean_matches = pd.concat([clean_matches_2018, clean_matches_2019])


# Apply 'Title Case' string format to names
clean_matches['winner_name'] = clean_matches['winner_name'].apply(lambda x : x.title())
clean_matches['losers_name'] = clean_matches['losers_name'].apply(lambda x : x.title())
# Replace - with spaces in names
clean_matches['winner_name'] = clean_matches['winner_name'].str.replace('-', ' ')
clean_matches['losers_name'] = clean_matches['losers_name'].str.replace('-', ' ')


# Rename columns to fit with already collected data
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

clean_matches.rename(columns= to_rename, inplace=True)

# Create empty columns for data we don't have
clean_matches['round'] = 0
clean_matches['best_of'] = 3

clean_matches['PlayerA_rank'] = 0
clean_matches['PlayerA_rank_points'] = 0
clean_matches['PlayerB_rank'] = 0
clean_matches['PlayerB_rank_points'] = 0

clean_matches['PlayerA_id'] = 0
clean_matches['PlayerB_id'] = 0
clean_matches['PlayerA_FR'] = 0
clean_matches['PlayerB_FR'] = 0
clean_matches['PlayerA_height'] = 0
clean_matches['PlayerB_height'] = 0

clean_matches['surface_Clay'] = 0
clean_matches['surface_Carpet'] = 0
clean_matches['surface_Grass'] = 0
clean_matches['surface_Hard'] = 0

clean_matches['PlayerA_Win'] = 1


# Best of of 5 if draw_size is 128 
for index, row in clean_matches.iterrows():
    if row['draw_size'] == 128:
        clean_matches.loc[index,'best_of'] = 5

# One hot encore the surface manually
for index, row in clean_matches.iterrows():
    if row['tourney_surface'] == 'Clay':
        clean_matches.loc[index,'surface_Clay'] = 1
    elif row['tourney_surface'] == 'Carpet':
        clean_matches.loc[index,'surface_Carpet'] = 1
    elif row['tourney_surface'] == 'Grass':
        clean_matches.loc[index,'surface_Grass'] = 1
    elif row['tourney_surface'] == 'Hard':
        clean_matches.loc[index,'surface_Hard'] = 1

clean_matches.drop(columns= ['tourney_surface'], inplace=True)

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
print("Clean matches shape before merging :",clean_matches.shape)


# Get unique player names in the 2019 matches
matches_player_names_A = clean_matches.drop_duplicates("PlayerA_name", keep='last')['PlayerA_name']
matches_player_names_A.rename(columns={'PlayerA_name':'Name'}, inplace=True)
matches_player_names_B = clean_matches.drop_duplicates("PlayerB_name", keep='last')['PlayerB_name']
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
clean_matches.replace(to_replace=name_dict, inplace=True)

# Adding age of the players as a feature by merging clean_matches with players_age and righthandedness
clean_matches = pd.merge(clean_matches, player_ages, left_on="PlayerA_name", right_on="Name")
clean_matches = pd.merge(clean_matches, player_ages, left_on="PlayerB_name", right_on="Name")
clean_matches = pd.merge(clean_matches, player_righthandedness, left_on="PlayerA_name", right_on="Name")
clean_matches = pd.merge(clean_matches, player_righthandedness, left_on="PlayerB_name", right_on="Name")
to_rename = {
    'age_x':'PlayerA_age', 
    'age_y': 'PlayerB_age',
    'righthanded_x': 'PlayerA_righthanded',
    'righthanded_y': 'PlayerB_righthanded',
    }
clean_matches.rename(columns=to_rename, inplace=True)
clean_matches.drop(columns=['Name_x','Name_y'], inplace=True)

print("Clean matches shape after merging :",clean_matches.shape)

# Rearrange columns in the desired order
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
clean_matches = clean_matches[to_rearrange]

# Convert numerical columns to float and sort per day and year
clean_matches = clean_matches.astype('float', errors='ignore')
clean_matches.sort_values(by=['Year', 'Day'], inplace=True)

# Save the new matches to csv
clean_matches.to_csv("../Clean data/clean_matches_2018_2019.csv", index=False)

# Concat the new matches with the already collected data and save to csv
clean_data = pd.concat([clean_data,clean_matches], sort=False)
clean_data.drop(columns=['Unnamed: 0'], inplace=True)
print("Merging 1993-2018 with 2019 shape:" ,clean_data.shape)
clean_data.to_csv('../../data/original_dataset/cleaned_with_2019_data.csv')
