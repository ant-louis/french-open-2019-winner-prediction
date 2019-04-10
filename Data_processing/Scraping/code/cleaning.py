import pandas as pd




matches_2019 = pd.read_csv("../Original data/matches_2019.csv")
matches_2018 = pd.read_csv("../Original data/matches_2018.csv")
tournaments_2018 = pd.read_csv('../Original data/tournaments_2018-2018.csv')
tournaments_2019 = pd.read_csv('../Original data/tournaments_2019-2019.csv')


matches_col_to_keep = [
    'tourney_year',
    'tourney_month',
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


clean_matches_2019 = pd.merge(matches_2019, tournaments_2019, left_on='tourney_order', right_on='tourney_order')
clean_matches_2019 = clean_matches_2019[matches_col_to_keep]
clean_matches_2019.to_csv("../Clean data/clean_matches_2019.csv", index=False)

clean_matches_2018 = pd.merge(matches_2018, tournaments_2018, left_on='tourney_order', right_on='tourney_order')
clean_matches_2018 = clean_matches_2018[matches_col_to_keep]
clean_matches_2018.to_csv("../Clean data/clean_matches_2018.csv", index=False)
# matches_2019['winner_1st_serve%'] = matches_2019['winner_first_serves_total'] / matches_2019['winner_total_points_total']
# matches_2019['loser_1st_serve%'] = matches_2019['loser_first_serves_total'] / matches_2019['loser_total_points_total']

# matches_2019['winner_1st_serve_won%'] = matches_2019['winner_first_serve_points_won'] / matches_2019['winner_first_serve_points_total']
# matches_2019['loser_1st_serve_won%'] = matches_2019['loser_first_serve_points_won'] / matches_2019['loser_first_serve_points_total']

# matches_2019['winner_2nd_serve_won%'] = matches_2019['winner_second_serve_points_won'] / matches_2019['winner_second_serve_points_total']
# matches_2019['loser_2nd_serve_won%'] = matches_2019['loser_second_serve_points_won'] / matches_2019['loser_second_serve_points_total']

# matches_2019['winner_ace%'] = matches_2019['winner_aces'] / matches_2019['winner_total_points_total']
# matches_2019['winner_df%'] = matches_2019['winner_double_faults'] / matches_2019['winner_total_points_total']

# matches_2019['loser_ace%'] = matches_2019['loser_aces'] / matches_2019['loser_total_points_total']
# matches_2019['loser_df%'] = matches_2019['loser_double_faults'] / matches_2019['loser_total_points_total']
