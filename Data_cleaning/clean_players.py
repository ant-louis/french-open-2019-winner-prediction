import glob, os
import pandas as pd
import numpy as np



# Concatenate all csv files in one
path =r'Data/scrapedPlayerInfo'
all_files = glob.glob(os.path.join(path, "*.csv"))
files_df = (pd.read_csv(file, header=0) for file in all_files)
df = pd.concat(files_df, ignore_index=True)

# Replace all NaN with 0
df.fillna(0, inplace=True)

# Drop useless columns
to_drop = ['Active',
          'Backhand',
          'Best of 3', 'Best of 3: 0:0', 'Best of 3: 0:1', 'Best of 3: 0:2', 'Best of 3: 0:3', 'Best of 3: 1:0', 'Best of 3: 1:1', 'Best of 3: 1:2', 'Best of 3: 1:3', 'Best of 3: 2:0', 'Best of 3: 2:1', 'Best of 3: 2:3', 'Best of 3: 3:0', 'Best of 3: 3:1', 'Best of 3: 3:2',
          'Birthplace',
          'Bronze Medal',
          'Coach',
          'Current Elo Rank',
          'Davis Cup',
          'Facebook',
          'For Bronze Medal',
          'GOAT Rank',
           'Height',
          'Nicknames',
          'Olympics',
          'Opponent Elo Rating',
          'Peak Elo Rating',
          'Plays',
          'Prize Money',
          'Residence',
          'Retired',
           'Turned Pro',
          'Twitter',
          'Web Site',
          'Weeks at No. 1',
          'Weight',
          'Wikipedia',
          'World Team Cup']
df.drop(columns=to_drop, inplace=True)

# Reorder columns
cols = ['First Name', 'Last Name', 'Country', 'Current Rank', 'Age',
'Favorite Surface', 'Best Rank', 'Best Season', 'Last Appearance', 'Seasons', 'Retired', 'Prize Money', 'Win',
'ATP 250', 'ATP 500', 'Masters', 'Tour Finals', 'Grand Slam', 'Grand Slams',
'Round of 128', 'Round of 64', 'Round of 32', 'Round of 16', 'Quarter-Final', 'Semi-Final', 'Final',
'Vs No. 1', 'Vs Top 5', 'Vs Top 10', 'Vs Top 20', 'Vs Top 50', 'Vs Top 100',
'Matches Played', 'Matches Won %', 'Matches Won', 'After Losing 1st Set', 'After Winning 1st Set',
'Sets Played', 'Sets Won %', 'Sets Won', 'Sets per Match', 'Sets to Matches Ov.-Perf.', 'Deciding Set',
'Games per Match', 'Games per Set', 'Games Won %', 'Games Dominance', 'Service Games Won %', 'Rtn. Gms. Won per Match', 'Rtn. Gms. Won per Set', 'Return Games Won %', 'Total Games Played', 'Total Games Won', 'Gms. to Matches Ov.-Perf.', 'Gms. to Sets Over-Perf.', 'Svc. Gms. Lost per Match', 'Svc. Gms. Lost per Set',
'Points per Match', 'Points per Set', 'Points per Game', 'Points per Return Game', 'Points per Service Game', 'Points Dominance', 'Service Points Won %', 'Rtn. In-play Pts. Won %', 'Rtn. to Svc. Points Ratio', 'S. Pts. to S. Gms. Ov.-Perf.', 'Pts. Lost per Svc. Game', 'Pts. Won per Rtn. Game', 'Pts. to Gms. Over-Perf.', 'Pts. to Matches Over-Perf.', 'Pts. to Sets Over-Perf.', 'Pts. to TBs. Over-Perf.', 'R. Pts. to R. Gms. Ov.-Perf.', 'Return Points Won %', 'Svc. In-play Pts. Won %', 'Total Points Played', 'Total Points Won', 'Total Points Won %',
'Best of 5', 'Best of 5: 0:0', 'Best of 5: 0:1', 'Best of 5: 0:2', 'Best of 5: 0:3', 'Best of 5: 0:4', 'Best of 5: 1:0', 'Best of 5: 1:2', 'Best of 5: 1:3', 'Best of 5: 1:4', 'Best of 5: 2:0', 'Best of 5: 2:1', 'Best of 5: 2:2', 'Fifth Set',
'Match Time', 'Set Time (minutes)', 'Game Time (minutes)', 'Point Time (seconds)',
'Ace %', 'Aces', 'Aces per Match', 'Aces per Set', 'Aces per Svc. Game', 'Aces / DFs Ratio', 'Ace Against %', 'Ace Against',
'1st Serve %', '1st Serve Won %', '1st Srv. Return Won %', '2nd Serve Won %', '2nd Srv. Return Won %',
'Double Fault %', 'Double Faults', 'Double Fault Against %', 'Double Faults Against', 'DFs per Match', 'DFs per Set', 'DFs per Svc. Game', 'DFs per 2nd Serve %',
'BPs per Match', 'BPs Faced per Match', 'BPs per Set', 'BPs Faced per Set', 'BPs per Svc. Game', 'BPs per Return Game', 'Break Points Won %', 'Break Points Saved %', 'Break Points Ratio', 'BPs Conv. Over-Perf.', 'BPs Over-Performing', 'BPs Saved Over-Perf.',
'Tie Breaks', 'Tie Breaks Played', 'Tie Breaks Won', 'Tie Breaks Won %', 'Tie Breaks per Match', 'Tie Breaks per Set %', 'Deciding Set Tie Breaks',
'Upsets', 'Upsets %', 'Upsets scored', 'Upsets scored %', 'Upsets against', 'Upsets against %',
'Very Slow', 'Slow', 'Medium', 'Medium Slow', 'Medium Fast', 'Fast', 'Very Fast',
'Carpet', 'Clay', 'Grass', 'Indoor', 'Outdoor', 'Hard',
'H2H %', 'Titles', 'Round-Robin', 'Alt. Finals', 'Opponent Rank']
df = df[cols]

# Get rid off retired players
df.rename(columns={'Current Rank' : 'Ranking'}, inplace=True)
df = df[df.Ranking != 0]

# Sort rows by ranking and reset index
df.sort_values(by=['Ranking'], inplace = True)
df.reset_index(drop=True, inplace=True)

# Export to csv
df.to_csv('players_data.csv', sep=',', encoding='utf-8')