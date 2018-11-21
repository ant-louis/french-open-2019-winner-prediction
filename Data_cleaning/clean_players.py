import glob, os
import pandas as pd
import numpy as np


# Concatenate all csv files in one
path =r'Original_data/scrapedPlayerInfo'
all_files = glob.glob(os.path.join(path, "*.csv"))
files_df = (pd.read_csv(file, header=0) for file in all_files)
df = pd.concat(files_df, ignore_index=True)

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
'Favorite Surface', 'Best Rank', 'Best Season', 'Last Appearance', 'Seasons', 'Win',
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
'H2H %', 'Opponent Rank','Alt. Finals', 'Round-Robin', 'Titles']
df = df[cols]

# Rename columns with percentages
to_rename={'Win' : 'Win %',
           'ATP 250' : 'ATP 250 %',
           'ATP 500' : 'ATP 500 %', 
           'Masters' : 'Masters %', 
           'Tour Finals' : 'Tour Finals %', 
           'Grand Slam' : 'Grand Slam %',
           'Round of 128' :'Round of 128 %',
           'Round of 64':'Round of 64 %', 
           'Round of 32':'Round of 32 %', 
           'Round of 16':'Round of 16 %',
           'Quarter-Final':'Quarter-Final %', 
           'Semi-Final':'Semi-Final %', 
           'Final':'Final %',
           'Vs No. 1':'Vs No. 1 %', 
           'Vs Top 5':'Vs Top 5 %', 
           'Vs Top 10':'Vs Top 10 %', 
           'Vs Top 20':'Vs Top 20 %', 
           'Vs Top 50':'Vs Top 50 %', 
           'Vs Top 100':'Vs Top 100 %',
           'Deciding Set':'Deciding Set %',
           'Best of 5':'Best of 5 %', 
           'Best of 5: 0:0':'Best of 5: 0:0 %', 
           'Best of 5: 0:1':'Best of 5: 0:1 %', 
           'Best of 5: 0:2':'Best of 5: 0:2 %', 
           'Best of 5: 0:3':'Best of 5: 0:3 %', 
           'Best of 5: 0:4':'Best of 5: 0:4 %', 
           'Best of 5: 1:0':'Best of 5: 1:0 %', 
           'Best of 5: 1:2':'Best of 5: 1:2 %', 
           'Best of 5: 1:3':'Best of 5: 1:3 %', 
           'Best of 5: 1:4':'Best of 5: 1:4 %', 
           'Best of 5: 2:0':'Best of 5: 2:0 %', 
           'Best of 5: 2:1':'Best of 5: 2:1 %', 
           'Best of 5: 2:2':'Best of 5: 2:2 %', 
           'Fifth Set':'Fifth Set %',
           'Very Slow':'Very Slow %', 
           'Slow':'Slow %', 
           'Medium':'Medium %', 
           'Medium Slow':'Medium Slow %', 
           'Medium Fast':'Medium Fast %', 
           'Fast':'Fast %', 
           'Very Fast':'Very Fast %',
           'Carpet':'Carpet %', 
           'Clay':'Clay %', 
           'Grass':'Grass %', 
           'Indoor':'Indoor %', 
           'Outdoor':'Outdoor %', 
           'Hard':'Hard %',
          }
df.rename(columns=to_rename, inplace=True)

# Take all % columns and convert % in number between 0 and 1
percentage_cols = [col for col in df.columns if '%' in col]
for col in percentage_cols:
    df[col] = df[col].str.rstrip('%').astype('float') / 100.0

# Get rid off retired players
df = df[np.isfinite(df['Current Rank'])]

# Sort rows by ranking and reset index
df.sort_values(by=['Current Rank'], inplace = True)

# Rearrange names in the same way that matches_data.csv
df['First Name'] = df['First Name'].astype(str).str[0] + '.'
df['Name']= df['Last Name'] + ' ' + df['First Name']
df.drop(columns=['First Name', 'Last Name'], inplace=True)
cols = df.columns.tolist()
cols = cols[-1:] + cols[:-1]
df = df[cols]

# Count the number of NaN in a column
for col in df:
    num_nan = df.loc[(pd.isna(df[col])), col].shape[0]
    print(f"There are {num_nan} NaNs in column {col}")

# Drop columns with too many NaN
to_drop = ['Alt. Finals',
           'Round-Robin',
           'Titles',
           'Best Season',
           'Last Appearance',
           'Win %',
           'Grand Slams',
           'Best of 5: 0:0 %',
           'Best of 5: 0:1 %',
           'Best of 5: 0:2 %',
           'Best of 5: 0:4 %',
           'Best of 5: 1:0 %',
           'Best of 5: 1:2 %',
           'Best of 5: 1:4 %',
           'Best of 5: 2:0 %',
           'Best of 5: 2:1 %',
           'Best of 5: 2:2 %',
           'Deciding Set Tie Breaks',
           'H2H %',
           'Tour Finals %',
           'Carpet %',
           'Grass %',
           'Very Slow %',
           'Slow %',
           'Medium %',
           'Medium Slow %',
           'Medium Fast %',
           'Fast %',
           'Very Fast %',
           'Indoor %',
           'Hard %',
           'Tie Breaks',
           'Best of 5 %',
           'Best of 5: 0:3 %',
           'Best of 5: 1:3 %',
           'Fifth Set %'
          ]
df.drop(columns=to_drop, inplace=True)

# Drop rows with too many missing values
df = df[np.isfinite(df['Points per Set'])]

# Reset the indexes
df.reset_index(drop=True, inplace=True)

# Replace NaN values
df['Favorite Surface'].fillna('None', inplace=True)
df.fillna(0, inplace=True)

# Export to csv
df.to_csv('cleaned_players_data.csv', sep=',', encoding='utf-8', float_format='%.3f', decimal='.')
