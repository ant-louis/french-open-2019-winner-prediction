import pandas as pd
import numpy as np


# Read the excel file containing 50.000 matches from 2000 to 2018
df = pd.read_excel('Original_data/2000-2018.xls')

# Replace all NaN with 0
df.fillna(0, inplace=True)

# Drop useless columns
to_drop = ['ATP', 'Location', 'Tournament']
df.drop(columns=to_drop, inplace=True)

# Rename columns with clearer names
to_rename={'Best of' : 'Nb sets max',
          'WRank' : 'Winner ranking',
          'LRank' : 'Loser ranking',
          'WPts' : 'Winner points',
          'LPts' : 'Loser points',
          'W1' : 'Games won by winner in set 1',
          'W2' : 'Games won by winner in set 2',
          'W3' : 'Games won by winner in set 3',
          'W4' : 'Games won by winner in set 4',
          'W5' : 'Games won by winner in set 5',
          'L1' : 'Games won by loser in set 1',
          'L2' : 'Games won by loser in set 2',
          'L3' : 'Games won by loser in set 3',
          'L4' : 'Games won by loser in set 4',
          'L5' : 'Games won by loser in set 5',
          'Wsets' : 'Winner sets',
          'Lsets' : 'Loser sets',
           'Comment' : 'Completed or retired'
          }
df.rename(columns=to_rename, inplace=True)

# Split the date
new = df["Date"].astype(str).str.split("-", n = 2, expand = True)
df["Year"] = new[0]
df["Month"] = new[1]
df.drop(columns=["Date"], inplace = True)

# Rearrange columns
cols = df.columns.tolist()
cols = cols[-2:] + cols[:-2]
df = df[cols]

# Export to csv
df.to_csv('2000_2018_cleaned.csv', sep=',', encoding='utf-8', float_format='%.0f')