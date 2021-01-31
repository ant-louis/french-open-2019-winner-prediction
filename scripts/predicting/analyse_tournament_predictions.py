import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')


# Current year
year = 2019

# Files
players_names = "../../data/predictions/stats_players_{}.csv".format(year)
predicted_probas = "../../data/predictions/players_rounds_predictions_{}.csv".format(year)

# Read the csv files
players_df = pd.read_csv(players_names, header=0, index_col=0)
df = pd.read_csv(predicted_probas, names = ["Winner%", "Final%", "SF%", "QF%", 'R16%', 'R32%', 'R64%'])
df.index = range(1, len(df)+1)

# Concat names and probas
players_df = players_df.iloc[:,[0,4]]
new_df = pd.concat([players_df, df], axis=1)


# Sort by highest probabilities for Winner
winner_df = new_df.sort_values(by=['Winner%'], ascending=False)
winner_df = winner_df.iloc[:,[0,1,2]]

# Sort by highest probabilities for Final
final_df = new_df.sort_values(by=['Final%'], ascending=False)
final_df = final_df.iloc[:,[0,1,3]]

# Sort by highest probabilities for SF
SF_df = new_df.sort_values(by=['SF%'], ascending=False)
SF_df = SF_df.iloc[:,[0,1,4]]

# Sort by highest probabilities for QF
QF_df = new_df.sort_values(by=['QF%'], ascending=False)
QF_df = QF_df.iloc[:,[0,1,5]]

# Sort by highest probabilities for R16
R16_df = new_df.sort_values(by=['R16%'], ascending=False)
R16_df = R16_df.iloc[:,[0,1,6]]

# Sort by highest probabilities for R32
R32_df = new_df.sort_values(by=['R32%'], ascending=False)
R32_df = R32_df.iloc[:,[0,1,7]]

# Sort by highest probabilities for R64
R64_df = new_df.sort_values(by=['R64%'], ascending=False)
R64_df = R64_df.iloc[:,[0,1,8]]

#-------------------------------------------------------------------------------------------
# BAR PLOT FOR WINNER
#-------------------------------------------------------------------------------------------
nb_players = 10

# Get names, ranks and probas of highest probabilities players
names = winner_df.iloc[:nb_players, 0].tolist()
ranks = winner_df.iloc[:nb_players, 1].tolist()
probas = winner_df.iloc[:nb_players, 2].tolist()

# Compute new labels for x-axis : Name (rank)
x_labels = []
for i, name in enumerate(names):
    new_label = name + ' (' + str(int(ranks[i])) + ')'
    x_labels.append(new_label)

# Plot
plt.figure() 
plt.subplots_adjust(bottom=0.23)
x_pos = np.arange(0, nb_players, 1)
plt.bar(x_pos, probas)

# Set axes
plt.xticks(x_pos, x_labels, rotation=35, ha='right', fontsize=8)
plt.xlabel('Players (rank)', fontsize=11)
plt.ylabel('Probability', fontsize=11)

# Save the plot
plt.savefig("_Figures/rounds_{}_winner.eps".format(year), bbox_inches = "tight")

#-------------------------------------------------------------------------------------------
# BAR PLOT FOR FINAL
#-------------------------------------------------------------------------------------------
nb_players = 10

# Get names, ranks and probas of highest probabilities players
names = final_df.iloc[:nb_players, 0].tolist()
ranks = final_df.iloc[:nb_players, 1].tolist()
probas = final_df.iloc[:nb_players, 2].tolist()

# Compute new labels for x-axis : Name (rank)
x_labels = []
for i, name in enumerate(names):
    new_label = name + ' (' + str(int(ranks[i])) + ')'
    x_labels.append(new_label)

# Plot
plt.figure() 
plt.subplots_adjust(bottom=0.23)
x_pos = np.arange(0, nb_players, 1)
plt.bar(x_pos, probas)

# Set axes
plt.xticks(x_pos, x_labels, rotation=35, ha='right', fontsize=8)
plt.xlabel('Players (rank)', fontsize=11)
plt.ylabel('Probability', fontsize=11)

# Save the plot
plt.savefig("_Figures/rounds_{}_final.eps".format(year), bbox_inches = "tight")

#-------------------------------------------------------------------------------------------
# BAR PLOT FOR SF
#-------------------------------------------------------------------------------------------
nb_players = 10

# Get names, ranks and probas of highest probabilities players
names = SF_df.iloc[:nb_players, 0].tolist()
ranks = SF_df.iloc[:nb_players, 1].tolist()
probas = SF_df.iloc[:nb_players, 2].tolist()

# Compute new labels for x-axis : Name (rank)
x_labels = []
for i, name in enumerate(names):
    new_label = name + ' (' + str(int(ranks[i])) + ')'
    x_labels.append(new_label)

# Plot
plt.figure() 
plt.subplots_adjust(bottom=0.23)
x_pos = np.arange(0, nb_players, 1)
plt.bar(x_pos, probas)

# Set axes
plt.xticks(x_pos, x_labels, rotation=35, ha='right', fontsize=8)
plt.xlabel('Players (rank)', fontsize=11)
plt.ylabel('Probability', fontsize=11)

# Save the plot
plt.savefig("_Figures/rounds_{}_SF.eps".format(year), bbox_inches = "tight")

#-------------------------------------------------------------------------------------------
# BAR PLOT FOR QF
#-------------------------------------------------------------------------------------------
nb_players = 10

# Get names, ranks and probas of highest probabilities players
names = QF_df.iloc[:nb_players, 0].tolist()
ranks = QF_df.iloc[:nb_players, 1].tolist()
probas = QF_df.iloc[:nb_players, 2].tolist()

# Compute new labels for x-axis : Name (rank)
x_labels = []
for i, name in enumerate(names):
    new_label = name + ' (' + str(int(ranks[i])) + ')'
    x_labels.append(new_label)

# Plot
plt.figure() 
plt.subplots_adjust(bottom=0.23)
x_pos = np.arange(0, nb_players, 1)
plt.bar(x_pos, probas)

# Set axes
plt.xticks(x_pos, x_labels, rotation=35, ha='right', fontsize=8)
plt.xlabel('Players (rank)', fontsize=11)
plt.ylabel('Probability', fontsize=11)

# Save the plot
plt.savefig("_Figures/rounds_{}_QF.eps".format(year), bbox_inches = "tight")

#-------------------------------------------------------------------------------------------
# BAR PLOT FOR R16
#-------------------------------------------------------------------------------------------
nb_players = 10

# Get names, ranks and probas of highest probabilities players
names = R16_df.iloc[:nb_players, 0].tolist()
ranks = R16_df.iloc[:nb_players, 1].tolist()
probas = R16_df.iloc[:nb_players, 2].tolist()

# Compute new labels for x-axis : Name (rank)
x_labels = []
for i, name in enumerate(names):
    new_label = name + ' (' + str(int(ranks[i])) + ')'
    x_labels.append(new_label)

# Plot
plt.figure() 
plt.subplots_adjust(bottom=0.23)
x_pos = np.arange(0, nb_players, 1)
plt.bar(x_pos, probas)

# Set axes
plt.xticks(x_pos, x_labels, rotation=35, ha='right', fontsize=8)
plt.xlabel('Players (rank)', fontsize=11)
plt.ylabel('Probability', fontsize=11)

# Save the plot
plt.savefig("_Figures/rounds_{}_R16.eps".format(year), bbox_inches = "tight")

#-------------------------------------------------------------------------------------------
# BAR PLOT FOR R32
#-------------------------------------------------------------------------------------------
nb_players = 10

# Get names, ranks and probas of highest probabilities players
names = R32_df.iloc[:nb_players, 0].tolist()
ranks = R32_df.iloc[:nb_players, 1].tolist()
probas = R32_df.iloc[:nb_players, 2].tolist()

# Compute new labels for x-axis : Name (rank)
x_labels = []
for i, name in enumerate(names):
    new_label = name + ' (' + str(int(ranks[i])) + ')'
    x_labels.append(new_label)

# Plot
plt.figure() 
plt.subplots_adjust(bottom=0.23)
x_pos = np.arange(0, nb_players, 1)
plt.bar(x_pos, probas)

# Set axes
plt.xticks(x_pos, x_labels, rotation=35, ha='right', fontsize=8)
plt.xlabel('Players (rank)', fontsize=11)
plt.ylabel('Probability', fontsize=11)

# Save the plot
plt.savefig("_Figures/rounds_{}_R32.eps".format(year), bbox_inches = "tight")

#-------------------------------------------------------------------------------------------
# BAR PLOT FOR R64
#-------------------------------------------------------------------------------------------
nb_players = 10

# Get names, ranks and probas of highest probabilities players
names = R64_df.iloc[:nb_players, 0].tolist()
ranks = R64_df.iloc[:nb_players, 1].tolist()
probas = R64_df.iloc[:nb_players, 2].tolist()

# Compute new labels for x-axis : Name (rank)
x_labels = []
for i, name in enumerate(names):
    new_label = name + ' (' + str(int(ranks[i])) + ')'
    x_labels.append(new_label)

# Plot
plt.figure()
plt.subplots_adjust(bottom=0.23)
x_pos = np.arange(0, nb_players, 1)
plt.bar(x_pos, probas)

# Set axes
plt.xticks(x_pos, x_labels, rotation=35, ha='right', fontsize=8)
plt.xlabel('Players (rank)', fontsize=11)
plt.ylabel('Probability', fontsize=11)

# Save the plot
plt.savefig("_Figures/rounds_{}_R64.eps".format(year), bbox_inches = "tight")
