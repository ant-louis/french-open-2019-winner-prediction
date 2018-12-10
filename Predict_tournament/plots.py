import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('feature_importance.csv', sep=',')

feature = data['feature'].tolist()
importance = data['importance'].tolist()
 
# Choose the position of each barplots on the x-axis (space=1,4,3,1)
x_pos = np.arange(0, 50, 1)
 
# Create bars
plt.figure(figsize=(30, 15)) 
plt.bar(x_pos, importance[:50])
 
# Create names on the x-axis
plt.xticks(x_pos, feature[:50], rotation='vertical', fontsize = 25)
plt.yticks(fontsize = 25)

plt.ylabel('Importance', fontsize = 40)
plt.xlabel('Feature', fontsize = 40)

plt.savefig('importance.svg',format="svg")