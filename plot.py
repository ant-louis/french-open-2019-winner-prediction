import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")  # Fixing a bug of matplotlib on MacOS
#matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
plt.style.use('ggplot')



def timing_function(t):
    return 0.8**t

def plot_timing_function():
    t = np.arange(1, 17, 1)
    fig, ax = plt.subplots()

    ax.plot(t, timing_function(t).astype(np.float), color='lightcoral')
    ax.hlines(y=1, xmin=0, xmax=1, color='lightcoral')
    ax.vlines(x=1, ymin=0.8, ymax=1, color='lightcoral')

    ax.set_xlabel(r'$\Delta t$', fontsize=16)
    ax.set_ylabel(r'$w_{time}$', fontsize=18, rotation=0)
    ax.yaxis.set_label_coords(-0.07,1.05)
    ax.xaxis.set_label_coords(1,-0.07)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.xlim(0, 16)
    plt.ylim(0, 1)
    plt.show()


def plot_feature_importance(filename, nb_features=None):
    """Plot the desired number of features along with their accuracy"""

    data = pd.read_csv(filename, sep=',')

    if nb_features is None:
        nb_features = len(data)

    feature = data.iloc[:nb_features, 0].tolist()
    importance = data.iloc[:nb_features, 1].tolist()

    # Choose the position of each barplots on the x-axis (space=1,4,3,1)
    x_pos = np.arange(0, nb_features, 1)
    
    # Create bars
    plt.figure() 
    plt.subplots_adjust(bottom=0.23)
    plt.bar(x_pos, importance)
    
    # Create names on the x-axis
    plt.xticks(x_pos, feature, rotation=35, ha='right', fontsize=8)

    plt.ylabel('Importance', fontsize=13)
    plt.xlabel('Feature', fontsize=13)
    plt.show()




if __name__ == "__main__":

    #plot_timing_function()

    features_file = '_Data/Training_dataset/feature_importance.csv'
    plot_feature_importance(features_file)


    