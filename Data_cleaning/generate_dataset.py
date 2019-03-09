import pandas as pd
import numpy as np
from math import exp

def load_from_csv(path, delimiter=','):
    """
    Load csv file and return a NumPy array of its data

    Parameters
    ----------
    path: str
        The path to the csv file to load
    delimiter: str (default: ',')
        The csv field delimiter

    Return
    ------
    D: array
        The NumPy array of the data contained in the file
    """
    return pd.read_csv(path, delimiter=delimiter,dtype=float)



matches = load_from_csv("matches.csv")
dataset = np.matrix(np.shape(matches))

for i in range(np.shape(matches)[0]):
    for j in range(np.shape(matches)[1]):
        hist = matches[i,j]
        wheight = 1
        for k in range(i):
            if matches[i,'name'] == matches[k,'name']:
                w = min(1, exp(matches[k,'date']-matches[i,'date']))
                hist += w * matches[k,j]
                wheight += w
        dataset[i,j] = hist/wheight
