import numpy as np
import pandas as pd
import random

pred = {}

def gagner(tab):
    if(len(tab) == 1):
        return tab[0]
    
    q = int(len(tab)/2)
    tab1 = tab[0:q]
    tab2 = tab[q:]
    return predict(gagner(tab1), gagner(tab2))

def predict(a, b):
    if(a<b):
        return pred[(a,b)]
    else:
        return pred[(b,a)]

def load(filename):
    return pd.read_csv(filename, header = 0)

if __name__ == '__main__':
    nb_iter = 10000

    tableau = np.array(load("matches.csv"))
    
    players1 = tableau[:,0]
    players2 = tableau[:,1]
    match_result = tableau[:,2]

    results = np.zeros(len(players1))

    for i in range(len(players1)):
        pred[(players1[i],players2[i])] = match_result[i]

    for i in range(nb_iter):
        tab = [1,2,3,4]
        random.shuffle(tab)
        #tab = generateRandomGrid()
        results[gagner(tab)-1] += 1
    
    results /= results.sum()
    print(results)
    print( np.argmax(results) + 1)
    