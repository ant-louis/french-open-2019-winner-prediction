import numpy as np
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
    if(random.randint(1,3) == 1):
        return a
    else:
        return b



if __name__ == '__main__':
    nb_iter = 10000

    #A changer avec les tableaux pris sur MongoDB
    players1 = [1,2,3,4]
    players2 = [4,5,6,11]
    match_result = [7,8,9,12]

    tab = [1,2,3,4]

    results = np.zeros(len(players1))

    for i in range(len(players1)):
        pred[(players1[i],players2[i])] = match_result[i]

    for i in range(nb_iter):
        random.shuffle(tab)
        #tab = generateRandomGrid()
        results[gagner(tab)-1] += 1
    
    results /= results.sum()
    print(results)
    print( np.argmax(results) + 1)