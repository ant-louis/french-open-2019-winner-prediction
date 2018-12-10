import numpy as np
import pandas as pd
import random
from generate_draws import Draws

class TournamentPredictor:

    def __init__(self):
        self.pred = {}
        file_predict = np.array(pd.read_csv("matches.csv", header = 0))
        players1 = file_predict[:,0]
        players2 = file_predict[:,1]
        match_result = file_predict[:,2]
        for i in range(len(players1)):
            self.pred[(players1[i],players2[i])] = match_result[i]

    def gagner(self, draw):
        if(len(draw) == 1):
            return draw[0]
        q = int(len(draw)/2)
        draw1 = draw[0:q]
        draw2 = draw[q:]
        return self.predict(self.gagner(draw1), self.gagner(draw2))

    def predict(self, a, b):
        if(a<b):
            return self.pred[(a,b)]
        else:
            return self.pred[(b,a)]
    
    def predict(self):
        nb_iter = 10000
        results = np.zeros(len(self.pred))

        draw_generate = Draws()
        draws = draw_generate.generate_draws(nb_iter)

        for draw in draws:
            results[self.gagner(draw)-1] += 1
        results /= results.sum()
        return results

if __name__ == '__main__':
    predicator = TournamentPredictor()
    results = predicator.predict()
    print(results)
    print( np.argmax(results) + 1)
    