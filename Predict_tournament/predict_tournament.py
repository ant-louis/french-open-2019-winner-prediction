import numpy as np
import pandas as pd
import random
from generate_draws import Draws
import sys
class TournamentPredictor:

    def __init__(self, filename):
        """
        Initialize the prdictionary with the prediction of winners in matches.csv
        """
        self.pred_dictionary = {}
        file_predict = np.array(pd.read_csv(filename, header = 0, dtype=int))
        for i in range(len(file_predict[:,0])):
            self.pred_dictionary[(file_predict[i,0],file_predict[i,1])] = file_predict[i,2]

    def winner(self, draw):
        """
        Return the winner of one draw

        Parameters
        ----------
        draw: array
            An array of the rank of players corresponding to one
            particular draw

        Return
        ------
        W: int
            The rank of the winner of the draw
        """
        if(len(draw) == 1):
            return draw[0]
        q = int(len(draw)/2)
        draw1 = draw[0:q]
        draw2 = draw[q:]
        return self.predict_match(self.winner(draw1), self.winner(draw2))

    def predict_match(self, a, b):
        """
        Predict the winner of a match

        Parameters
        ----------
        a: int
            The rank of the first player
        b: int
            The rank of the second player

        Return
        ------
        R: int
            The rank of the winner of game
        """
        if((a,b) not in self.pred_dictionary and (b,a) not in self.pred_dictionary):
            print(a,b)
            print("Not in Dictionnary")
            exit()
        if(a<b):
            return self.pred_dictionary[(a,b)]
        else:
            return self.pred_dictionary[(b,a)]
    
    def predict(self, nb_draws):
        """
        Return the probability for all players to win the tournament

        Parameters
        ----------
        nb_draws: int
            The number of prediction to make

        Return
        ------
        R: array
            An array of probabilities for each player to win
            the tournament
        """
        results = np.zeros(32)

        draw_generate = Draws()
        draws = draw_generate.generate_draws(nb_draws)

        for draw in draws:
            results[self.winner(draw)-1] += 1
        results /= results.sum()
        return results


if __name__ == '__main__':

    if(len(sys.argv) != 2):
        print("Call with \"python predict_tournament.py matches_examples.csv\"")
        exit()
    matches_file = sys.argv[1]

    predicator = TournamentPredictor(matches_file)
    results = predicator.predict(10000)

    print(results)
    print(np.argmax(results) + 1)
    
    