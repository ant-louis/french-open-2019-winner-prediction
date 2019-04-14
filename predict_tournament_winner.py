import numpy as np
import pandas as pd
import random
from generate_draws import Draws
import sys


class TournamentPredictor:
    def __init__(self, filename):
        """
        Initialize the dictionary with the prediction of winners in matches.csv
        """
        self.pred_dictionary = {}
        file_predict = np.array(pd.read_csv(filename, header=0))
        for i in range(len(file_predict[:,0])):
            self.pred_dictionary[(int(file_predict[i,2]), int(file_predict[i,3]))] = float(file_predict[i,5])

    def winner(self, draw):
        """
        Given a draw, return the winner of that draw according to our predictions.
        """
        if(len(draw) == 1):
            return draw[0]
        q = int(len(draw)/2)
        draw1 = draw[0:q]
        draw2 = draw[q:]
        return self.predict_match(self.winner(draw1), self.winner(draw2))

    def predict_match(self, a, b):
        """
        Given the ids of two plays, return the winner of the match according to
        our predictions.
        """
        if((a, b) not in self.pred_dictionary and (b, a) not in self.pred_dictionary):
            print(a, b)
            print("Not in Dictionnary")
            exit()
        if(a < b):
            if random.random() < self.pred_dictionary[(a, b)]:
                return a
            else:
                return b
        else:
            if random.random() < self.pred_dictionary[(b, a)]:
                return b
            else:
                return a
    
    def predict(self, nb_draws):
        """
        Considering a ceertain number of possible draws, return the probability for
        each player to win the tournament.
        """
        results = np.zeros(128)
        draw = Draws()
        draws = draw.generate_draws(nb_draws)

        for draw in draws:
            results[self.winner(draw)-1] += 1
        results /= results.sum()
        return results


if __name__ == '__main__':

    if(len(sys.argv) != 2):
        print("Call with \"python predict_tournament.py matches_examples.csv\"")
        exit()
    matches_file = sys.argv[1]
    matches_file = "_Data/Predictions/" + matches_file
    predicator = TournamentPredictor(matches_file)
    results = predicator.predict(1000000)


    print("Winning probabilities wrt. seed rank with filename '{}':".format(matches_file))
    indices = np.argsort(-results) # Minus in front to sort by descending
    print("Seed rank \t Probability")
    for i in indices:
        print("{} \t\t {}".format(i+1, results[i]))
