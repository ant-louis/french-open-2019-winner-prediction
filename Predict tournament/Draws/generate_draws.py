import itertools
import random
import operator
import copy


class Draws:

    def __init__(self):
        self.seeds_1_8_with_seeds_25_32, self.seeds_9_16_with_seeds_17_24 = self.__generate_all_draws()

    def __generate_all_draws(self):
        # Starting point of the draws
        seeds_1_to_8 = [
            [1,5,7,3,4,6,8,2], [1,7,5,3,4,6,8,2], [1,5,7,3,4,8,6,2], [1,7,5,3,4,8,6,2],
            [1,6,7,3,4,5,8,2], [1,7,6,3,4,5,8,2], [1,6,7,3,4,8,5,2], [1,7,6,3,4,8,5,2],
            [1,5,8,3,4,6,7,2], [1,8,5,3,4,6,7,2], [1,5,8,3,4,7,6,2], [1,8,5,3,4,7,6,2],
            [1,6,8,3,4,5,7,2], [1,8,6,3,4,5,7,2], [1,6,8,3,4,7,5,2], [1,8,6,3,4,7,5,2],
            [1,5,7,4,3,6,8,2], [1,7,5,4,3,6,8,2], [1,5,7,4,3,8,6,2], [1,7,5,4,3,8,6,2],
            [1,6,7,4,3,5,8,2], [1,7,6,4,3,5,8,2], [1,6,7,4,3,8,5,2], [1,7,6,4,3,8,5,2],
            [1,5,8,4,3,6,7,2], [1,8,5,4,3,6,7,2], [1,5,8,4,3,7,6,2], [1,8,5,4,3,7,6,2],
            [1,6,8,4,3,5,7,2], [1,8,6,4,3,5,7,2], [1,6,8,4,3,7,5,2], [1,8,6,4,3,7,5,2]
        ]
        seeds_9_to_16 = copy.deepcopy(seeds_1_to_8)
        for grid in seeds_9_to_16:
            for n, i in enumerate(grid):
                if i==1:
                    grid[n]=9
                if i==2:
                    grid[n]=10
                if i==3:
                    grid[n]=11
                if i==4:
                    grid[n]=12
                if i==5:
                    grid[n]=13
                if i==6:
                    grid[n]=14
                if i==7:
                    grid[n]=15
                if i==8:
                    grid[n]=16

        # make all possible permutations
        seeds_1_8_with_25_32 = []
        seeds_9_16_with_17_24 = []
        seeds_25_to_32 = list(itertools.permutations([25,26,27,28,29,30,31,32]))
        seeds_17_to_24 = list(itertools.permutations([17,18,19,20,21,22,23,24]))

        for draw in seeds_1_to_8:
            for permut_draw in seeds_25_to_32:
                new_draw = []
                for i in range (0, 8):
                    new_draw.append((draw[i], permut_draw[i])) 

                seeds_1_8_with_25_32.append(new_draw)
                
        for draw in seeds_9_to_16:
            for permut_draw in seeds_17_to_24:
                new_draw = []
                for i in range (0, 8):
                    new_draw.append((draw[i], permut_draw[i])) 
                
                seeds_9_16_with_17_24.append(new_draw)
        
        return seeds_1_8_with_25_32, seeds_9_16_with_17_24

    def generate_draws(self, nb_draws):
        draws = []

        for i in range(nb_draws):
            random1 = random.randint(1, len(self.seeds_1_8_with_seeds_25_32))
            random2 = random.randint(1, len(self.seeds_9_16_with_seeds_17_24))

            draw1 = self.seeds_1_8_with_seeds_25_32[random1]
            draw2 = self.seeds_9_16_with_seeds_17_24[random2]

            new_draw = [x for x in itertools.chain.from_iterable(itertools.zip_longest(draw1,draw2)) if x]
            draws.append(new_draw)

        return draws

if __name__ == "__main__":

    x = Draws()
    draws = x.generate_draws(10)

    for draw in draws:
        print(draw)