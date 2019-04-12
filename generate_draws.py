import random

class Draws:
    
    def generate_seed_draw(self, draw, round, max_round):
        """
        Generate one possible draw according to the well known
        distribution of the seeds.

        Parameters
        ----------
            - draw : the draw of the current round
            - round : the current round
            - max_round : the round we want to achieve

        Return
        ------
        A possible draw as a list of 32 id players
        """
        if round > max_round:
            return draw

        next_round_draw = []
        if round==1:
            playersA = [1,2]
            advA = [3,4]
        if round==2:
            playersA = [1,2]
            playersB = [3,4]
            advA = [7,8]
            advB = [5,6]
        if round==3:
            playersA = list(range(1, 5))
            playersB = list(range(5, 9))
            advA = list(range(13, 17))
            advB = list(range(9, 13))
        if round==4:
            playersA = list(range(1, 9))
            playersB = list(range(9, 17))
            advA = list(range(25, 33))
            advB = list(range(17, 25))
        
        for player_id in draw:
            next_round_draw.append(player_id)

            if player_id in playersA:
                random_player_id = advA[random.randint(0, len(advA)-1)]
                advA.remove(random_player_id)
            else:
                random_player_id = advB[random.randint(0, len(advB)-1)]
                advB.remove(random_player_id)

            next_round_draw.append(random_player_id)
            
        return self.generate_seed_draw(next_round_draw, round+1, max_round)

    
    def generate_one_draw(self, seed_draw):
        """
        From a draw of seeds, add randomly the rest of players to have a
        draw of 128 players.
        """
        players = list(range(33, 129))

        # Select 32 players for the 2nd round
        to_select_2nd_round = list(random.sample(players, 32))

        # Get the 64 last players for the 1st round
        to_select_1st_round = [x for i, x in enumerate(players) if not x in to_select_2nd_round]
        to_select_1st_round = list(random.sample(to_select_1st_round, len(to_select_1st_round)))

        # Combine the seed draw with 32 random players for the 2nd round
        draw_2nd_round = [None]*(len(seed_draw)+len(to_select_2nd_round))
        draw_2nd_round[::2] = seed_draw
        draw_2nd_round[1::2] = to_select_2nd_round

        # Combine the remaining players with the draw from 2nd round
        draw = [None]*(len(draw_2nd_round)+len(to_select_1st_round))
        draw[::2] = draw_2nd_round
        draw[1::2] = to_select_1st_round
        return draw

        
    
    def generate_draws(self, nb_draws):
        """Generate a certain amount of possible draws according
        to the well known distribution of the seeds.

        Parameters
        ----------
            - nb_draw : the number of wished draws 

        Return
        ------
        A a certain amount of possible draws as a list of lists
        """
        draws = []
        for i in range(0, nb_draws):
            new_seed_draw = self.generate_seed_draw([1,2], 1, 4)
            new_draw = self.generate_one_draw(new_seed_draw)
            draws.append(new_draw)
        return draws


if __name__ == "__main__":
    x = Draws()

    draws = x.generate_draws(10)
    for draw in draws:
        print(draw)