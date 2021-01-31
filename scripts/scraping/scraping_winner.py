import pandas as pd 
import urllib.request
import os
from bs4 import BeautifulSoup

if __name__== '__main__':

    url_prefix = "http://www.atpworldtour.com"

    matches_2019 = pd.read_csv('match_stats_2018_1.csv')
    length = len(matches_2019)
    winners = []
    losers = []
    i = 0
    for suffix in matches_2019['match_stats_url_suffix']:
        i+= 1
        if i%10 == 0:
            print("{}\{}".format(i, length))
        url = url_prefix + suffix
        request =  urllib.request.Request(url,headers={'User-Agent': 'Mozilla/5.0'})
        page = urllib.request.urlopen(request)
        soup = BeautifulSoup(page,"html.parser")

        # Table containing player info and match results
        match_stats = soup.find("div", {"class":"match-stats-scores"})
        
        # Get first names and last names of both players
        player_left = match_stats.find("div", {"class":"match-stats-player-left"})
        player_left_fn = player_left.find("span", {"class":"first-name"}).text.strip()
        player_left_ln = player_left.find("span", {"class":"last-name"}).text.strip()
        
        player_right = match_stats.find("div", {"class":"match-stats-player-right"})
        player_right_fn = player_right.find("span", {"class":"first-name"}).text.strip()
        player_right_ln = player_right.find("span", {"class":"last-name"}).text.strip()

        # Match results
        match_results_table = match_stats.find( "table", {"class":"scores-table"} )
        table_rows = match_results_table.findAll(lambda tag: tag.name=='tr')

        # Row 'won-game' contains the winner
        player_1_isWinner = table_rows[0].find('td', {'class': 'won-game'})
        player_2_isWinner = table_rows[1].find('td', {'class': 'won-game'})

        player_1_name = table_rows[0].find('a', {'class': 'scoring-player-name'}).text
        player_2_name = table_rows[1].find('a', {'class': 'scoring-player-name'}).text
        _, player_1_ln = player_1_name.split('.')
        _, player_2_ln = player_2_name.split('.')
        

        # Check which of the two players is the winner
        winner_fn, winner_ln, loser_fn, loser_ln = '','','',''
        if player_1_isWinner is None:
            winner_ln = player_2_ln.strip()
            loser_ln  =player_1_ln.strip()
        elif player_2_isWinner is None:
            winner_ln = player_1_ln.strip()
            loser_ln  = player_2_ln.strip()
        else:
            print("No case works")
            winners.append("")
            losers.append("")

        # Match the first name to the last name
        if winner_ln == player_left_ln:
            winner_fn = player_left_fn
            loser_fn = player_right_fn
        elif winner_ln == player_right_ln:
            winner_fn = player_right_fn
            loser_fn = player_left_fn
        else:
            print("Error: {}".format(url))
            
        # Append all names in a list to add to dataframe at the end
        winners.append("{} {}".format(winner_fn, winner_ln))
        losers.append("{} {}".format(loser_fn, loser_ln))

    matches_2019["winner_name"] = winners 
    matches_2019["losers_name"] = losers 


    matches_2019.to_csv("matches_2018.csv")
