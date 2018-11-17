from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import logging

"""
Logging mecanism
"""
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# create a file handler
handler = logging.FileHandler('scraping.log')
handler.setLevel(logging.INFO)
# create a logging format
formatter = logging.Formatter('%(asctime)s - %(message)s')
handler.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(handler)

try:
    for playerID in range(74,500):
        """
        Getting profile information
        """
        browser = webdriver.Chrome(executable_path="/home/tom/Documents/Master1_DataScience/1er QUADRI/Big-Data-Project/chromedriver") #replace with .Firefox(), or with the browser of your choice
        profile = "http://www.ultimatetennisstatistics.com/playerProfile?playerId={}".format(playerID)
        browser.get(profile) #navigate to the page
        innerHTML = browser.execute_script("return document.body.innerHTML") #returns the inner HTML as a string
        soup = BeautifulSoup(innerHTML,"html.parser")

        #Get the name of the player
        Names = soup.find('h3').text.strip().split(" ")
        FirstName = Names[0]
        LastName = " ".join(Names[1:])
        #Dictionnary to contain all info, starting with name
        info = {"First Name" : FirstName, "Last Name" : LastName}
        #Output progress to logfile
        logger.info("Scraping ID: {} Name: {} {}".format(playerID,FirstName,LastName))

        #Loop over table rows
        for tr in  soup.find_all("tr"):
            for td, th in zip(tr.find_all("td"),tr.find_all("th")):
                info[th.text] = [td.text.strip().split(" ")[0]] #Values are lists to simplify dataframe building 

        #Checking best rank, disregarding if higher than 200
        if int(info['Best Rank'][0].split(" ")[0]) > 200:
            logger.info("Dropping cause low rank")
            browser.close()
            continue
        else:
            browser.close()

        """
        Getting performance information
        """
        browser = webdriver.Chrome(executable_path="/home/tom/Documents/Master1_DataScience/1er QUADRI/Big-Data-Project/chromedriver") #replace with .Firefox(), or with the browser of your choice
        performance ="http://www.ultimatetennisstatistics.com/playerPerformance?playerId={}".format(playerID)
        browser.get(performance) #navigate to the page
        innerHTML = browser.execute_script("return document.body.innerHTML") #returns the inner HTML as a string
        soup = BeautifulSoup(innerHTML,"html.parser")

        #Loop over table rows
        for tr in  soup.find_all("tr"):
            for td, th in zip(tr.find_all("td"),tr.find_all("th")):
                #Th and Td are inverted - is normal
                info[td.text] = [th.text.strip().split(" ")[0]] #Values are lists to simplify dataframe building 
        browser.close()
        """
        Getting statistics information
        """
        browser = webdriver.Chrome(executable_path="/home/tom/Documents/Master1_DataScience/1er QUADRI/Big-Data-Project/chromedriver") #replace with .Firefox(), or with the browser of your choice
        stats ="http://www.ultimatetennisstatistics.com/playerStatsTab?playerId={}".format(playerID)
        browser.get(stats) #navigate to the page
        innerHTML = browser.execute_script("return document.body.innerHTML") #returns the inner HTML as a string
        soup = BeautifulSoup(innerHTML,"html.parser")

        #Loop over table rows
        for tr in  soup.find_all("tr"):
            for td, th in zip(tr.find_all("td"),tr.find_all("th")):
                info[td.text] = [th.text.strip().split(" ")[0]] #Values are lists to simplify dataframe building 
        browser.close()

        #Write into dataframe and export to csv using playerID as filename
        df = pd.DataFrame(info)
        df.to_csv("scrapedPlayerInfo/{}.csv".format(playerID), index=False)
except:
    print("An error occured")
    logger.info("An error occured")
    browser.quit()