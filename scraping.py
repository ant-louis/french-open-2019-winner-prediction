from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd

browser = webdriver.Chrome(executable_path="/home/tom/Documents/Master1_DataScience/1er QUADRI/Big-Data-Project/chromedriver") #replace with .Firefox(), or with the browser of your choice

# profile ='http://www.ultimatetennisstatistics.com/playerProfile?playerId=3819'
# browser.get(profile) #navigate to the page

# innerHTML = browser.execute_script("return document.body.innerHTML") #returns the inner HTML as a string
# soup = BeautifulSoup(innerHTML,"html.parser")

# titles = []
# values = []
# for tr in  soup.find_all("tr"):
#     for td, th in zip(tr.find_all("td"),tr.find_all("th")):
#         titles.append(th.text)
#         values.append(td.text)
# print(titles)
# print(values)

# performance ='http://www.ultimatetennisstatistics.com/playerPerformance?playerId=3819'
# browser.get(performance) #navigate to the page
# innerHTML = browser.execute_script("return document.body.innerHTML") #returns the inner HTML as a string
# soup = BeautifulSoup(innerHTML,"html.parser")
# titles = []
# values = []
# for tr in  soup.find_all("tr"):
#     for td, th in zip(tr.find_all("td"),tr.find_all("th")):
#         titles.append(th.text)
#         values.append(td.text)
# print(titles)
# print(values)
# browser.close()



stats ='http://www.ultimatetennisstatistics.com/playerStatsTab?playerId=3819'
browser.get(stats) #navigate to the page
innerHTML = browser.execute_script("return document.body.innerHTML") #returns the inner HTML as a string
soup = BeautifulSoup(innerHTML,"html.parser")
titles = []
values = []
for tr in  soup.find_all("tr"):
    for td, th in zip(tr.find_all("td"),tr.find_all("th")):
        titles.append(th.text)
        values.append(td.text)
print(titles)
print(values)
browser.close()