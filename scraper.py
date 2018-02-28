from urllib.request import urlopen
from urllib.error import HTTPError
from bs4 import BeautifulSoup
import pymysql
import configparser
import time
import re
import sys
import mysql.connector
import numpy as np
import requests
import time
import urllib

# er =14440,16645,18219　エラー
er = 18220 #前回中断した所　ループが途中で切れた

for page in range(er,25800+1):
    print(page)
    url = "https://erogamescape.dyndns.org/~ap2/ero/toukei_kaiseki/game.php?game="
    url = url + str(page)
    time.sleep(5)
    result = requests.get(url)

    c = result.content
    soup = BeautifulSoup(c, 'html.parser')
    samples = soup.find_all(class_="characters_data")

    length = len(samples)
    for k in range(length):
        role = samples[k].find('h5').string
        #print(role)
        for img in samples[k].find_all("img"):
            img_url = img['src']
            img_name = img_url.split("img")[1].lstrip("/").replace('/','-')

            req = urllib.request.urlopen(img_url)
            time.sleep(5)

            if role == "主人公":
                img_name = "hero/" + img_name
            elif role == "メイン":
                img_name = "main/" + img_name
            elif role == "サブ":
                img_name = "sub/" + img_name

            f = open(img_name, "wb")
            f.write(req.read())
            f.close()
        #for chara_name in samples[k].find_all("a"):
        #    print(chara_name.string)
        #print("-------------------------------------------------------")

    #print("--------------------------")
