from hypixel_api_lib import Elections
import numpy as np
from datetime import datetime
import requests
import sys
import re


current_datetime = datetime.now()

url = "https://sky.coflnet.com/api/mayor?from=2025-02-17T20%3A03%3A10.937Z&to="+ current_datetime.strftime("%Y-%m-%dT%H%%3A%M%%3A%S.%fZ")
response = requests.get(url)
html_content = response.text
mayors = html_content.split("\"start\"")
mayors.pop(0)


mayor_perks = []
for mayor in mayors:
    binary_perks = [0 for _ in range(40)]
    matches = re.findall(r'"name":"([^"]*)"', mayor)
    for perk_name in matches:
        with open("perk_names.txt", "r") as f:
            perk_names = f.read().splitlines()
        if perk_name in perk_names:
            perk_index = perk_names.index(perk_name)
            binary_perks[perk_index] = 1
    mayor_perks.append(binary_perks)
print(mayor_perks)

    

    



    

