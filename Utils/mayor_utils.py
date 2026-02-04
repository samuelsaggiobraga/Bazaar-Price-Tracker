import requests
import re
import os
from datetime import datetime, timezone
import json

def get_mayor_perks():
    """Fetch mayor perks data and return as binary vectors with timestamps.
    
    Returns:
        List of dictionaries containing:
            - start_date: datetime object for when the mayor term started
            - perks: list of 40 binary values (0 or 1) representing active perks
    """
    current_datetime = datetime.now(timezone.utc)
    url = f"https://sky.coflnet.com/api/mayor?from=2022-05-17T20%3A03%3A10.937Z&to={current_datetime.strftime('%Y-%m-%dT%H%%3A%M%%3A%S.%fZ')}"
    
    response = requests.get(url)
    html_content = response.text
    mayors = html_content.split('"start"')
    mayors.pop(0)
    
    mayor_data = []
    mayor_file = os.path.join(os.path.dirname(__file__), "88-207mayors.json")
    with open(mayor_file, 'r') as file:
        data = json.load(file)
    perk_file = os.path.join(os.path.dirname(__file__), "perk_names.txt")
    with open(perk_file, 'r') as file:
        perk_names = file.read().splitlines()

    for entry in data:
        temp_start_date = entry["start_date"]
        start_date = datetime.strptime(temp_start_date, '%Y-%m-%d')
        mayor_perks = entry["mayor_perks"]
        for perk in mayor_perks:
            if perk in perk_names:
                binary_perks = [0 for _ in range(40)]
                perk_index = perk_names.index(perk)
                binary_perks[perk_index] = 1
        mayor_data.append({
            'start_date': start_date,
            'perks': binary_perks
        })
    

    for mayor in mayors:
        temp = mayor.find('"year":')
        mayor = mayor[:temp]

        # Date format is MM/DD/YYYY in the API response
        start_match = re.search(r'(\d{2}/\d{2}/\d{4})', mayor)
        if not start_match:
            continue
        start_date = datetime.strptime(start_match.group(1), '%m/%d/%Y')
        
        binary_perks = [0 for _ in range(40)]
        matches = re.findall(r'"name":"([^"]*)"', mayor)
        
        perk_file = os.path.join(os.path.dirname(__file__), "perk_names.txt")
        with open(perk_file, "r") as f:
            perk_names = f.read().splitlines()
        
        for perk_name in matches:
            if perk_name in perk_names:
                perk_index = perk_names.index(perk_name)
                binary_perks[perk_index] = 1
        
        mayor_data.append({
            'start_date': start_date,
            'perks': binary_perks
        })
    
    return mayor_data

get_mayor_perks()



def match_mayor_perks(timestamp_str, mayor_data):
    """Match a timestamp to the appropriate mayor perks.
    
    Args:
        timestamp_str: ISO format timestamp string (e.g., "2025-02-20T12:00:00.000Z")
        mayor_data: List of mayor data from get_mayor_perks()
        
    Returns:
        List of 40 binary values representing active mayor perks for that timestamp
    """
    try:
        data_date = datetime.strptime(timestamp_str[:10], '%Y-%m-%d')
    except:
        return [0] * 40
    
    for i, mayor in enumerate(mayor_data):
        if i + 1 < len(mayor_data):
            if mayor['start_date'] <= data_date < mayor_data[i + 1]['start_date']:
                return mayor['perks']
        else:
            if mayor['start_date'] <= data_date:
                return mayor['perks']
    
    return [0] * 40
