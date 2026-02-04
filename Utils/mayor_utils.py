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
        start_date = datetime.strptime(temp_start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        mayor_perks = entry["mayor_perks"]
        binary_perks = [0 for _ in range(40)]
        for perk in mayor_perks:
            if perk in perk_names:
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
        start_date = datetime.strptime(start_match.group(1), '%m/%d/%Y').replace(tzinfo=timezone.utc)


        
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






def match_mayor_perks(timestamp, mayor_data):
    if isinstance(timestamp, str):
        try:
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except Exception as e:
            print(f"Failed to parse timestamp: {timestamp} -> {e}")
            return [0] * 40

    for mayor in mayor_data:
        if isinstance(mayor['start_date'], str):
            try:
                mayor['start_date'] = datetime.fromisoformat(mayor['start_date'].replace("Z", "+00:00"))
            except Exception as e:
                print(f"Failed to parse mayor start_date: {mayor['start_date']} -> {e}")
                mayor['start_date'] = datetime.min  # fallback

    for i, mayor in enumerate(mayor_data):
        next_start = mayor_data[i + 1]['start_date'] if i + 1 < len(mayor_data) else None
        if next_start:
            if mayor['start_date'] <= timestamp < next_start:
                return mayor['perks'] if isinstance(mayor['perks'], list) else [0] * 40
        else:
            if mayor['start_date'] <= timestamp:
                return mayor['perks'] if isinstance(mayor['perks'], list) else [0] * 40

    return [0] * 40