import requests
import re
import os
from datetime import datetime


def get_mayor_perks():
    """Fetch mayor perks data and return as binary vectors with timestamps.
    
    Returns:
        List of dictionaries containing:
            - start_date: datetime object for when the mayor term started
            - perks: list of 40 binary values (0 or 1) representing active perks
    """
    current_datetime = datetime.now()
    url = f"https://sky.coflnet.com/api/mayor?from=2025-02-17T20%3A03%3A10.937Z&to={current_datetime.strftime('%Y-%m-%dT%H%%3A%M%%3A%S.%fZ')}"
    
    response = requests.get(url)
    html_content = response.text
    mayors = html_content.split('"start"')
    mayors.pop(0)
    
    mayor_data = []
    for mayor in mayors:
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


def get_mayor_start_date(mayor_data):
    """Get the earliest date when mayor data becomes available.
    
    Args:
        mayor_data: List of mayor data from get_mayor_perks()
        
    Returns:
        datetime object of the earliest mayor start date, or None if no data
    """
    if not mayor_data:
        return None
    return min(mayor['start_date'] for mayor in mayor_data)


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
