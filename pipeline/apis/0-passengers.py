#!/usr/bin/env python3
"""Task 0. Can I Join?"""
import requests
API = 'https://swapi-api.alx-tools.com/api/'


def availableShips(pCount):
    """Returns the list of ships that can hold a given number of passengers.
    Args:
        passengerCount (int): The number of passengers.
    Returns:
        list: The list of ships that can hold the given number of passengers.
    """
    params = {'page': 1}
    response = requests.get(API + 'starships', params=params)
    if response.status_code == 200:
        ss = response.json()
        sList = [s['name'] for s in ss['results'] if s['passengers'] != 'n/a'
                 and int(s['passengers'].replace(',', '')) >= pCount]

    while (ss['next']):
        params['page'] += 1
        response = requests.get(ss['next'], params=params)
        if response.status_code == 200:
            ss = response.json()
            sList.extend([s['name'] for s in ss['results']
                          if s['passengers'] not in ['n/a', 'unknown']
                          and int(s['passengers'].replace(',', '')) >= pCount])
    return sList
