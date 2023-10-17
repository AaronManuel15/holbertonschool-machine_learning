#!/usr/bin/env python3
"""Task 1. Where I am?"""
import requests
API = 'https://swapi-api.alx-tools.com/api/'


def sentientPlanets():
    """returns the list of names of the home planets of all sentient species.
    Returns:
        list: The list of names of the home planets of all sentient species.
    """
    params = {'page': 1}
    response = requests.get(API + 'species', params=params)
    if response.status_code == 200:
        sp = response.json()
        sList = [s['homeworld'] for s in sp['results']
                 if s['designation'] == 'sentient'
                 or s['classification'] == 'sentient']

    while (sp['next']):
        params['page'] += 1
        response = requests.get(sp['next'], params=params)
        if response.status_code == 200:
            sp = response.json()
            sList.extend([s['homeworld'] for s in sp['results']
                          if s['designation'] == 'sentient'
                          or s['classification'] == 'sentient'])

    sList.remove(None)
    for i, s in enumerate(sList):
        response = requests.get(s)
        if response.status_code == 200:
            sList[i] = response.json()['name']
    return sList
