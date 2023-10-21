#!/usr/bin/env python3
"""Task 4. Rocket Frequency"""

if __name__ == '__main__':
    """Pulls and prints data from SpaceX API"""
    import requests

    # API endpoints
    LAUNCHES = 'https://api.spacexdata.com/v4/launches/'
    ROCKETS = 'https://api.spacexdata.com/v4/rockets/'

    # Get data from API endpoints
    launches = requests.get(LAUNCHES).json()
    rockets = requests.get(ROCKETS).json()

    # Get rocket names and launch counts
    r_counts = {r['name']: sum([1 for launch in launches
                                if launch['rocket'] == r['id']])
                for r in rockets}

    # Sort by launch count
    r_counts1 = dict(sorted(r_counts.items(),
                            key=lambda x: x[1],
                            reverse=True))

    # Print rocket names and launch counts
    for k, v in r_counts1.items():
        if v > 0:
            print('{}: {}'.format(k, v))
