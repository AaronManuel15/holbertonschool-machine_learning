#!/usr/bin/env python3
"""Task 3. First launch"""

if __name__ == '__main__':
    """Pulls and prints data from SpaceX API"""
    import requests

    LAUNCH = 'https://api.spacexdata.com/v5/launches/upcoming'
    ROCKET = 'https://api.spacexdata.com/v4/rockets/'
    LAUNCHPAD = 'https://api.spacexdata.com/v4/launchpads/'

    # Getting all upcoming launches
    S_response_LAUNCH = requests.get(LAUNCH)

    # Sorting for first launch
    launch = sorted(S_response_LAUNCH.json(), key=lambda x: x['date_unix'])[0]

    # Getting launch data for launch and id's for rocket and launchpad
    launchname = launch['name']
    launchdate = launch['date_local']
    rocketid = launch['rocket']
    launchpadid = launch['launchpad']

    # Getting rocket data with rocketid
    S_response_ROCKET = requests.get(ROCKET + rocketid)
    rocketname = S_response_ROCKET.json()['name']

    # Getting launchpad data with launchpadid
    S_response_LAUNCHPAD = requests.get(LAUNCHPAD + launchpadid)
    launchpadname = S_response_LAUNCHPAD.json()['name']
    launchpadlocality = S_response_LAUNCHPAD.json()['locality']

    # Printing data
    print('{} ({}) {} - {} ({})'.format(launchname, launchdate, rocketname,
                                        launchpadname, launchpadlocality))
