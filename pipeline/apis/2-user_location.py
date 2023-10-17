#!/usr/bin/env python3
"""Task 2. Rate me is you can!"""
import requests
import sys

if __name__ == '__main__':
    """Initializing script on use"""

    user = requests.get(sys.argv[1])

    if user.status_code == 200:
        print(user.json()['location'])
    elif user.status_code == 403:
        print('Reset in {} min'.format(user.headers['X-RateLimit-Reset'] / 60))
    if user.status_code == 404:
        if user.json()['message'] == 'Not Found':
            print('Not found')
