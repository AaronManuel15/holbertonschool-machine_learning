#!/usr/bin/env python3
"""Task 2. Rate me is you can!"""

if __name__ == '__main__':
    """Initializing script on use"""
    import requests
    import sys
    import time

    user = requests.get(sys.argv[1])

    if user.status_code == 200:
        print(user.json()['location'])
    elif user.status_code == 403:
        time = (int(user.headers['X-RateLimit-Reset']) - int(time.time()))
        print('Reset in {} min'.format(time // 60))
    if user.status_code == 404:
        print('Not found')
