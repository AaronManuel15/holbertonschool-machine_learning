#!/usr/bin/env python3
"""Task 1. Create the loop"""


while(True):
    question = input('Q: ')
    if question.lower() in ['exit', 'quit', 'goodbye', 'bye']:
        print('A: Goodbye')
        break
    else:
        print('A: ')
