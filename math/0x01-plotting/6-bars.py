#!/usr/bin/env python3
""" Plotting Task 6-bars"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

plt.bar(["Farrah", "Fred", "Felicia"],
        fruit[:][0], color='red',
        label='apples', width=0.5)
plt.bar(["Farrah", "Fred", "Felicia"],
        fruit[:][1], bottom=fruit[0],
        color='yellow', label='bananas', width=0.5)
plt.bar(["Farrah", "Fred", "Felicia"],
        fruit[:][2],
        bottom=fruit[0] + fruit[1],
        color='#ff8000', label='oranges', width=0.5)
plt.bar(["Farrah", "Fred", "Felicia"],
        fruit[:][3],
        bottom=fruit[0] + fruit[1] + fruit[2],
        color='#ffe5b4', label='peaches', width=0.5)
plt.title("Number of Fruit per Person")
plt.yticks(range(0, 81, 10))
plt.ylabel("Quantity of Fruit")
plt.legend()
plt.show()
