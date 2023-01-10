#!/usr/bin/env python3
""" Plotting Task 4-frequency"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

""" This makes it like the picture in the project but i big time disagree with
    the way it was implemented. I believe using .bar will be better for manipulating
    each bar after it has been filled with data"""
plt.hist(student_grades, bins=[40, 50, 60, 70, 80, 90, 100], edgecolor='black')
plt.title("Project A")
plt.xlabel("Grades")
plt.ylabel("Number of Students")
plt.xlim([0, 100])
plt.ylim([0, 30])
plt.xticks(range(0, 110, 10))
plt.show()
