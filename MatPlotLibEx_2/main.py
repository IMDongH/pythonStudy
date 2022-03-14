# 201835506 임동혁
# MatPlotLib Coding Exercise Histogram
import numpy as np
import matplotlib.pyplot as plt


def ColBMI(weight, height):
    BMI = weight / ((height * 0.01) ** 2)
    return BMI

wt = np.random.random(100) * 40 + 50
ht = np.random.random(100) * 60 + 140
bmi = ColBMI(wt, ht)

langs = ['Underweight', 'Healthy', 'Overweight', 'Obese']


plt.hist(bmi, bins=[0,18.5, 25, 30, 50])
plt.xticks([0,18.5,25,30,])
plt.xlabel('BMI')

plt.show()
