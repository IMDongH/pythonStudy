# 201835506 임동혁
# Numpy Coding Exercise
import numpy as np


def ColBMI(weight, height):
    BMI = weight / ((height * 0.01) ** 2)
    return BMI


wt = np.random.random(100) * 40 + 50
ht = np.random.random(100) * 60 + 140
bmi = ColBMI(wt, ht)

print(bmi)

for i in range(10):
    print(bmi[i])
