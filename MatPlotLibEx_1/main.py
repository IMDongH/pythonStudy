# 201835506 임동혁
# MatPlotLib Coding Exercise Bar chart
import numpy as np
import matplotlib.pyplot as plt

def ColBMI(weight, height):
    BMI = weight / ((height * 0.01) ** 2)
    return BMI

def ColBMIStatus(BMI):
    count = np.zeros(4)
    for i in range(bmi.size):
        if bmi[i] < 18.5:
            count[0] += 1
        elif 18.5 <= bmi[i] < 25.0:
            count[1] += 1
        elif 25.0 <= bmi[i] < 30.0:
            count[2] += 1
        else:
            count[3] += 1
    return count

wt = np.random.random(100) * 40 + 50
ht = np.random.random(100) * 60 + 140
bmi = ColBMI(wt, ht)

langs =['Underweight','Healthy','Overweight','Obese']

count = ColBMIStatus(bmi)

plt.bar(langs,count)
plt.show()

