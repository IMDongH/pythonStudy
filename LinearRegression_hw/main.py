# 201835506 임동혁
# hw_LinearRegression

import numpy as np
import pandas as pd
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

spends = np.array([2400, 3650, 2350, 4950, 3100, 2500, 5106, 3100, 2900, 1750])
spends = spends.reshape(-1, 1)
income = np.array([41200, 50100, 52000, 66000, 44500, 37700, 73500, 37500, 56700, 35600])

model = LinearRegression()
scores = []
model.fit(spends, income)
income_predicted = model.predict(spends)
print("coef_",model.coef_, "intercept_ ",model.intercept_)
plt.xlabel('spends')
plt.ylabel('income')
plt.plot(spends, income, 'o')
plt.plot(spends, income_predicted)
plt.show()

total = spends.sum()
ytotal = income.sum()
sum=0
for i in range(spends.size):
    sum = sum + spends[i]*spends[i]

ysum=0
for i in range(spends.size):
        ysum = ysum + spends[i] * income[i]

print("sum", sum)
print("syum", ysum)
print("xsum", total)
print("ysum", ytotal)

a=9.79600427
b=18322.828826279147

a1=b+a*3500
a2=b+a*5300
print(a1)
print(a2)