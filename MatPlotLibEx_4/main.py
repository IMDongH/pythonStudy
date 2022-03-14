# 201835506 임동혁
# MatPlotLib Coding Exercise Bar chart
import numpy as np
import matplotlib.pyplot as plt


wt = np.random.random(100) * 40 + 50
ht = np.random.random(100) * 60 + 140

plt.scatter(wt, ht, color='b')
plt.xlabel('WEIGHT')
plt.ylabel('HEIGHT')
plt.show()
