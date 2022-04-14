# 201835506
# Ex_covariance matrix

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import linear_model
import seaborn as sns
import matplotlib.pyplot as plt

Age = [30,40,50,60,40]
Income = [200,300,800,600,300]
Yrs_Worked = [10,20,20,20,20]
Vacation = [4,4,1,2,5]

data = {'Age': [30,40,50,60,40],
'Income': [200,300,800,600,300],
'Yrs_Worked': [10,20,20,20,20],
'Vacation': [4,4,1,2,5]
}

df = pd.DataFrame(data,columns=['Age','Income','Yrs_Worked','Vacation'])
# sample covariance matrix
covMatrix = pd.DataFrame.cov(df)
print("------------sample covariance matrix------------")
print(covMatrix)

sns.heatmap(covMatrix, annot=True, fmt='g')
plt.show()
