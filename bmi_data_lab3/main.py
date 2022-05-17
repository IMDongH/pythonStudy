# 201835506 임동혁
# lab3

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import preprocessing, linear_model
import openpyxl
data = pd.read_csv('data/자동차정비업체현황.csv')
nan_data = data
print("***** feature name *****")
for col in data.columns:
    print(col)

print(len(data))
data.dropna(inplace=True)
print(len(data))
data.to_excel('C:/Users/idh10/Desktop/자동차정비업체현황1.xlsx', sheet_name='new_name')
