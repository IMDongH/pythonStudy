# 201835506 임동혁
# lab4_problem2

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

dir = './data/winequality-red.csv'
df = pd.read_csv(dir,delimiter=';')

# split the data into target and features
x = df.drop(['quality'], axis=1)
y = df['quality']

# Split the dataset into 9/10 for training and 1/10 for testing
x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(x, y, test_size=0.1, shuffle=True)

# Make Decision Tree model
model = tree.DecisionTreeClassifier(criterion='entropy')
model = model.fit(x_train_1, y_train_1)
y_prediction = model.predict(x_test_1)

print('________When Split the dataset into 9/10 for training and 1/10 for testing________')
print('Accurancy : ',accuracy_score(y_test_1, y_prediction))

# Split the dataset into 8/10 for training and 2/10 for testing
x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(x, y, test_size=0.2, shuffle=True)

# Make Decision Tree model
model = tree.DecisionTreeClassifier(criterion='entropy')
model = model.fit(x_train_2, y_train_2)
y_prediction = model.predict(x_test_2)

print('\n________When Split the dataset into 8/10 for training and 2/10 for testing________')
print('Accurancy : ',accuracy_score(y_test_2, y_prediction))

# Split the dataset into 7/10 for training and 3/10 for testing
x_train_3, x_test_3, y_train_3, y_test_3 = train_test_split(x, y, test_size=0.3, shuffle=True)

# Make Decision Tree model
model = tree.DecisionTreeClassifier(criterion='entropy')
model = model.fit(x_train_3, y_train_3)
y_prediction = model.predict(x_test_3)

print('\n________When Split the dataset into 7/10 for training and 3/10 for testing________')
print('Accurancy : ',accuracy_score(y_test_3, y_prediction))