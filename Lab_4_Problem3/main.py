# 201835506 임동혁
# lab4_problem3

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier

dir_train = './data/mnist_train.csv'
dir_test = './data/mnist_test.csv'
df_train = pd.read_csv(dir_train)
df_test = pd.read_csv(dir_test)

# split the data into target and features
x = df_train.drop(['label'], axis=1)
y = df_train['label']

# create a KNN model
knn_3 = KNeighborsClassifier(n_neighbors=3)
# train model with cv of 5
cv_scores_3 = cross_val_score(knn_3, x, y, cv=5)
print("\n________When train model with cv of 5 and k is 3________")
print(cv_scores_3)

# create a KNN model
knn_5 = KNeighborsClassifier(n_neighbors=5)
# train model with cv of 5
cv_scores_5 = cross_val_score(knn_5, x, y, cv=5)
print("\n________When train model with cv of 5 and k is 5________")
print(cv_scores_5)

# GridSearch to test all values for n_neighbors
param_grid = {'n_neighbors': np.arange(1, 25)}
knn_gscv = GridSearchCV(knn_5, param_grid, cv=5)
knn_gscv.fit(x, y.values.ravel())
print("\n________GridSearch________")
print("Best parameter : ", knn_gscv.best_params_)
print("Best score : ", knn_gscv.best_score_)

# RandomizedSearchCV to test all values for n_neighbors
param_grid = {'n_neighbors': np.arange(1, 25)}
knn_gscv = RandomizedSearchCV(knn_5, param_grid, cv=5)
knn_gscv.fit(x, y.values.ravel())
print("\n________RandomizedSearchCV________")
print("Best parameter : ", knn_gscv.best_params_)
print("Best score : ", knn_gscv.best_score_)
