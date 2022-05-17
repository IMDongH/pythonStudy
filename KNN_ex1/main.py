import pandas as pd
import numpy as np
from sklearn import preprocessing

dir = "data/KNN.xlsx"
data = pd.read_excel(dir)

print("___________DATA SET___________")
print(data)

predict_data=[161,61]
# Select train data
train = data.loc[:, 'HEIGHT':'WEIGHT']

# Add data to predict
train = train.append({'HEIGHT': 161, 'WEIGHT': 61}, ignore_index=True)

# Select target data
target = data.loc[:, 'T SHIRT SIZE']

# Normalization the features
scaler = preprocessing.StandardScaler()
scaled_train = scaler.fit_transform(train)
scaled_train = np.array(scaled_train)


# Calculate distance using Euclidean Distance
def distance(point_1, point_2):
    dis = 0.0
    for i in range(len(point_1)):
        dis += (point_1[i] - point_2[i]) ** 2
    return np.sqrt(dis)


# distance list
def distance_list(train, predict):
    distances = list()
    for train_row in train:
        dis = distance(predict, train_row)
        distances.append(dis)
    return distances


# predict data was scaled
scaled_predict = scaled_train[18]

# Delete data for prediction in train data
np.delete(scaled_train, 18, axis=0)

# Calculate distance from new customer
dis = distance_list(scaled_train, scaled_predict)

# Show Data frames with added distance calculations
train['T SHIRT SIZE'] = target
train['DISTANCE'] = dis
print("___________Distance___________")
print(train)

# Sort distance data
sorted_train = train.sort_values(by=['DISTANCE'], axis=0)
print("___________Sorted Distance___________")
print(sorted_train)

# Prediction results when k is 5
neighbors = sorted_train.iloc[1:6, 2:4]

predict = neighbors['T SHIRT SIZE'].max()
print("___________Result___________")
print("k = 5, predict size = ", predict)
