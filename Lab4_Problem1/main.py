# 201835506 임동혁
# lab4_problem1

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

dir = './data/housing.csv'
df = pd.read_csv(dir)

# handle missing value
data = df.dropna(axis=1, how='any')

# split the data into target and features
x = data.drop(['median_house_value'], axis=1)
y = data['median_house_value']

# Encoding categorical value
enc = LabelEncoder()
encoding = pd.DataFrame(data['ocean_proximity'])
enc.fit(encoding)
x['ocean_proximity'] = pd.DataFrame(enc.transform(encoding))

# scaling the features
model = StandardScaler()
scaled_x = model.fit_transform(x)

stratify_data = pd.qcut(data['median_house_value'], 10)
reg = LinearRegression()

# Split the dataset into 4/5 for training and 1/5 for testing
x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(x, y, test_size=0.2, shuffle=True, stratify=stratify_data)
reg.fit(x_train_1, y_train_1)
predict_1 = reg.predict(x_test_1)
accuracy_1 = reg.score(x_test_1, y_test_1)
print("________Result of Split the dataset into 4/5 for training and 1/5 for testing________")
print("Accuracy : ", accuracy_1)
print("prediction = ", predict_1)

# Split the dataset into 4/5 for training and 1/5 for testing
x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(x, y, test_size=0.4, shuffle=True, stratify=stratify_data)
reg.fit(x_train_2, y_train_2)
predict_2 = reg.predict(x_test_2)
accuracy_2 = reg.score(x_test_2, y_test_2)
print("\n\n________Result of Split the dataset into 3/5 for training and 2/5 for testing________")
print("Accuracy : ", accuracy_2)
print("prediction = ", predict_2)
