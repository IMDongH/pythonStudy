import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from mpl_toolkits.mplot3d import Axes3D
dir = "data/flights.xlsx"
data = pd.read_excel(dir)
practice = data.loc[data['flight'] == 1,:]
data = data.loc[data['flight'] == 1, ["flight", 'time', 'battery_current', 'position_x', 'position_y', 'position_z']]
# plt.plot(data['position_z'], data['battery_current'])
# plt.show()
# print(data)
# robustScaler = RobustScaler()
# print(robustScaler.fit(data))
# train_data_robustScaled = robustScaler.transform(data)
# print(train_data_robustScaled)

# fig = plt.figure(figsize=(9, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(data['position_x'], data['position_y'], data['position_z'])
# plt.show()
#
# fig = plt.figure(figsize=(9, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(practice['velocity_x'], practice['velocity_y'], practice['velocity_z'])
# plt.show()

# fig = plt.figure(figsize=(9, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(practice['angular_x'], practice['angular_y'], practice['angular_z'])
# plt.show()
from sklearn.preprocessing import RobustScaler
robustScaler = RobustScaler()
print(robustScaler.fit(train_data))
train_data_robustScaled = robustScaler.transform(train_data)