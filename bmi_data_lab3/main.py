# 201835506 임동혁
# lab3

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import preprocessing, linear_model

data = pd.read_csv('data/bmi_data_lab3.csv')
nan_data = data
print("***** statistical data *****")
print(data.describe())

print("***** feature name *****")
for col in data.columns:
    print(col)

print("***** Data type *****")
print(data.dtypes)

# plot height&weight histograms(bins=10) for each BMI value
print("*****height hitograms*****")
g = sns.FacetGrid(data, col='BMI')
g.map(plt.hist, 'Height (Inches)', bins=10)
plt.show()

print("*****weight hitograms*****")
g = sns.FacetGrid(data, col='BMI')
g.map(plt.hist, 'Weight (Pounds)', bins=10)
plt.show()

# Preparing plot scaling results for height and weight
HW_data = \
    pd.DataFrame({'Height (Inches)': data['Height (Inches)'],
                  'Weight (Pounds)': data['Weight (Pounds)']})

# # Standard Scaler
# Standard_scaler = preprocessing.StandardScaler()
# Standard_scaled_df = Standard_scaler.fit_transform(HW_data)
# Standard_scaled_df = pd.DataFrame(Standard_scaled_df, columns=['Height (Inches)', 'Weight (Pounds)'])
# fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 5))
# ax1.set_title('Before Scaling')
# sns.kdeplot(HW_data['Height (Inches)'], ax=ax1)
# sns.kdeplot(HW_data['Weight (Pounds)'], ax=ax1)
# ax2.set_title('After Standard Scaler')
# sns.kdeplot(Standard_scaled_df['Height (Inches)'], ax=ax2)
# sns.kdeplot(Standard_scaled_df['Weight (Pounds)'], ax=ax2)
# plt.show()
#
# # MinMax Scaler
# MIN_MAX_scaler = preprocessing.MinMaxScaler()
# MIN_MAX_scaled_df = MIN_MAX_scaler.fit_transform(HW_data)
# MIN_MAX_scaled_df = pd.DataFrame(MIN_MAX_scaled_df, columns=['Height (Inches)', 'Weight (Pounds)'])
# fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 5))
# ax1.set_title('Before Scaling')
# sns.kdeplot(HW_data['Height (Inches)'], ax=ax1)
# sns.kdeplot(HW_data['Weight (Pounds)'], ax=ax1)
# ax2.set_title('After MINMAX Scaler')
# sns.kdeplot(MIN_MAX_scaled_df['Height (Inches)'], ax=ax2)
# sns.kdeplot(MIN_MAX_scaled_df['Weight (Pounds)'], ax=ax2)
# plt.show()
#
# # Robust Scaler
# Robust_scaler = preprocessing.RobustScaler()
# Robust_scaled_df = MIN_MAX_scaler.fit_transform(HW_data)
# Robust_scaled_df = pd.DataFrame(Robust_scaled_df, columns=['Height (Inches)', 'Weight (Pounds)'])
# fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 5))
# ax1.set_title('Before Scaling')
# sns.kdeplot(HW_data['Height (Inches)'], ax=ax1)
# sns.kdeplot(HW_data['Weight (Pounds)'], ax=ax1)
# ax2.set_title('After Robust Scaler')
# sns.kdeplot(Robust_scaled_df['Height (Inches)'], ax=ax2)
# sns.kdeplot(Robust_scaled_df['Weight (Pounds)'], ax=ax2)
# plt.show()

# Identify all dirty records with likely-wrong or missing height or weight values
MissingValue = [0, -104.4205547, 592.6244266, 664.4877548, 622.0486612, -141.8241248, 665.4650594,
                -130.9261617, -161.9949135, 1110.621115]
# Check number of NAN value before replace NAN value
print("*****check before replace dirty record to NAN value*****")
print(data.isna().sum())

# Replace dirty record to NAN value
data.replace(MissingValue, np.nan, inplace=True)

# Check number of NAN value after replace dirty record to NAN value
print("*****check after replace dirty record to NAN value*****")
print(data.isna().sum())

# Drop all rows with NAN
data_drop = data.dropna(axis=0, how='any')

print("*****check NAN for each row column*****")
print(data.isna().sum(axis=0))
print("*****check NAN for each row*****")
print(data.isna().sum(axis=1))

# Fill NAN value to mean value
print('*****Fill NAN value to  mean value*****')
print(data_drop.mean(numeric_only=True))
print(data.fillna(data_drop.mean(numeric_only=True)))

# Fill NAN value to median value
print('*****Fill NAN value to median value*****')
print(data_drop.median(numeric_only=True))
print(data.fillna(data_drop.median(numeric_only=True)))

# Fill NAN using ffill
print('*****Fill NAN using ffill*****')
print(data.fillna(axis=0, method='ffill'))

# Fill NAN using bfill
print('*****Fill NAN using bfill*****')
print(data.fillna(axis=0, method='bfill'))



X = data_drop['Height (Inches)']
Y = data_drop['Weight (Pounds)']

Nan_weight = data[data['Weight (Pounds)'].isna()]
Nan_height = data[data['Height (Inches)'].isna()]

reg = linear_model.LinearRegression()
reg.fit(X.values.reshape(-1, 1), Y)
plt.scatter(X,Y,color = 'blue')


hx=np.array([data_drop['Height (Inches)'].min()-1,data_drop['Height (Inches)'].max()+1])
hy=reg.predict(hx[:,np.newaxis])
plt.plot(hx,hy,color = 'black')


pred_W_x = Nan_weight['Height (Inches)'].values.reshape(-1,1)
pred_W_y = reg.predict(pred_W_x)
plt.scatter(pred_W_x,pred_W_y,color = 'red')

pred_H_x = (Nan_height['Weight (Pounds)']-reg.predict([[0]]))/reg.coef_
pred_H_y = reg.predict(pred_H_x.values.reshape(-1, 1))
plt.scatter(pred_H_x, pred_H_y, color='red')

plt.title("Linear Regression")
plt.xlabel("Height (Inches)")
plt.ylabel("Weight (Pounds)")
plt.show()

# group by female
female = data_drop[data_drop['Sex'] == 'Female']
Nan_female_x = Nan_weight[Nan_weight['Sex'] == 'Female']
Nan_female_y = Nan_height[Nan_height['Sex'] == 'Female']
female_X = female['Height (Inches)']
female_Y = female['Weight (Pounds)']
f_reg = linear_model.LinearRegression()
f_reg.fit(female_X.values.reshape(-1, 1), female_Y)
plt.scatter(female_X, female_Y,color='blue')
pred_F_W_x = Nan_female_x['Height (Inches)']
pred_F_W_y = f_reg.predict(pred_F_W_x.values.reshape(-1, 1))
pred_F_H_x=(Nan_female_y['Weight (Pounds)']-f_reg.predict([[0]]))/f_reg.coef_
pred_F_H_y = f_reg.predict(pred_F_H_x.values.reshape(-1, 1))
plt.plot(female_X, f_reg.predict(female_X.values.reshape(-1,1)),color='black')
plt.scatter(pred_F_W_x, pred_F_W_y, color='red')
plt.scatter(pred_F_H_x, pred_F_H_y, color='red')
plt.title("female  Linear Regression ")
plt.xlabel("Height (Inches)")
plt.ylabel("Weight (Pounds)")
plt.show()

# group by male
male = data_drop[data_drop['Sex'] == 'Male']
Nan_male_x = Nan_weight[Nan_weight['Sex'] == 'Male']
Nan_male_y = Nan_height[Nan_height['Sex'] == 'Male']
male_X = male['Height (Inches)']
male_Y = male['Weight (Pounds)']
m_reg = linear_model.LinearRegression()
m_reg.fit(male_X.values.reshape(-1, 1), male_Y)
plt.scatter(male_X, male_Y,color='blue')
pred_M_W_x = Nan_male_x['Height (Inches)']
pred_M_W_y = f_reg.predict(pred_M_W_x.values.reshape(-1, 1))
pred_M_H_x=(Nan_male_y['Weight (Pounds)']-f_reg.predict([[0]]))/f_reg.coef_
pred_M_H_y = f_reg.predict(pred_M_H_x.values.reshape(-1, 1))
plt.plot(male_X, f_reg.predict(male_X.values.reshape(-1,1)),color='black')
plt.scatter(pred_M_W_x, pred_M_W_y, color='red')
plt.scatter(pred_M_H_x, pred_M_H_y, color='red')
plt.title("Male Linear Regression ")
plt.xlabel("Height (Inches)")
plt.ylabel("Weight (Pounds)")
plt.show()

# group by BMI (2)
BMI2 = data_drop[data_drop['BMI'] == 2]
Nan_BMI2_x = Nan_weight[Nan_weight['BMI'] == 2]
Nan_BMI2_y = Nan_height[Nan_height['BMI'] == 2]
BMI2_reg = linear_model.LinearRegression()
BMI2_X = BMI2['Height (Inches)']
BMI2_Y = BMI2['Weight (Pounds)']
BMI2_reg.fit(BMI2_X.values.reshape(-1, 1), BMI2_Y)
plt.scatter(BMI2_X, BMI2_Y)

pred_B2_W_x = Nan_BMI2_x['Height (Inches)'].values.reshape(-1, 1)
pred_B2_W_y = BMI2_reg.predict(pred_B2_W_x)

pred_B2_H_x = (Nan_BMI2_y['Weight (Pounds)']-BMI2_reg.predict([[0]]))/BMI2_reg.coef_
pred_B2_H_y = BMI2_reg.predict(pred_B2_H_x.values.reshape(-1, 1))

plt.plot(BMI2_X, BMI2_reg.predict(BMI2_X.values.reshape(-1,1)), color='black')
plt.scatter(pred_B2_W_x, pred_B2_W_y ,color='red')
plt.scatter(pred_B2_H_x, pred_B2_H_y, color='red')
plt.title("BMI2 Linear Regression ")
plt.xlabel("Height (Inches)")
plt.ylabel("Weight (Pounds)")
plt.show()

# group by BMI (3)
BMI3 = data_drop[data_drop['BMI'] == 3]
Nan_BMI3_x = Nan_weight[Nan_weight['BMI'] == 3]
Nan_BMI3_y = Nan_height[Nan_height['BMI'] == 3]
BMI3_reg = linear_model.LinearRegression()
BMI3_X = BMI3['Height (Inches)']
BMI3_Y = BMI3['Weight (Pounds)']
BMI3_reg.fit(BMI3_X.values.reshape(-1, 1), BMI3_Y)
plt.scatter(BMI3_X, BMI3_Y)

pred_B3_W_x = Nan_BMI3_x['Height (Inches)'].values.reshape(-1, 1)
pred_B3_W_y = BMI3_reg.predict(pred_B3_W_x)

pred_B3_H_x = (Nan_BMI3_y['Weight (Pounds)']-BMI3_reg.predict([[0]]))/BMI3_reg.coef_
pred_B3_H_y = BMI3_reg.predict(pred_B3_H_x.values.reshape(-1, 1))

plt.plot(BMI3_X, BMI3_reg.predict(BMI3_X.values.reshape(-1,1)), color='black')
plt.scatter(pred_B3_W_x, pred_B3_W_y ,color='red')
plt.scatter(pred_B3_H_x, pred_B3_H_y, color='red')
plt.title("BMI3 Linear Regression ")
plt.xlabel("Height (Inches)")
plt.ylabel("Weight (Pounds)")
plt.show()

