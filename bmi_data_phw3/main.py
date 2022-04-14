# 201835506 임동혁 201735819 김용겸 201735834 박정영 2018 김한뫼
# phw3

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import linear_model
import seaborn as sns
import matplotlib.pyplot as plt

dir = "data/bmi_data_phw3.xlsx"

df = pd.read_excel(dir)
df.rename(columns={'Height (Inches)': 'Height', 'Weight (Pounds)': 'Weight'}, inplace=True)

# print dataframe stataistical data
print(df.info())
print(df.describe())

# Graph Output for Each BMI
# BMI level 0
plt.figure(0)
plt.subplot(211)
plt.title('extremely weak (BMI = 0)')
plt.hist(df[df['BMI'].isin([0])]['Height'], bins=10)
plt.xlabel('Height')
plt.ylabel('number of people')

plt.subplot(212)
plt.hist(df[df['BMI'].isin([0])]['Weight'], bins=10)
plt.xlabel('Weight')
plt.ylabel('number of people')

# BMI level 1
plt.figure(1)
plt.subplot(211)
plt.title('weak (BMI = 1)')
plt.hist(df[df['BMI'].isin([1])]['Height'], bins=10)
plt.xlabel('Height')
plt.ylabel('number of people')

plt.subplot(212)
plt.hist(df[df['BMI'].isin([1])]['Weight'], bins=10)
plt.xlabel('Weight')
plt.ylabel('number of people')

# BMI level 2
plt.figure(2)
plt.subplot(211)
plt.title('normal (BMI = 2)')
plt.hist(df[df['BMI'].isin([2])]['Height'], bins=10)
plt.xlabel('Height')
plt.ylabel('number of people')

plt.subplot(212)
plt.hist(df[df['BMI'].isin([2])]['Weight'], bins=10)
plt.xlabel('Weight')
plt.ylabel('number of people')

# BMI level 3
plt.figure(3)
plt.subplot(211)
plt.title('overweight (BMI = 3)')
plt.hist(df[df['BMI'].isin([3])]['Height'], bins=10)
plt.xlabel('Height')
plt.ylabel('number of people')

plt.subplot(212)
plt.hist(df[df['BMI'].isin([3])]['Weight'], bins=10)
plt.xlabel('Weight')
plt.ylabel('number of people')

# BMI level 4
plt.figure(4)
plt.subplot(211)
plt.title('obesity (BMI = 4)')
plt.hist(df[df['BMI'].isin([4])]['Height'], bins=10)
plt.xlabel('Height')
plt.ylabel('number of people')

plt.subplot(212)
plt.hist(df[df['BMI'].isin([4])]['Weight'], bins=10)
plt.xlabel('Weight')
plt.ylabel('number of people')

plt.show()

# height and weight scaling
scaler = [preprocessing.StandardScaler(), preprocessing.MinMaxScaler(), preprocessing.RobustScaler()]
scaled_df = [0, 1, 2]  # Declaring a list of size 3  / 0, 1, 2 are meaningless

for x in range(3):
    scaled_df[x] = scaler[x].fit_transform(df.loc[:, ['Height', 'Weight']])  # Use only Height and weight data
    scaled_df[x] = pd.DataFrame(scaled_df[x], columns=['Height', 'Weight'])
"""
ax1 -> before Scaling
ax2 -> standard Scaling
ax3 -> MINMAX Scaling
ax4 -> Robust Scaling
"""
fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(12, 5))

# Print Graph each Scaling Result
ax1.set_title('Before Scaling')
sns.kdeplot(df['Height'], ax=ax1)
sns.kdeplot(df['Weight'], ax=ax1)

ax2.set_title('After Standard Scaler')
sns.kdeplot(scaled_df[0]['Height'], ax=ax2)
sns.kdeplot(scaled_df[0]['Weight'], ax=ax2)

ax3.set_title('After MINMAX Scaler')
sns.kdeplot(scaled_df[1]['Height'], ax=ax3)
sns.kdeplot(scaled_df[1]['Weight'], ax=ax3)

ax4.set_title('After Robust Scaler')
sns.kdeplot(scaled_df[2]['Height'], ax=ax4)
sns.kdeplot(scaled_df[2]['Weight'], ax=ax4)

plt.show()

####### All data Linear Regression #######
E = linear_model.LinearRegression()
E.fit(df['Height'].values.reshape(-1, 1), df['Weight'])

pw = E.predict((df['Height']).values.reshape(-1,1))
e = df['Weight'] - pw

z = (e - e.mean()) / e.std()

# print Ze Histogram
plt.hist(z, bins=10)
plt.xlabel('Ze_all')
plt.ylabel('frequency')
plt.show()

# setting a
a = 1.46  # After several attempts, this value was the best.
est_BMI = z.copy()
# Estimate BMI
for i in range(len(est_BMI)):
    if est_BMI[i] < -a:  # z < -a -> BMI = 0
        est_BMI[i] = 0
    elif est_BMI[i] > a:  # z > a -> BMI = 4
        est_BMI[i] = 4
    else:  # other is nan
        est_BMI[i] = np.nan

# add Esitmate BMI for dataframe
df['Estimate BMI'] = est_BMI
# make mask Actual BMI 0,4 and estimated BMI 0,4
df_mask = df['BMI'].isin([0, 4]) | df['Estimate BMI'].isin([0, 4])
print(df[df_mask])
correct = 0
count = 0
for i in range(len(df)):
    if df.loc[i, 'BMI'] == 0 or df.loc[i, 'BMI'] == 4:
        count += 1
        if df.loc[i, 'BMI'] == df.loc[i, 'Estimate BMI']:
            correct += 1

print("alpha: ", a, "Accuracy: ", correct/count)

# Esitmate BMI reset
df['Estimate BMI'] = np.nan

####### male and female linear Regression #######
####### Male
E_m = linear_model.LinearRegression()
df_m = df[df['Sex'].isin(['Male'])].copy()  # Male for dataframe
E_m.fit((df_m['Height']).values.reshape(-1,1), df_m['Weight'])

pw_m = E_m.predict((df_m['Height']).values.reshape(-1,1))
e_m = df_m['Weight'] - pw_m

z_m = (e_m - e_m.mean()) / e_m.std()

####### Female
E_f = linear_model.LinearRegression()
df_f = df[df['Sex'].isin(['Female'])].copy()  # Female for dataframe
E_f.fit((df_f['Height']).values.reshape(-1,1), df_f['Weight'])

pw_f = E_f.predict((df_f['Height']).values.reshape(-1,1))
e_f = df_f['Weight'] - pw_f

z_f = (e_f - e_f.mean()) / e_f.std()

# Male Histogram
plt.figure(figsize=(6, 9))
plt.subplot(211)
plt.hist(z_m, bins=10)
plt.xlabel('Ze_Male')
plt.ylabel('frequency')
# Female Histogram
plt.subplot(212)
plt.hist(z_f, bins=10)
plt.xlabel('Ze_Female')
plt.ylabel('frequency')

plt.show()

# setting a for Male
a_m = 1.32  # After several attempts, this value was the best.
est_BMI_m = np.array(z_m)
# Estimate BMI for Male
for i in range(len(est_BMI_m)):
    if est_BMI_m[i] < -a_m:  # z < -a -> BMI = 0
        est_BMI_m[i] = 0
    elif est_BMI_m[i] > a_m:  # z > a -> BMI = 4
        est_BMI_m[i] = 4
    else:  # other is nan
        est_BMI_m[i] = np.nan

df_m['Estimate BMI'] = est_BMI_m
# make mask Actual BMI 0,4 and estimated BMI 0,4 in Male
df_mask_m = df_m['BMI'].isin([0, 4]) | df_m['Estimate BMI'].isin([0, 4])
print("Male\n", df_m[df_mask_m])
# calculate accuracy
correct = 0
count = 0
for i in range(len(df_m.index)):
    if df_m.loc[df_m.index[i], 'BMI'] == 0 or df_m.loc[df_m.index[i], 'BMI'] == 4:
        count += 1
        if df_m.loc[df_m.index[i], 'BMI'] == df_m.loc[df_m.index[i], 'Estimate BMI']:
            correct += 1
print("alpha : ", a_m, "Accuracy: ", correct/count)

# setting a for Female
a_f = 1.77  # After several attempts, this value was the best.
est_BMI_f = np.array(z_f)
# Estimate BMI for Female
for i in range(len(est_BMI_f)):
    if est_BMI_f[i] < -a_f:  # z < -a -> BMI = 0
        est_BMI_f[i] = 0
    elif est_BMI_f[i] > a_f:  # z > a -> BMI = 4
        est_BMI_f[i] = 4
    else:  # other is nan
        est_BMI_f[i] = np.nan

df_f['Estimate BMI'] = est_BMI_f
# make mask Actual BMI 0,4 and estimated BMI 0,4 in Female
df_mask_f = df_f['BMI'].isin([0, 4]) | df_f['Estimate BMI'].isin([0, 4])
print("Female\n", df_f[df_mask_f])

# calculate accuracy
correct = 0
count = 0
for i in range(len(df_f.index)):
    if df_f.loc[df_f.index[i], 'BMI'] == 0 or df_f.loc[df_f.index[i], 'BMI'] == 4:
        count += 1
        if df_f.loc[df_f.index[i], 'BMI'] == df_f.loc[df_f.index[i], 'Estimate BMI']:
            correct += 1
print("alpha : ", a_f, "Accuracy: ", correct/count)
