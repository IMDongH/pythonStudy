# 201835506 임동혁
# Pandas exercise

import numpy as np
import pandas as pd

npData = np.array([3., '?', 2., 5., '*', 4., 5., 6., '+', 3., 2., '&', 5., '?', 7., '!']).reshape(4, 4)

# Create a Pandas (4, 4) DataFrame
df = pd.DataFrame(
    npData
)

str = ['?', '*', '+', '&', '!']

print(df)

# replace non numeric data
df.replace(str, np.nan, inplace=True)

print(df)
# Display the DataFrame replaced non numeric data

print(df.isna().sum())
# Display the DataFrame isna with any, and sum

print(df.dropna(axis=0, how='all'))
# Display the DataFrame dropna how is all

print(df.dropna(axis=0, how='any'))
# Display the DataFrame dropna how is any

print(df.dropna(axis=0, thresh=1))
# Display the DataFrame dropna thresh is 1

print(df.dropna(axis=0, thresh=2))
# Display the DataFrame dropna thresh is 2

df = df.apply(pd.to_numeric)
# make dataframe to numerical data to get mean and median value

print(df.fillna(100))
# Display the DataFrame fill nan value to 100

print(df.fillna(df.mean()))
# Display the DataFrame fill nan value to mean value

print(df.fillna(df.median()))
# Display the DataFrame fill nan value to median value

print(df.fillna(axis=0, method='ffill'))
# Display the DataFrame replaced missing values with the values in the previous row

print(df.fillna(axis=0, method='bfill'))
# Display the DataFrame replaced missing values with the values in the backward row
