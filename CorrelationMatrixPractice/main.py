#201835506 임동혁
#EX_Correlation Matrix with Heatmap

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
from sklearn.decomposition import PCA

housing_data = pd.read_csv('data/housing.csv')
#target column, i.e., house value
target_feature = housing_data.iloc[:, -2]
#negative value can't use in fit
negative_feature=housing_data.iloc[:,0]
negative_feature=-negative_feature


independent_feature = housing_data.iloc[:, 2:8]
categorical_value = housing_data.iloc[:, [-1]]
#define onehotencoder because of categorical value
ohe = OneHotEncoder(sparse=False)
onehot_encoder = ohe.fit(categorical_value)

#concat separated columns
new_independent_feature = pd.concat([target_feature,independent_feature], axis=1)
#hadling missing value
new_independent_feature = new_independent_feature.fillna(0)

corrmat = new_independent_feature.corr()
top_corr_features=corrmat.index
plt.figure(figsize=(20,20))
g=sns.heatmap(housing_data[top_corr_features].corr(),annot=  True,cmap="RdYlGn")
plt.show()
