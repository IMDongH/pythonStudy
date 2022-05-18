#201835506 임동혁
#EX_feature reduction using PCA

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from matplotlib import pyplot as plt
housing_data = pd.read_csv('data/housing.csv')
#target column, i.e., house value

target_feature = housing_data.iloc[:, -2]
independent_feature = housing_data.iloc[:, 2:8]
categorical_value = housing_data.iloc[:, [-1]]
#define onehotencoder because of categorical value
ohe = OneHotEncoder(sparse=False)
onehot_encoder = ohe.fit(categorical_value)
#categorical value changes to numerical value
categorical_feature = pd.DataFrame(ohe.transform(categorical_value), columns=['case1', 'case2', 'case3', 'case4', 'case5'])
print(categorical_feature)
#concat separated columns
new_independent_feature= pd.concat([independent_feature,categorical_feature], axis=1)
#hadling missing value
new_independent_feature = new_independent_feature.fillna(0)

dfcolumns = pd.DataFrame(new_independent_feature.columns)
new_independent_feature = StandardScaler().fit_transform(new_independent_feature)

#beacause of scree plot 5 is appropriate to no.of components
# pca = PCA(n_components=len(dfcolumns)-1)
# pca.fit(new_independent_feature)
# plt.plot(pca.explained_variance_,'o-')
# plt.show()

pca = PCA(5)
principalComponents = pca.fit_transform(new_independent_feature)
principalDf = pd.DataFrame(data=principalComponents,
                           columns = ['principal component1', 'principal component2'
                                      , 'principal component3', 'principal component4'
                                      , 'principal component5'])

print(principalDf)
print(target_feature)
print(pd.concat([principalDf,target_feature], axis=1))