#201835506 임동혁
#EX_Univariate selection

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import OneHotEncoder

housing_data = pd.read_csv('data/housing.csv')
#target column, i.e., house value
target_feature = housing_data.iloc[:, -2]
#negative value can't use in fit
negative_feature=housing_data.iloc[:,0]
negative_feature=-negative_feature


independent_feature = housing_data.iloc[:, 1:8]
categorical_value = housing_data.iloc[:, [-1]]
#define onehotencoder because of categorical value
ohe = OneHotEncoder(sparse=False)
onehot_encoder = ohe.fit(categorical_value)
#categorical value changes to numerical value
categorical_feature = pd.DataFrame(ohe.transform(categorical_value), columns=['case1', 'case2', 'case3', 'case4', 'case5'])

#concat separated columns
new_independent_feature = pd.concat([negative_feature,independent_feature], axis=1)
new_independent_feature= pd.concat([new_independent_feature,categorical_feature], axis=1)
#hadling missing value
new_independent_feature = new_independent_feature.fillna(0)

bestfeatures = SelectKBest(score_func=chi2, k=13)
fit = bestfeatures.fit(new_independent_feature, target_feature)
dfcolumns = pd.DataFrame(new_independent_feature.columns)
dfscores = pd.DataFrame(fit.scores_)
featureScores = pd.concat([dfcolumns, dfscores],axis=1)
featureScores.columns = ['Specs','value']
print(featureScores.nlargest(10,'value'))
