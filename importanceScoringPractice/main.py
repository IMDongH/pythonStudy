#201835506 임동혁
#EX_importanceScoring

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import ExtraTreeClassifier

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

#concat separated columns
new_independent_feature= pd.concat([independent_feature,categorical_feature], axis=1)
#hadling missing value
new_independent_feature = new_independent_feature.fillna(0)

model=ExtraTreeClassifier()
model.fit(new_independent_feature,target_feature)
print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=new_independent_feature.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()
