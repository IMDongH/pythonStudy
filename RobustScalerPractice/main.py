# 201835506 임동혁
# RobustScaler Exercise

import numpy as np

scores = [28, 35, 26, 32, 28, 28, 35, 34, 46, 42, 37]


Q1 = np.percentile(scores,25)
Q3 = np.percentile(scores,75)
median = np.percentile(scores,75) #Calculate mean value

print("The 1st quartile  :", '%.2f' %Q1)
print("The 2nd quartile :", '%.2f' %median)
print("The 3rd quartile :", '%.2f' %Q3)
print(scores)
normalized_scores=[]
for value in scores:

    normalized_scores.append(round((value - median)/(37-Q1), 2))

print("The standard scores :", normalized_scores)

f_num=[]
count=0;
for value in normalized_scores:
    if(value<=-1.0):
        f_num.append(count)
    count+=1

for count in f_num:
    print("Student who will recieve F score: ",scores[count])