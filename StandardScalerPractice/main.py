# 201835506 임동혁
# StandardScler Exercise

import numpy as np

scores = [28, 35, 26, 32, 28, 28, 35, 34, 46, 42, 37]

mean = np.mean(scores) #Calculate mean value
std = np.std(scores) #Calculate standard diviation value

print("The mean :", '%.2f' %np.mean(scores))
print("The standard deviation :", '%.2f' %np.std(scores))
normalized_scores=[]
for value in scores:

    normalized_scores.append(round((value - mean)/std, 2))

print("The standard scores :", normalized_scores)

f_num=[]
count=0;
for value in normalized_scores:
    if(value<=-1.0):
        f_num.append(count)
    count+=1

for count in f_num:
    print("Student who will recieve F score: ",scores[count])

