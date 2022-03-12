#201835506 임동혁
#EX_2

def makeDict(K, V):
    D = {}
    for i in range(0,len(K)):
        D[K[i]] = V[i]

    return D


key = ('Korean', 'Mathematics', 'English')
value = (90.3, 85.5, 92.7)

dic = makeDict(key, value)

for key, value in dic.items():
    print(key, value)

