# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


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

