# This is a sample Python script.
import math


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def CheckPrimeN(Pnum):
    for i in range(2, int(math.sqrt(Pnum)) + 1):
        if Pnum % i == 0:
            print(str(i))
            return
        else:
            if i == int(math.sqrt(Pnum)):
                print(str(Pnum) + " is prime number")
            continue


number = input("Type a number : ")
CheckPrimeN(int(number))
