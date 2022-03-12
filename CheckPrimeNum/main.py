# 201835506 임동혁
# EX_1
import math


def CheckPrimeN(Pnum):
    check = True

    for i in range(2, int(math.sqrt(Pnum)) + 1):
        if Pnum % i == 0:
            check=False
            break;

    return check


number = 0
while True:

    if number < 2 or number > 32767:
        number = int(input("Enter a number between 2 and 32767 : "))
    else :
        if CheckPrimeN(number)==True:
            print(number,' is prime number!')
        else:
            print(number,' is not prime number')
        break


