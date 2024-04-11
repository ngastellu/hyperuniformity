#!/usr/bin/env python

def f(x,L):
    if size>2:
        print('Ye boi')
        return x+L
    return (x**2) * L


size = 100

y = f(1,4)
print('f(1,4) = %d'%y)

size = 1
z = f(1,4)
print('f(1,4) = %d'%z)
