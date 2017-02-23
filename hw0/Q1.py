#! python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 13:37:27 2017
"""

import sys
import numpy as np

FILE_A = open(sys.argv[1], "r")
A = []
for line in FILE_A:
    number_strings = line.split(',')
    numbers = [int(n) for n in number_strings]
    A.append(numbers)

FILE_B = open(sys.argv[2], "r")
B = []
for line in FILE_B:
    number_strings = line.split(',')
    numbers = [int(n) for n in number_strings]
    B.append(numbers)

C = np.dot(A, B)
C = np.reshape(C, C.size)
C = np.sort(C)
np.savetxt('ans_one.txt', C, delimiter='\n', fmt='%d')
