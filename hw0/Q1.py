# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 13:37:27 2017
"""

import sys
import numpy as np
from numpy import genfromtxt

A = genfromtxt(sys.argv[1], delimiter=',')
B = genfromtxt(sys.argv[2], delimiter=',')

C = np.dot(A, B)
C = np.reshape(C, C.size)
C = np.sort(C)
np.savetxt('ans_one.txt', C, delimiter='\n', fmt='%d')
