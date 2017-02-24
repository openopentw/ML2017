#! python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 13:47:27 2017
"""

import sys
from scipy.misc import imread
from scipy.misc import imsave

IN1 = imread(sys.argv[1])
OUT = imread(sys.argv[2])

INDEX = OUT.shape
for i in range(INDEX[0]):
    for j in range(INDEX[1]):
        same = 1
        for k in range(INDEX[2]):
            if IN1[i][j][k] != OUT[i][j][k]:
                same = 0
                break
        if same == 1:
            for k in range(INDEX[2]):
                OUT[i][j][k] = 0
imsave('ans_two.png', OUT)
