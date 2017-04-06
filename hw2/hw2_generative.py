#! python3
"""
@author: b04902053
"""

# import# {{{
import pandas
import numpy as np
# from numpy.linalg import inv
import math
import sys# }}}

# Argvs# {{{
DATA_FILE = "data/X_train.csv"
ANS_FILE = "data/Y_train.csv"
TEST_FILE = "data/X_test.csv"
OUTPUT = "submission.csv"
DATA_FILE = sys.argv[1]
ANS_FILE = sys.argv[2]
TEST_FILE = sys.argv[3]
OUTPUT = sys.argv[4]
# }}}

# Input Data# {{{
ALL_DATA = pandas.read_csv(DATA_FILE, encoding='big5')
ALL_DATA = ALL_DATA.values
ALL_ANS = np.genfromtxt(ANS_FILE, delimiter=',')# }}}

VALI = 0
if VALI == 0:
    DEBUG = 0
elif VALI == 1:
    SLICE = int(ALL_DATA.shape[0] * 2/3)
# slice validation set# {{{
if VALI == 0:
    DATA = ALL_DATA
    ANS = ALL_ANS
elif VALI == 1:
    DATA = ALL_DATA[ : SLICE]
    ANS = ALL_ANS[ : SLICE]
    VALI_DATA = ALL_DATA[SLICE : ]
    VALI_ANS = ALL_ANS[SLICE : ]
# }}}

# BIG SELECTs# {{{
S_workclass = list(range(6, 15))
S_education = list(range(15, 31))
S_marital_status = list(range(31, 38))
S_occupation = list(range(38, 53))
S_relation = list(range(53, 59))
S_race = list(range(59, 64))
S_native_country = list(range(64, 105))
# }}}

# SELECTs# {{{
SELECT = np.array(list(range(DATA.shape[1])))
# SELECT = np.array([0,2,3,4,5] + S_workclass + S_education + S_marital_status + S_occupation + S_relation + S_race + [102])
# }}}

RAND = 0
if RAND == 1:
    SEED = 3
# remove trash data# {{{
if RAND == 1:
    np.random.seed(SEED)
    np.random.shuffle(SELECT)
DATA = DATA[:,SELECT]# }}}

NORM = 1
# Normalize# {{{
if NORM == 1:
    mean = np.mean(DATA, axis=0)
    # TODO: delete - mean
    std = np.std(DATA - mean, axis=0)
    DATA = (DATA - mean) / std# }}}

# Calc SIGMA# {{{
CLASS = [0, 0]
MEAN = [0, 0]
SIGMA = 0
for i in range(2):
    CLASS[i] = DATA[ANS == i,:]
    MEAN[i] = np.mean(CLASS[i], axis=0).reshape((CLASS[i].shape[1], 1))
    diff = CLASS[i] - MEAN[i].T
    sums = np.zeros((diff.shape[1], diff.shape[1]))
    for j in range(diff.shape[0]):
        tmp = diff[j,:].reshape((diff.shape[1], 1))
        sums += np.dot(tmp, tmp.T)
    SIGMA += sums
SIGMA /= DATA.shape[0]# }}}

# Calc W# {{{
INV_SIGMA = np.linalg.inv(SIGMA)
W = (MEAN[0] - MEAN[1]).T.dot(INV_SIGMA).T# }}}

# Calc b # {{{
# b = 1/2 * (MEAN[0].T * INV_SIGMA * MEAN[0] - MEAN[1].T * INV_SIGMA * MEAN[1]) + math.log(CLASS[0].shape[1] / CLASS[1].shape[1])
b = 0
for i in range(2):
    b += ((-1)**(1+i)) * MEAN[i].T.dot(INV_SIGMA).dot(MEAN[i])
b /= 2
b += math.log(CLASS[0].shape[0] / CLASS[1].shape[0])
b = b[0][0]# }}}

# Input TEST# {{{
if VALI == 0:
    TEST = np.genfromtxt(TEST_FILE, delimiter=',')[1:]
elif VALI == 1:
    TEST = VALI_DATA
TEST = TEST[:,SELECT]# }}}

# Normalize TEST# {{{
if NORM == 1:
    TEST = (TEST - mean) / std# }}}

# Calc ans# {{{
z = TEST.dot(W) + b
z = z.reshape(z.size)
ans = np.ones(TEST.shape[0])
ans[z > 0] = 0
# ans[z < 0] = 1
ans = ans.astype(int)
# }}}

# [Addition] Calc & Print Sol Rate# {{{
if VALI == 1:
    same = np.zeros(TEST.shape[0])
    same[ans == VALI_ANS] = 1
    same = same.astype(int)
    print(np.sum(same) / TEST.shape[0])
# elif VALI == 0:
    # print(np.sum(ans) / TEST.shape[0])
# }}}

# save to 'submission.csv'# {{{
f = open(OUTPUT, "w+")
print("id,label", file = f, end = "\n")
for i in range(ans.size):
    if VALI == 0:
        if DEBUG == 0:
            print(str(i+1) + "," + str(ans[i]), file = f, end = "\n")
        elif DEBUG == 1:
            print(str(i+1) + "," + str(ans[i]) + "," + str(z[i]), file = f, end = "\n")
    elif VALI == 1:
        print(str(i+1) + "," + str(ans[i]) + "," + str(VALI_ANS[i]) + "," + str(same[i]) + "," + str(z[i]), file = f, end = "\n")
f.close()# }}}
