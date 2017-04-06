#! python3
"""
@author: b04902053
"""

# import# {{{
import pandas
import numpy as np
from numpy.linalg import inv
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

# SELECTs# {{{
# BIG SELECTs# {{{
S_workclass = list(range(6, 15))
S_education = list(range(15, 31))
S_marital_status = list(range(31, 38))
S_occupation = list(range(38, 53))
S_relation = list(range(53, 59))
S_race = list(range(59, 64))
S_native_country = list(range(64, 105))
# }}}
SELECT = np.array(list(range(DATA.shape[1])))
N_SELECT = SELECT.size #}}}

# Select DATA# {{{
DATA = DATA[:,SELECT]# }}}

dim = 3
DIM = dim
# Add DIM# {{{
OLD_DATA = DATA
DATA = np.zeros((OLD_DATA.shape[0], OLD_DATA.shape[1]*dim))
for i in range(dim):
    DATA[:,OLD_DATA.shape[1]*i:OLD_DATA.shape[1]*(i+1)] = OLD_DATA ** (i+1)
# }}}

NORM = 1
# Normalize# {{{
if NORM == 1:
    mean = np.mean(DATA, axis=0)
    std = np.std(DATA, axis=0)
    DATA = (DATA - mean) / std# }}}

SPECIAL_DIM = 1
# Add special DIM# {{{
if SPECIAL_DIM == 1:
    old_mean = np.mean(OLD_DATA, axis=0)
    old_std = np.std(OLD_DATA, axis=0)
    OLD_DATA = (OLD_DATA - old_mean) / old_std
    DIM += 1
    DATA = np.append(DATA, np.sin(OLD_DATA), axis=1)
# }}}

LMBD = 0
# LMBD = 1e2
YETA = 1

X_slice = 0
if X_slice == 1:
    Xslice = int(DATA.shape[0] * 2/3)
# slice X# {{{
if X_slice == 0:
    X = DATA
    ans = ANS.reshape((X.shape[0], 1))
    test_X = DATA
    test_ans = ANS
elif X_slice == 1:
    X = DATA[ : Xslice]
    ans = ANS[ : Xslice].reshape((X.shape[0], 1))
    test_X = DATA[Xslice : ]
    test_ans = ANS[Xslice : ]

inv_test_ans = 1 - test_ans
plus_test_ans = 1 + test_ans
num_test = test_ans.shape[0]
# }}}

b = 0
W = np.zeros((1, N_SELECT*DIM))
# INIT Training# {{{
ROOT_SUM_b = 0
ROOT_SUM_W = np.zeros((1, N_SELECT*DIM))

LOSS = 0
LAST_LOSS = 0

loss_count = 0
# }}}

# TODO: Add batch size
PRINT_LOSS = 0
TRAIN_RANGE = 5000
# Train# {{{
for i in range(TRAIN_RANGE):
    # gradient
    z = b + np.dot(X, W.T)
    half_loss = ans - 1 / (1 + np.exp(-z))
    grad_b = 2 * np.sum(half_loss)
    grad_W = 2 * (np.sum(half_loss * X, axis=0) + LMBD * W)

    # learning_rate
    ROOT_SUM_b += grad_b ** 2
    rate_b = YETA / np.sqrt(ROOT_SUM_b)
    ROOT_SUM_W += grad_W ** 2
    rate_W = YETA / np.sqrt(ROOT_SUM_W)

    # descent
    b += rate_b * grad_b
    W += rate_W * grad_W

    # loss
    # if X_slice == 1:
        # # binary loss# {{{
        # z = (test_X.dot(W.T) + b).reshape(test_X.shape[0])
        # z_ans = np.ones(test_X.shape[0], dtype=int)
        # z_ans[z < 0] = 0
        # same = np.zeros(test_X.shape[0], dtype=int)
        # same[z_ans == test_ans] = 1
        # LOSS += np.sum(same) / num_test# }}}

        # # cross entropy# {{{
        # z = (test_X.dot(W.T) + b).reshape(test_X.shape[0])
        # expz = np.exp(-z)
        # LOSS = - np.average(inv_test_ans * np.log(expz) - plus_test_ans * np.log(1+expz))
        # # }}}

        # # loss_count# {{{
        # loss_count += 1
        # if loss_count == 50:
            # LOSS /= 50
            # print(i, LOSS)
            # LAST_LOSS = LOSS
            # loss_count = 0
            # LOSS = 0# }}}

    # print(i)

    # if PRINT_LOSS == 1:

    # if i > 100 and LOSS > LAST_LOSS:
        # break

# print(i, np.sqrt(LOSS/num_test))
# }}}

# TESTing DATA# {{{
# Input TEST# {{{
if VALI == 0:
    TEST = np.genfromtxt(TEST_FILE, delimiter=',')[1:]
elif VALI == 1:
    TEST = VALI_DATA
TEST = TEST[:,SELECT]# }}}

# Add DIM# {{{
OLD_TEST = TEST
TEST = np.zeros((OLD_TEST.shape[0], OLD_TEST.shape[1]*dim))
for i in range(dim):
    TEST[:,OLD_TEST.shape[1]*i:OLD_TEST.shape[1]*(i+1)] = OLD_TEST ** (i+1)# }}}

# Normalize TEST# {{{
if NORM == 1:
    TEST = (TEST - mean) / std# }}}

# Add special DIM# {{{
if SPECIAL_DIM == 1:
    OLD_TEST = (OLD_TEST - old_mean) / old_std
    TEST = np.append(TEST, np.sin(OLD_TEST), axis=1)
# }}}
# }}}

# Calc ans# {{{
z = (TEST.dot(W.T) + b).reshape(TEST.shape[0])
ans = np.ones(TEST.shape[0], dtype=int)
ans[z < 0] = 0
# }}}

# [Addition] Calc & Print Sol Rate# {{{
if VALI == 1:
    same = np.zeros(TEST.shape[0], dtype=int)
    same[ans == VALI_ANS] = 1
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
