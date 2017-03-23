#! python3
"""
Created on Thu Mar  9 09:31:27 2017
@author: b04902053
"""

# imports# {{{
import pandas
import numpy as np
import sys
DATA_FILE = sys.argv[1]
TEST_FILE = sys.argv[2]
OUTPUT = sys.argv[3]

# print(sys.argv[1], sys.argv[2], sys.argv[3])
# }}}

NORM = 1

# Input Data# {{{
# format: DATA['feature']['hour']

DATA = pandas.read_csv(DATA_FILE, encoding='big5')
DATA = DATA.values

# append the same features together
DATA = DATA[ np.argsort(DATA[:,2], kind='mergesort') ]
DATA = DATA[:,3:]
DATA = DATA.reshape(18, int(DATA.size / 18))

# convert DATA to float
DATA[DATA == 'NR'] = 0
DATA = DATA.astype(float)

INPUT_DATA = DATA

# normalize DATA
if NORM == 1:
    MEAN = np.mean(DATA, axis=1).reshape(18, 1)
    STD = np.std(DATA, axis=1).reshape(18, 1)

    DATA = (DATA - MEAN) / STD
# }}}

DIM = 2

"""
Features:

0 AMB_TEMP	1 CH4		2 CO		3 NMHC		4 NO
5 NO2		6 NOx		7 O3		8 PM10		9 PM2.5
10 RAINFALL	11 RH		12 SO2		13 THC		14 WD_HR
15 WIND_DIREC	16 WIND_SPEED	17 WS_HR
"""
FEATURE = np.array([9])     # best
if sys.argv[4] == '1':        # not best
    FEATURE = np.array([8, 9, 10])
N_FEATURE = FEATURE.size

# Adjust Data To Be Trainable# {{{
# format: DATA[ i-th data to train (month * feature * hour) ]

# let DATA be "0123456789 12345678910" format
DATA_DIM = np.zeros((DIM, DATA.shape[0], DATA.shape[1]))
for i in range(DIM):
    DATA_DIM[i] = DATA ** (i+1)

DATA = np.zeros(12 * 471 * (N_FEATURE*9*DIM+1))
pos = 0
for i in range(12):
    for j in range(i*471, (i+1)*471):
        for k in range(N_FEATURE):
            for l in range(DIM):
                DATA[pos:pos+9] = DATA_DIM[l][FEATURE[k]][j:j+9]
                pos += 9
        DATA[pos] = DATA_DIM[0][9][j+9]
        pos += 1
DATA = DATA.reshape(12*471, N_FEATURE*9*DIM + 1)
# }}}

LMBD = 1e2
YETA = 1
if NORM == 1:
    LMBD /= STD[9][0]
    YETA /= STD[9][0]

train = DATA[ : int(DATA.shape[0]/2)]
test = DATA[int(DATA.shape[0]/2) : ]
# train = DATA[:]
# test = DATA[:]

"""
INIT Training
"""
ROOT_SUM_b = 0
ROOT_SUM_W = np.zeros((1, N_FEATURE*9*DIM))

b = 0
W = np.zeros((1, N_FEATURE*9*DIM))
LOSS = 0
LAST_LOSS = 0

"""
Train
"""
num_test = test.shape[0]

X = train[:,:-1]
ans = train[:,-1].reshape((train.shape[0], 1))
testX = test[:,:-1]
testans = test[:,-1].reshape((test.shape[0], 1))

for i in range(5000000):
    # gradient
    sqrt_loss = ans - ( b + np.dot(X, W.T ) )
    grad_b = 2 * np.sum(sqrt_loss)
    grad_W = 2 * (np.sum(sqrt_loss * X, axis=0) + LMBD * W)

    # learning_rate
    ROOT_SUM_b += grad_b ** 2
    rate_b = YETA / np.sqrt(ROOT_SUM_b)
    ROOT_SUM_W += grad_W ** 2
    rate_W = YETA / np.sqrt(ROOT_SUM_W)

    # descent
    b += rate_b * grad_b
    W += rate_W * grad_W

    # loss
    LAST_LOSS = LOSS
    sqrt_loss = testans - (b + np.dot(testX, W.T))
    if NORM == 1:
        sqrt_loss *= STD[9][0]
    LOSS = np.sum(sqrt_loss ** 2) + LMBD * np.sum(W ** 2)

    # print(i, np.sqrt(LOSS/num_test))
    # if i > 3000 and LOSS > LAST_LOSS:
    if i > 50 and LOSS > LAST_LOSS:
        # print(i, np.sqrt(LOSS/num_test))
        break

# print(b)
# print(W)

sqrt_loss = DATA[:,-1].reshape((DATA.shape[0], 1)) - ( b + np.dot(DATA[:,:-1], W.T ) )
if NORM == 1:
    sqrt_loss *= STD[9][0]
LOSS = np.sum(sqrt_loss ** 2)
# print(np.sqrt(LOSS/DATA.shape[0]))

"""
Calculate Ans & Print Out
"""
TEST = np.genfromtxt(TEST_FILE, delimiter=',')
TEST = TEST[:,2:]

# convert TEST to float
TEST = np.nan_to_num(TEST)
TEST = TEST.astype(float)

# normalize & slice unused features
TEST = TEST.reshape(240, 18, 9)
if NORM == 1:
    TEST = (TEST - MEAN) / STD
TEST = TEST[:, FEATURE,:]

# append TEST & DIM.S together
TEST = TEST.reshape(240*N_FEATURE, 9)
TEST_DIM = np.zeros((TEST.shape[0], TEST.shape[1] * DIM))
for i in range(DIM):
    TEST_DIM[:, 9*i: 9*(i+1)] = TEST ** (i+1)
TEST_DIM = TEST_DIM.reshape(240, int(TEST_DIM.size / 240))

ans = b + np.dot(TEST_DIM , W.T)
if NORM == 1:
    ans = ans * STD[9][0] + MEAN[9][0]

# save to 'submission.csv'
f = open(OUTPUT, "w+")
print("id,value", file = f, end = "\n")
for i in range(ans.size):
    print("id_" + str(i) + "," + str(np.round(ans[i][0])), file = f, end = "\n")
f.close()
