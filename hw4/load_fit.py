#! python3
"""
@author: b04902053
"""

# import# {{{
import numpy as np
import sys
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import ZeroPadding2D
from keras.optimizers import SGD, Adam, rmsprop
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
# }}}

# Argvs# {{{
TEST_FILE = sys.argv[1]
OUTPUT = sys.argv[2]

# TEST_FILE = "d:/data.npz"
# OUTPUT = "./testsubmission.csv"
# }}}

# Load Data# {{{
data = np.load(TEST_FILE)

eigh = np.zeros((200, 60))
for i in range(200):
    x = data[str(i)]
    val, vec = np.linalg.eigh(np.cov(x.T))
    eigh[i] = val[:60]

x_test = eigh.reshape(eigh.shape[0], eigh.shape[1], 1)
# }}}

model_list_int = [
        './dnn_mm_10_400.hdf5'
        ]
model_list = [
        './dnn_mm_14_200.hdf5',
        './dnn_mm_15_300.hdf5',
        './dnn_mm_16_300.hdf5',
        './dnn_mm_17_250.hdf5',
        ]
n_list = len(model_list_int) + len(model_list)

ans = []
y_pred = 0

#training using int# {{{
for m in model_list_int:
    print(m)
    model = load_model(m)
    y_pred += model.predict(x_test)
# }}}

# training using log# {{{
for m in model_list:
    print(m)
    model = load_model(m)
    y_pred += np.exp(model.predict(x_test))
# }}}

y_pred /= n_list
y_print = np.log(np.round(y_pred))

def save_submission(y_print): # with execute# {{{
    f = open(OUTPUT, "w")
    print("SetId,LogDim", file=f)
    for i in range(y_print.size):
        print(str(i) + "," + str(y_print[i][0]), file=f)
    f.close()
save_submission(y_print)# }}}
