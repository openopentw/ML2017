#! python3
"""
@author: b04902053
"""

# # Use CPU# {{{
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# # }}}

BATCH_SIZE = 128
EPOCHS = 250

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
# TRAIN_FILE = sys.argv[1]
# TEST_FILE = sys.argv[2]
# OUTPUT = sys.argv[3]

X_TRAIN_FILE = "./gen/mm_x_train.npy"
Y_TRAIN_FILE = "./gen/mm_y_train.csv"
# TEST_FILE = "./data.npz"
TEST_FILE = "./eigh.csv"
OUTPUT = "./submission.csv"
# }}}

# Load Data# {{{
CHOOSE = 40000
END    = 60000

x_train_data = np.load(X_TRAIN_FILE                                        )[:CHOOSE]
x_train_data = np.append(x_train_data, np.load('./gen/7_x_train_20000.npy' )[:CHOOSE,40:], axis=0)
x_train_data = np.append(x_train_data, np.load('./gen/7_x_train_50000.npy' )[:CHOOSE,40:], axis=0)
x_train_data = np.append(x_train_data, np.load('./gen/7_x_train_80000.npy' )[:CHOOSE,40:], axis=0)
x_train_data = np.append(x_train_data, np.load('./gen/7_x_train_100000.npy')[:CHOOSE,40:], axis=0)
x_train = x_train_data.reshape(x_train_data.shape[0], x_train_data.shape[1], 1)

y_train_data = np.genfromtxt(Y_TRAIN_FILE, delimiter=','         )[:CHOOSE]
y_train = y_train_data.reshape(y_train_data.shape[0], 1)
y_train = np.append(y_train, np.load('./gen/7_y_train_20000.npy' )[:CHOOSE], axis=0)
y_train = np.append(y_train, np.load('./gen/7_y_train_50000.npy' )[:CHOOSE], axis=0)
y_train = np.append(y_train, np.load('./gen/7_y_train_80000.npy' )[:CHOOSE], axis=0)
y_train = np.append(y_train, np.load('./gen/7_y_train_100000.npy')[:CHOOSE], axis=0)
y_train = np.log(y_train)

x_vali_data = np.load('./gen/mm_x_vali.npy'                              )[CHOOSE:END]
x_vali_data = np.append(x_vali_data, np.load('./gen/7_x_train_20000.npy' )[CHOOSE:END,40:], axis=0)
x_vali_data = np.append(x_vali_data, np.load('./gen/7_x_train_50000.npy' )[CHOOSE:END,40:], axis=0)
x_vali_data = np.append(x_vali_data, np.load('./gen/7_x_train_80000.npy' )[CHOOSE:END,40:], axis=0)
x_vali_data = np.append(x_vali_data, np.load('./gen/7_x_train_100000.npy')[CHOOSE:END,40:], axis=0)
x_vali = x_vali_data.reshape(x_vali_data.shape[0], x_vali_data.shape[1], 1)

y_vali_data = np.genfromtxt('./gen/mm_y_vali.csv', delimiter=',')[CHOOSE:END]
y_vali = y_vali_data.reshape(y_vali_data.shape[0], 1)
y_vali = np.append(y_vali, np.load('./gen/7_y_train_20000.npy'  )[CHOOSE:END], axis=0)
y_vali = np.append(y_vali, np.load('./gen/7_y_train_50000.npy'  )[CHOOSE:END], axis=0)
y_vali = np.append(y_vali, np.load('./gen/7_y_train_80000.npy'  )[CHOOSE:END], axis=0)
y_vali = np.append(y_vali, np.load('./gen/7_y_train_100000.npy' )[CHOOSE:END], axis=0)
y_vali = np.log(y_vali)

x_test = np.genfromtxt(TEST_FILE)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
# }}}

def generate_model(): # {{{
    model = Sequential()
    model.add(Flatten(input_shape=(60, 1)))

    model.add(Dense(100, activation='elu'))
    model.add(Dense(100, activation='elu'))
    model.add(Dense(100, activation='elu'))
    model.add(Dense(100, activation='elu'))
    model.add(Dense( 60, activation='elu'))
    model.add(Dense( 60, activation='elu'))

    model.add(Dense(1))
    model.summary()
    return model# }}}

# load & fit# {{{
model = generate_model()
# model = load_model('./dnn_mm_14_200.hdf5')

model.compile(loss='mean_absolute_error', optimizer='adam')
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=50, validation_data=(x_vali, y_vali))

model.compile(loss='mean_absolute_error', optimizer='adagrad')
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS-50, validation_data=(x_vali, y_vali))
# }}}

# print score# {{{
score = model.evaluate(x_train, y_train)
print ('Train Acc:', score)
score = model.evaluate(x_vali, y_vali)
print ('Vali Acc:', score)
y_pred = model.predict(x_test)
y_print = np.log(np.round(np.exp(y_pred)))
# }}}

def save_submission(y_pred): # with execute# {{{
    f = open('y_pred.csv', "w")
    print("SetId,LogDim", file=f)
    for i in range(y_pred.size):
        print(str(i) + "," + str(y_pred[i][0]), file=f)
    f.close()
save_submission(np.round(y_pred))# }}}

def save_submission(y_print): # with execute# {{{
    f = open(OUTPUT, "w")
    print("SetId,LogDim", file=f)
    for i in range(y_print.size):
        print(str(i) + "," + str(y_print[i][0]), file=f)
    f.close()
save_submission(y_print)# }}}

model.save('dnn_mm_18_250.hdf5')
