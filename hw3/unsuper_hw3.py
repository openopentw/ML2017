#! python3
"""
@author: b04902053
"""

# Constants
BATCH_SIZE = 128
# EPOCHS = 150

# LOAD DATA# {{{
# import# {{{
import pandas
import numpy as np
import scipy
# import sys
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

TRAIN_FILE = "data/train.csv"
TEST_FILE = "data/test.csv"
OUTPUT = "vali_submission.csv"

BAD_TRAIN_FILE = 'data/bad_train_data.txt'
# BAD_TEST_FILE = 'data/bad_test_data.txt'  # for unsupervised use

NUM_CLASSES = 7
# }}}

# Load Data# {{{
print('LOADING DATA...')
def normalize(data):# {{{
    mean = np.mean(data, axis=1).reshape(data.shape[0], 1)
    std = 1e-8
    std += np.std(data, axis=1).reshape(data.shape[0], 1)
    data = (data - mean) / std
    return data
# }}}

def histeq(data):# {{{
    for i in range(data.shape[0]):
        imhist, bins = np.histogram(data[i], 256, normed=True)
        cdf = imhist.cumsum()
        cdf = 255 * cdf / cdf[-1]
        data[i] = np.interp(data[i], bins[:-1], cdf)
    return data
# }}}

def load_train_data(): # with execute# {{{
    # load from file
    train_data = pandas.read_csv(TRAIN_FILE, encoding='big5')
    train_data = train_data.values

    # remove bad data
    bad_train_data = np.genfromtxt(BAD_TRAIN_FILE, dtype=int)
    train_data = np.delete(train_data, bad_train_data, axis=0)

    # generate y_train_data# {{{
    y_train_data = train_data[:,0].reshape(train_data.shape[0], 1)
    y_train = np_utils.to_categorical(y_train_data, 7)# }}}

    # generate x_train_data# {{{
    # split train
    x_train_data = np.zeros((train_data.shape[0], 2304), int)
    for i in range(train_data.shape[0]):
        x_train_data[i,:] = np.fromstring(train_data[i,1], dtype=np.int, sep=' ')
    x_train_data = x_train_data.astype(float)
    # normalize
    x_train_data = normalize(x_train_data)
    # }}}

    return (x_train_data, y_train, y_train_data)
(x_train_data, y_train, y_train_data) = load_train_data()# }}}

def load_test_data(): # with execute# {{{
    # load from file
    test_data = pandas.read_csv(TEST_FILE, encoding='big5')
    test_data = test_data.values

    # generate x_test_data# {{{
    # split test
    x_test_data = np.zeros((test_data.shape[0], 2304), int)
    for i in range(test_data.shape[0]):
        x_test_data[i,:] = np.fromstring(test_data[i,1], dtype=np.int, sep=' ')
    x_test_data = x_test_data.astype(float)
    # normalize
    x_test_data = normalize(x_test_data)
    # }}}

    return x_test_data
x_test_data = load_test_data()# }}}
print('FINISH LOADING!!!')
# }}}
# }}}

datagen = ImageDataGenerator(# {{{
    horizontal_flip=True,
    rotation_range=10.,
    width_shift_range=0.1,
    height_shift_range=0.1,
    # shear_range=0.1,
    # # datagen = ImageDataGenerator(# {{{
        # featurewise_center=False,
        # samplewise_center=False,
        # featurewise_std_normalization=False,
        # samplewise_std_normalization=False,
        # zca_whitening=False,
        # rotation_range=10.,
        # width_shift_range=0.2,
        # height_shift_range=0.2,
        # shear_range=0.,
        # zoom_range=0.,
        # channel_shift_range=0.,
        # fill_mode='nearest',
        # cval=0.,
        # horizontal_flip=True,
        # vertical_flip=False,
        # rescale=None,
        # preprocessing_function=None
        # )# }}}
    )# }}}

def generate_model(): # with execute# {{{
    model = Sequential()
    # 48

    model.add(Conv2D(32, 3, activation='relu', input_shape=(48, 48, 1)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    # 46
    model.add(Conv2D(32, 3, activation='relu'))
    model.add(BatchNormalization())
    # 44
    model.add(AveragePooling2D(pool_size=2))
    model.add(Dropout(0.3))
    # 22
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    # 20
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(BatchNormalization())
    # 18
    model.add(AveragePooling2D(pool_size=2))
    model.add(Dropout(0.3))
    # 9
    model.add(Conv2D(128, 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    # 7
    model.add(Conv2D(128, 3, activation='relu'))
    model.add(BatchNormalization())
    # 5
    model.add(ZeroPadding2D(padding=1))
    # 7
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))
    # 3

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(NUM_CLASSES, activation='softmax'))
    model.summary()

    # opt = rmsprop(lr=0.0001, decay=1e-6)
    # opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
model = generate_model()# }}}

# reshape# {{{
x_train = x_train_data.reshape(x_train_data.shape[0], 48, 48, 1)
# }}}
x_vali  = x_train[x_train.shape[0] - 1400:]
y_vali  = y_train[y_train.shape[0] - 1400:]
x_test  = x_train[x_train.shape[0] - 1400:]
x_train = x_train[:x_train.shape[0] - 1400]
y_train = y_train[:y_train.shape[0] - 1400]

# slice unlabeled
x_slice = x_train[int(x_train_data.shape[0] / 10):]
x_train = x_train[:int(x_train_data.shape[0] / 10)]
y_train = y_train[:int(x_train_data.shape[0] / 10)]

# fit & print score# {{{
# # callback# {{{
# CALLBACKS = [
    # EarlyStopping(monitor='val_acc', patience=30, verbose=0),
    # # ModelCheckpoint('best_weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_acc', save_best_only=True, verbose=0),
# ]
# model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2, callbacks=CALLBACKS)# }}}
# model = load_model('./tmp_slice_50.hdf5')
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=50, validation_data=(x_vali, y_vali))

for i in range(50):
    y_slice = model.predict(x_slice, batch_size=128, verbose=1)

    choose = np.max(y_slice, axis=1) > 0.7
    new_x_slice = x_slice[choose]
    new_y_slice = y_slice[choose]

    new_x_train = np.append(x_train, new_x_slice, axis=0)
    new_y_train = np.append(y_train, new_y_slice, axis=0)

    model.fit(new_x_train, new_y_train, batch_size=BATCH_SIZE, epochs=1, validation_data=(x_vali, y_vali))

# score = model.evaluate(x_train, y_train, batch_size=128)
# print ('Train Acc:', score[1])
# y_pred = model.predict_classes(x_test, batch_size=128, verbose=1)
# }}}

# PREDICT DATA# {{{
"""
def save_submission(y_pred): # with execute# {{{
    f = open(OUTPUT, "w+")
    print("id,label", file = f)
    for i in range(y_pred.size):
        print(str(i) + "," + str(y_pred[i]), file=f)
    f.close()
save_submission(y_pred)# }}}

# test_score # {{{
exec(open('test_score.py').read())
# }}}
"""

def shift_1(data):# {{{
    origin = data.reshape(data.shape[0], 48, 48)
    shift_up    = np.zeros((data.shape[0], 48, 48))
    shift_down  = np.zeros((data.shape[0], 48, 48))
    shift_left  = np.zeros((data.shape[0], 48, 48))
    shift_right = np.zeros((data.shape[0], 48, 48))
    shift_up[:,:47,:]   = origin[:,1:,:]
    shift_down[:,1:,:]  = origin[:,:47,:]
    shift_left[:,:,:47] = origin[:,:,1:]
    shift_right[:,:,1:] = origin[:,:,:47]
    shift_up    = shift_up.reshape(shift_up.shape[0], 48, 48, 1)
    shift_down  = shift_down.reshape(shift_up.shape[0], 48, 48, 1)
    shift_left  = shift_left.reshape(shift_up.shape[0], 48, 48, 1)
    shift_right = shift_right.reshape(shift_up.shape[0], 48, 48, 1)
    return (shift_up, shift_down, shift_left, shift_right)
(shift_up, shift_down, shift_left, shift_right) = shift_1(x_test)
# }}}
# predict on shift# {{{
y_prob_0 = model.predict(x_test, batch_size=128, verbose=1)
y_prob_1 = model.predict(shift_up, batch_size=128, verbose=1)
y_prob_1 += model.predict(shift_down, batch_size=128, verbose=1)
y_prob_1 += model.predict(shift_left, batch_size=128, verbose=1)
y_prob_1 += model.predict(shift_right, batch_size=128, verbose=1)
# }}}

def shift_1_1(data):# {{{
    origin = data.reshape(data.shape[0], 48, 48)
    shift_up    = np.zeros((data.shape[0], 48, 48))
    shift_down  = np.zeros((data.shape[0], 48, 48))
    shift_left  = np.zeros((data.shape[0], 48, 48))
    shift_right = np.zeros((data.shape[0], 48, 48))
    shift_up[:,:47,:47]   = origin[:,1:,1:]
    shift_down[:,1:,1:]  = origin[:,:47,:47]
    shift_left[:,1:,:47] = origin[:,:47,1:]
    shift_right[:,:47,1:] = origin[:,1:,:47]
    shift_up    = shift_up.reshape(shift_up.shape[0], 48, 48, 1)
    shift_down  = shift_down.reshape(shift_up.shape[0], 48, 48, 1)
    shift_left  = shift_left.reshape(shift_up.shape[0], 48, 48, 1)
    shift_right = shift_right.reshape(shift_up.shape[0], 48, 48, 1)
    return (shift_up, shift_down, shift_left, shift_right)
(shift_up, shift_down, shift_left, shift_right) = shift_1_1(x_test)
# }}}
# predict on shift# {{{
y_prob_1_1  = model.predict(shift_up, batch_size=128, verbose=1)
y_prob_1_1 += model.predict(shift_down, batch_size=128, verbose=1)
y_prob_1_1 += model.predict(shift_left, batch_size=128, verbose=1)
y_prob_1_1 += model.predict(shift_right, batch_size=128, verbose=1)
# }}}

def shift_2_1(data):# {{{
    origin = data.reshape(data.shape[0], 48, 48)
    shift_up    = np.zeros((data.shape[0], 48, 48))
    shift_down  = np.zeros((data.shape[0], 48, 48))
    shift_left  = np.zeros((data.shape[0], 48, 48))
    shift_right = np.zeros((data.shape[0], 48, 48))
    shift_up[:,:46,:47]   = origin[:,2:,1:]
    shift_down[:,2:,1:]  = origin[:,:46,:47]
    shift_left[:,2:,:47] = origin[:,:46,1:]
    shift_right[:,:46,1:] = origin[:,2:,:47]
    shift_up    = shift_up.reshape(shift_up.shape[0], 48, 48, 1)
    shift_down  = shift_down.reshape(shift_up.shape[0], 48, 48, 1)
    shift_left  = shift_left.reshape(shift_up.shape[0], 48, 48, 1)
    shift_right = shift_right.reshape(shift_up.shape[0], 48, 48, 1)
    return (shift_up, shift_down, shift_left, shift_right)
(shift_up, shift_down, shift_left, shift_right) = shift_2_1(x_test)
# }}}
# predict on shift# {{{
y_prob_2_1  = model.predict(shift_up, batch_size=128, verbose=1)
y_prob_2_1 += model.predict(shift_down, batch_size=128, verbose=1)
y_prob_2_1 += model.predict(shift_left, batch_size=128, verbose=1)
y_prob_2_1 += model.predict(shift_right, batch_size=128, verbose=1)
# }}}

def shift_1_2(data):# {{{
    origin = data.reshape(data.shape[0], 48, 48)
    shift_up    = np.zeros((data.shape[0], 48, 48))
    shift_down  = np.zeros((data.shape[0], 48, 48))
    shift_left  = np.zeros((data.shape[0], 48, 48))
    shift_right = np.zeros((data.shape[0], 48, 48))
    shift_up[:,:47,:46]   = origin[:,1:,2:]
    shift_down[:,1:,2:]  = origin[:,:47,:46]
    shift_left[:,1:,:46] = origin[:,:47,2:]
    shift_right[:,:47,2:] = origin[:,1:,:46]
    shift_up    = shift_up.reshape(shift_up.shape[0], 48, 48, 1)
    shift_down  = shift_down.reshape(shift_up.shape[0], 48, 48, 1)
    shift_left  = shift_left.reshape(shift_up.shape[0], 48, 48, 1)
    shift_right = shift_right.reshape(shift_up.shape[0], 48, 48, 1)
    return (shift_up, shift_down, shift_left, shift_right)
(shift_up, shift_down, shift_left, shift_right) = shift_1_2(x_test)
# }}}
# predict on shift# {{{
y_prob_1_2  = model.predict(shift_up, batch_size=128, verbose=1)
y_prob_1_2 += model.predict(shift_down, batch_size=128, verbose=1)
y_prob_1_2 += model.predict(shift_left, batch_size=128, verbose=1)
y_prob_1_2 += model.predict(shift_right, batch_size=128, verbose=1)
# }}}

def rotate_1(data):# {{{
    origin = data.reshape(data.shape[0], 48, 48)
    rotate_left  = scipy.ndimage.interpolation.rotate(origin, 4, axes=(1, 2), reshape=False)
    rotate_right = scipy.ndimage.interpolation.rotate(origin, 4, axes=(1, 2), reshape=False)
    rotate_left  = rotate_left.reshape(origin.shape[0], 48, 48, 1)
    rotate_right = rotate_right.reshape(origin.shape[0], 48, 48, 1)
    return (rotate_left, rotate_right)
(rotate_left, rotate_right) = rotate_1(x_test)
# }}}
# predict on rotate# {{{
y_prob_r1  = model.predict(rotate_left, batch_size=128, verbose=1)
y_prob_r1 += model.predict(rotate_right, batch_size=128, verbose=1)
# }}}

def rotate_2(data):# {{{
    origin = data.reshape(data.shape[0], 48, 48)
    rotate_left  = scipy.ndimage.interpolation.rotate(origin, 4, axes=(1, 2), reshape=False)
    rotate_right = scipy.ndimage.interpolation.rotate(origin, 4, axes=(1, 2), reshape=False)
    rotate_left  = rotate_left.reshape(origin.shape[0], 48, 48, 1)
    rotate_right = rotate_right.reshape(origin.shape[0], 48, 48, 1)
    return (rotate_left, rotate_right)
(rotate_left, rotate_right) = rotate_2(x_test)
# }}}
# predict on shift dim# {{{
y_prob_r2  = model.predict(rotate_left, batch_size=128, verbose=1)
y_prob_r2 += model.predict(rotate_right, batch_size=128, verbose=1)
# }}}

def rotate_3(data):# {{{
    origin = data.reshape(data.shape[0], 48, 48)
    rotate_left  = scipy.ndimage.interpolation.rotate(origin, 4, axes=(1, 2), reshape=False)
    rotate_right = scipy.ndimage.interpolation.rotate(origin, 4, axes=(1, 2), reshape=False)
    rotate_left  = rotate_left.reshape(origin.shape[0], 48, 48, 1)
    rotate_right = rotate_right.reshape(origin.shape[0], 48, 48, 1)
    return (rotate_left, rotate_right)
(rotate_left, rotate_right) = rotate_3(x_test)
# }}}
# predict on shift dim# {{{
y_prob_r3  = model.predict(rotate_left, batch_size=128, verbose=1)
y_prob_r3 += model.predict(rotate_right, batch_size=128, verbose=1)
# }}}

def rotate_4(data):# {{{
    origin = data.reshape(data.shape[0], 48, 48)
    rotate_left  = scipy.ndimage.interpolation.rotate(origin, 4, axes=(1, 2), reshape=False)
    rotate_right = scipy.ndimage.interpolation.rotate(origin, 4, axes=(1, 2), reshape=False)
    rotate_left  = rotate_left.reshape(origin.shape[0], 48, 48, 1)
    rotate_right = rotate_right.reshape(origin.shape[0], 48, 48, 1)
    return (rotate_left, rotate_right)
(rotate_left, rotate_right) = rotate_4(x_test)
# }}}
# predict on shift dim# {{{
y_prob_r4  = model.predict(rotate_left, batch_size=128, verbose=1)
y_prob_r4 += model.predict(rotate_right, batch_size=128, verbose=1)
# }}}

y_prob = 10*y_prob_0 + 8*y_prob_1 + 6*y_prob_1_1 + 4*y_prob_1_2 + 4*y_prob_2_1 + 1*y_prob_r1 + 0*y_prob_r2 + 0*y_prob_r3 + 0*y_prob_r4
y_pred = np.argmax(y_prob, axis=1)

def save_submission(y_pred): # with execute# {{{
    f = open(OUTPUT, "w+")
    print("id,label", file = f)
    for i in range(y_pred.size):
        print(str(i) + "," + str(y_pred[i]), file=f)
    f.close()
save_submission(y_pred)# }}}

# test_score # {{{
exec(open('./vali_test_score.py').read())
# }}}
# }}}

model.save('tmp_slice_unsuper_150_2.hdf5')
