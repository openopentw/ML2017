# Load Data# {{{
# import# {{{
import pandas
import numpy as np
import scipy
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
# TEST_FILE = "data/test.csv"
# OUTPUT = "submission.csv"
# }}}

# LOAD DATA# {{{
print('LOADING DATA...')
def normalize(data):# {{{
    mean = np.mean(data, axis=1).reshape(data.shape[0], 1)
    std = 1e-8
    std += np.std(data, axis=1).reshape(data.shape[0], 1)
    data = (data - mean) / std
    return data
# }}}

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

x_test  = x_test_data.reshape(x_test_data.shape[0], 48, 48, 1)
# }}}

def predict_data(model):# {{{
    print('0/4')
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

    print('1/4')
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

    print('2/4')
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

    print('3/4')
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

    print('4/4')
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

    y_prob = 10*y_prob_0 + 8*y_prob_1 + 6*y_prob_1_1 + 4*y_prob_1_2 + 4*y_prob_2_1 + 1*y_prob_r1

    return y_prob
# }}}

model_list = ['./674003.hdf5', './674561.hdf5', './659933.hdf5']
prob_list = []
for m_name in model_list:
    model = load_model(m_name)
    prob_list += [predict_data(model)]

y_prob = np.zeros((7178, 7))
for prob in prob_list:
    y_prob += prob

y_pred  = np.argmax(y_prob, axis=1)

def save_submission(y_pred): # with execute# {{{
    f = open(OUTPUT, "w")
    print("id,label", file = f)
    for i in range(y_pred.size):
        print(str(i) + "," + str(y_pred[i]), file=f)
    f.close()
save_submission(y_pred)# }}}
