"""
@author: b04902053
"""
# import# {{{
import pandas as pd
import numpy as np
import sys
# keras# {{{
import keras.backend as K
from keras.models import Sequential, load_model
from keras.models import Model
from keras.layers import Input, Flatten, Embedding, Dropout, Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Add, Dot, Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
# }}}
# }}}
def RMSE(y_true, y_pred):# {{{
    return K.sqrt(K.mean(K.square(y_pred - y_true)))
# }}}
# parameters #
# argv# {{{
test_path   = sys.argv[1] + 'test.csv'
output_path = sys.argv[2]
# test_path   = '../data/test.csv'
# output_path = './submission.csv'
print('Will save output to: {}'.format(output_path))
# }}}
# load test data# {{{
testID = pd.read_csv(test_path)['TestDataID'].values
test = pd.read_csv(test_path)[['UserID', 'MovieID']].values
test[:,0] = test[:,0] - 1
test[:,1] = test[:,1] - 1
# }}}
model_list = [
    '35.h5',
]
# load & predict# {{{
m = model_list[0]
model = load_model(m, custom_objects={'RMSE': RMSE})
y_pred = model.predict([test[:,0], test[:,1]])
# }}}
# save to h5 & csv# {{{
print('Saving submission to {}'.format(output_path))
f = open(output_path, 'w')
print('TestDataID,Rating', file=f)
for i, pred_rate in enumerate(y_pred):
    print('{},{}'.format(testID[i], pred_rate[0]), file=f)
f.close()
# }}}
