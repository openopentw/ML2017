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
    'model/11.h5',
    'model/14.h5',
    'model/15.h5',
    'model/16.h5',
    'model/17.h5',
    'model/25.h5',
    'model/34.h5',
    'model/35.h5',
    'model/36.h5',
    'model/37.h5',
    'model/40.h5',
    'model/41.h5',
    'model/42.h5',
    'model/43.h5',
    'model/44.h5',
    'model/45.h5',
    'model/46.h5',
    'model/47.h5',
    'model/48.h5',
    'model/49.h5',
]
# load & predict# {{{
preds = np.zeros((len(model_list), test.shape[0], 1))
for i,m in enumerate(model_list):
    print(i)
    model = load_model(m, custom_objects={'RMSE': RMSE})
    preds[i] = model.predict([test[:,0], test[:,1]])
y_pred = np.mean(preds, axis=0)
# }}}
# clip on 1 & 5# {{{
y_pred[y_pred < 1] = 1
y_pred[y_pred > 5] = 5
# }}}
# save to h5 & csv# {{{
print('Saving submission to {}'.format(output_path))
f = open(output_path, 'w')
print('TestDataID,Rating', file=f)
for i, pred_rate in enumerate(y_pred):
    print('{},{}'.format(testID[i], pred_rate[0]), file=f)
f.close()
# }}}
