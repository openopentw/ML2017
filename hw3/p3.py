#! python3
"""
@author: b04902053
"""

# import
import pandas
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib
import matplotlib.pyplot as plt

# Argvs
# OUTPUT = './con_submission.csv'
OUTPUT = './vali_submission.csv'
ANS = './data/train.csv'

# load data
output_data = pandas.read_csv(OUTPUT, encoding='big5')
output_data = output_data.values
output = output_data[:,1]

ans_data = pandas.read_csv(ANS, encoding='big5').values[:,0]
ans = ans_data.reshape(ans_data.size,)
ans = ans[ans.shape[0] - 1400:].astype(int)

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.jet):# {{{
    """
    This function prints and plots the confusion matrix.
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    matplotlib.rcParams.update({'font.size': 16})
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
# }}}

conf_mat = confusion_matrix(ans, output)
plt.figure()
plot_confusion_matrix(conf_mat, classes=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"], cmap=plt.cm.YlGnBu)
plt.show()
