# import# {{{
import numpy as np
import sys
# keras# {{{
import keras.backend as K 
from keras.preprocessing.text import Tokenizer,text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import GRU
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Activation, Dropout, Flatten, Dense,Conv1D,MaxPooling1D,BatchNormalization# }}}
import pickle
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import re
# }}}
ID = 18
# argv# {{{
test_path      = sys.argv[1]
output_path    = sys.argv[2]
# test_path    = '../data/test_data.csv'
# output_path  = './submission_{}.csv'.format(ID)

weights_path   = './weights_{}.h5'.format(ID)
tag_path       = './tag_list'
tokenizer_path = './tokenizer.pickle'
# }}}
#   Util   #
def read_data(path,training):# {{{
    print ('Reading data from ',path)
    stopword = stopwords.words('english')
    lmtzr = WordNetLemmatizer()
    with open(path,'r', encoding='utf8') as f:

        tags = []
        articles = []
        tags_list = []

        f.readline()
        for line in f:
            if training :
                start = line.find('\"')
                end = line.find('\"',start+1)
                tag = line[start+1:end].split(' ')
                article = line[end+2:]

                for t in tag :
                    if t not in tags_list:
                        tags_list.append(t)

                tags.append(tag)
            else:
                start = line.find(',')
                article = line[start+1:]

            article = re.sub('[^a-zA-Z]', ' ', article)
            article = text_to_word_sequence(article, lower=True, split=' ')
            article = [ w for w in article if w not in stopword ]
            article = [ lmtzr.lemmatize(w) for w in article ]
            article = ' '.join(article)
            articles.append(article)

        if training :
            assert len(tags_list) == 38,(len(tags_list))
            assert len(tags) == len(articles)
    return (tags,articles,tags_list)
# }}}
def f1_score(y_true,y_pred):# {{{
    thresh = 0.4
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true * y_pred,axis=-1)

    precision=tp/(K.sum(y_pred,axis=-1)+K.epsilon())
    recall=tp/(K.sum(y_true,axis=-1)+K.epsilon())
    return K.mean(2*((precision*recall)/(precision+recall+K.epsilon())))
# }}}
#   Main function   #
# load tags_list# {{{
f = open(tag_path, encoding='utf8')
lines = f.readlines()
f.close()
tags_list = [s.rstrip() for s in lines]
print (tags_list)
# }}}
(_,X_data,_) = read_data(test_path,False)
# load tokenizer & texts_to_matrix# {{{
tokenizer = pickle.load(open(tokenizer_path,'rb'))
print ('Convert to index sequences.')
X_test = tokenizer.texts_to_matrix(X_data, mode='tfidf')
# }}}
# generate model# {{{
print ('Building model.')

model = Sequential()
model.add(Dense(512,activation='elu',input_dim=40587))
model.add(Dense(512,activation='elu'))
model.add(Dense(512,activation='elu'))
model.add(Dense(512,activation='elu'))
model.add(Dense(38,activation='sigmoid'))
model.summary()
adam = Adam(lr=0.001,decay=1e-6,clipvalue=0.5)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[f1_score])
# }}}

# predict #
# load best# {{{
print('loading weights from {}'.format(weights_path))
model.load_weights(weights_path)
Y_pred = model.predict(X_test)
thresh = 0.4
# }}}
# predict# {{{
print('Saving submission to {}'.format(output_path))
with open(output_path,'w') as output:
    print ('\"id\",\"tags\"',file=output)
    Y_pred_thresh = (Y_pred > thresh).astype('int')
    for index,labels in enumerate(Y_pred_thresh):
        labels = [tags_list[i] for i,value in enumerate(labels) if value==1 ]
        labels_original = ' '.join(labels)
        print ('\"%d\",\"%s\"'%(index,labels_original),file=output)
# }}}
