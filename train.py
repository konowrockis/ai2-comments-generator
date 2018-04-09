import h5py
import math
import time
import numpy
from functools import reduce
from keras.models import Sequential
from keras.layers import GRU, LSTM, Dropout, Dense
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

with open('./data/fb_news_comments.txt', 'r', encoding='utf-8') as file:
    comments = file.read()

chars = list(sorted(set(comments)))

# print(len(train))
# for text in train:
#     print(text)

# chars = list(sorted(reduce(lambda x, y: x.union(y), map(lambda item: set(item), train))))

# print(chars)
# ch = open('test4.txt', 'w', encoding="utf-8")
# for item in chars:
#   ch.write("%s -> \n" % item)

char_to_int = dict((c, i) for i, c in enumerate(chars))

dataX = []
dataY = []

start = 1000000
items = 6000000
seq_length = 30

for i in range(start, start + items):
    seq_in = comments[i:i + seq_length]
    seq_out = comments[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])

n_vocab = len(chars)
n_patterns = len(dataX) 
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
X = X / float(n_vocab)
y = np_utils.to_categorical(dataY)

model = Sequential()

model.add(GRU(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(GRU(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.load_weights("./results/test_2/weights-improvement-10-1.7733.hdf5")
model.compile(loss='categorical_crossentropy', optimizer='adam')

filepath="./results/test_3/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
model.fit(X, y, epochs=10, batch_size=128, callbacks=callbacks_list)
