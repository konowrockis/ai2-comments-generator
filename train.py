import h5py
import math
import time
import numpy
import sys
from functools import reduce
from keras.models import Sequential
from keras.layers import GRU, LSTM, Dropout, Dense
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

with open('./data/fb_news_comments.txt', 'r', encoding='utf-8') as file:
    comments = file.read()

chars = list(sorted(set(comments)))

# print(''.join(chars))
# print([ord(x) for x in chars])
# exit()

start = 0
seq_length = 100
items = 200000

char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

n_vocab = len(chars)
n_patterns = items

model = Sequential()

model.add(GRU(512, input_shape=(seq_length, 1), return_sequences=True))
model.add(Dropout(0.2))
model.add(GRU(256))
model.add(Dropout(0.2))
model.add(Dense(n_vocab, activation='softmax'))
model.load_weights("./results/test_6/weights-improvement-60-1.7856.hdf5")
model.compile(loss='categorical_crossentropy', optimizer='adam')

filepath="./results/test_6/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, mode='min')
callbacks_list = [checkpoint]

def generate():
    seed = list("To me, something just doesn't add up.... It helps that this article says he killed them separately and a second person of interest might be involved.".lower())[:seq_length]
    pattern = [char_to_int[char] for char in seed]

    # temp = 2
    for i in range(1000):
        x = numpy.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)
        prediction = model.predict(x, verbose=0)
        
        index = numpy.random.choice(n_vocab, 1, p=numpy.reshape(prediction, n_vocab))[0]
        result = int_to_char[index]
        sys.stdout.write(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]

for i in range(33, 100):
    dataX = []
    dataY = []

    generate()
    exit()
    print()

    for j in range(start + items * i, start + items * (i + 1)):
        seq_in = comments[j:j + seq_length]
        seq_out = comments[j + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])

    X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
    X = X / float(n_vocab)
    y = np_utils.to_categorical(dataY)

    model.fit(X, y, epochs=i * 2 + 2, initial_epoch=i * 2, batch_size=128, callbacks=callbacks_list)
