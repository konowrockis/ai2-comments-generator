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

# print(len(train))
# for text in train:
#     print(text)

# chars = list(sorted(reduce(lambda x, y: x.union(y), map(lambda item: set(item), train))))

# print(chars)
# ch = open('test4.txt', 'w', encoding="utf-8")
# for item in chars:
#   ch.write("%s -> \n" % item)

start = 1000000
seq_length = 30
items = 100000

char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

n_vocab = len(chars)
n_patterns = items

model = Sequential()

model.add(GRU(256, input_shape=(seq_length, 1), return_sequences=True))
model.add(Dropout(0.2))
model.add(GRU(256))
model.add(Dropout(0.2))
model.add(Dense(n_vocab, activation='softmax'))
model.load_weights("./results/test_3/weights-improvement-04-1.7086.hdf5")
model.compile(loss='categorical_crossentropy', optimizer='adam')

filepath="./results/test_4/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]


def generate():
    seed = list("\0Americanscanscanscanscanscans".lower())[:seq_length]
    pattern = [char_to_int[char] for char in seed]

    temp = 2
    for i in range(1000):
        x = numpy.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)
        prediction = model.predict(x, verbose=0)
        indices = numpy.argpartition(numpy.reshape(prediction, (n_vocab)), - temp)[ - temp:]
        index2 = numpy.random.choice(temp, 1)[0]
        index = indices[index2]
        result = int_to_char[index]
        sys.stdout.write(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]

for i in range(0, 20):
    dataX = []
    dataY = []

    generate()
    print()

    for j in range(start + items * i, start + items * (i + 1)):
        seq_in = comments[j:j + seq_length]
        seq_out = comments[j + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])

    X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
    X = X / float(n_vocab)
    y = np_utils.to_categorical(dataY)

    model.fit(X, y, epochs=i + 1, initial_epoch=i, batch_size=128, callbacks=callbacks_list)
