import h5py
import numpy
import sys
from keras.models import Sequential
from keras.layers import GRU, Dropout, Dense
from keras.utils import np_utils

with open('./data/fb_news_comments.txt', 'r', encoding='utf-8') as file:
    comments = file.read()

chars = list(sorted(set(comments)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
n_vocab=len(chars)
seq_length = 30

model = Sequential()
model.add(GRU(256, input_shape=(seq_length, 1), return_sequences=True))
model.add(Dropout(0.2))
model.add(GRU(256))
model.add(Dropout(0.2))
model.add(Dense(n_vocab, activation='softmax'))
model.load_weights("./results/test_3/weights-improvement-04-1.7086.hdf5")
model.compile(loss='categorical_crossentropy', optimizer='adam')

seed = list("\0Americanscanscanscanscanscans".lower())[:seq_length]
pattern = [char_to_int[char] for char in seed]
# print "Seed:"
# print "\"", ''.join([int_to_char[value] for value in pattern]), "\""
# generate characters
sys.stdout.write(''.join(seed))

temp = 2
for i in range(1000):
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	x = x / float(n_vocab)
	prediction = model.predict(x, verbose=0)
	indices = numpy.argpartition(numpy.reshape(prediction, (n_vocab)), - temp)[ - temp:]
	index2 = numpy.random.choice(temp, 1)[0]
	index = indices[index2]
	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	sys.stdout.write(result)
	pattern.append(index)
	pattern = pattern[1:len(pattern)]
