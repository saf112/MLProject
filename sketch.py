import numpy as np
import keras
from keras.models import Sequential
import scipy.io
from keras.utils import np_utils
from keras.layers import Dense



model = Sequential()

indices=scipy.io.loadmat('sample_indices.mat')['sample_indices']
labels=scipy.io.loadmat('sample_labels.mat')['sample_labels']
patient=scipy.io.loadmat('sample_pat.mat')['sample_pat']


labels=np.array((0,0,0,1,1,1,2,2,2))

one_hot_labels=np_utils.to_categorical(labels, num_classes=9)
model.add(Dense(units=64, activation='relu', input_dim=24976))
model.add(Dense(units=9, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))

model.fit(patient.T, one_hot_labels, epochs=1000, batch_size=32)


#score = model.evaluate(x_test, y_test, batch_size=128)

