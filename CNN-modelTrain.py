import tensorflow as tf
import keras
import sys
import numpy as np
import pandas

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, Conv2D, MaxPooling2D
from keras.utils.np_utils import to_categorical

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib


# configCPU = tf.ConfigProto(device_count={'CPU': 1, 'GPU': 0})
# sess = tf.Session(config=configCPU)
sess = tf.Session()
keras.backend.set_session(sess)

#--------------Preprocessing-----------------#

data = pandas.read_csv('You_Data_Set', header=None)

feature = data.drop([48], 1)
label = data[48]
feature = feature.values
row, col = 6, 8
# row, col = int(sys.argv[2]), int(sys.argv[3])
feature = feature.reshape((feature.shape[0], row, col, 1)) # 6 * 10 when 60, 7 * 7 when 49

# --------------Train----------------- #

train_x, test_x, train_y, test_y = train_test_split(feature, label, test_size = 0.25)

train_y = to_categorical(train_y, 2)
test_y = to_categorical(test_y, 2)

validation_x = train_x[:2000]
train_x = train_x[2000:]
validation_y = train_y[:2000]
train_y = train_y[2000:]

print(row, col)

model = Sequential()
model.add(Conv2D(156, kernel_size=(3, 3),  # int(sys.argv[3])
                 activation='relu',
                 input_shape=(row, col, 1)))

model.add(Conv2D(214, kernel_size=(3, 3),  # int(sys.argv[3])
                 activation='relu',
                 input_shape=(row, col, 1)))

# model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))  # int(sys.argv[4])
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(48, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy']) mean_squared_error
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_x, train_y,
                epochs=60,
                batch_size=512,
                shuffle=True,
                validation_data=(validation_x, validation_y))

results = model.evaluate(test_x, test_y)
print("\n")
print(results)

model.save('models/CNN_Name.h5')

