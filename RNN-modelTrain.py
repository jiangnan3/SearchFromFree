import keras
import sys
import tensorflow as tf
import pandas
from keras.models import Sequential
from keras.layers import Dense, Lambda, Dropout, LSTM
from keras.utils.np_utils import to_categorical

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from keras.models import load_model

# configCPU = tf.ConfigProto(device_count={'CPU': 1, 'GPU': 0})
# sess = tf.Session(config=configCPU)
sess = tf.Session()
keras.backend.set_session(sess)

#--------------Preprocessing-----------------#

data = pandas.read_csv('You_Data_Set', header=None)

feature = data.drop([48], 1)
label = data[48]

feature = feature.values

# --------------Train----------------- #

train_x, test_x, train_y, test_y = train_test_split(feature, label, test_size=0.2)

train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], 1)
test_x = test_x.reshape(test_x.shape[0], test_x.shape[1], 1)

train_y = to_categorical(train_y, 2)
test_y = to_categorical(test_y, 2)

validation_x = train_x[:2000]
train_x = train_x[2000:]
validation_y = train_y[:2000]
train_y = train_y[2000:]


model = Sequential()
model.add(LSTM((246), batch_input_shape=(None, 48, 1), return_sequences=True))  # white-box 256
model.add(Dropout(0.25))
model.add(LSTM((148), batch_input_shape=(None, 246, 1), return_sequences=True))   # white-box 168
model.add(Dropout(0.25))
model.add(LSTM((108), batch_input_shape=(None, 148, 1), return_sequences=False))   # white-box 128
# model.add(Dropout(0.25))
# model.add(Dense(46, activation='relu'))
# model.add(Dropout(0.25))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.25))
model.add(Dense(2, activation='softmax'))

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy']) mean_squared_error
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_x, train_y,
                epochs=50,
                batch_size=512,
                shuffle=True,
                validation_data=(validation_x, validation_y))

results = model.evaluate(test_x, test_y)
print("\n")
print(results)

model.save('models/RNN_Name.h5')