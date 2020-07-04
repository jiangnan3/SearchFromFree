import keras
import pandas
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

# use CPU only
# configCPU = tf.ConfigProto(device_count={'CPU': 1, 'GPU': 0})
# sess = tf.Session(config=configCPU)

sess = tf.Session()  # use GPU

keras.backend.set_session(sess)

#--------------Preprocessing-----------------#

data = pandas.read_csv('You_Data_Set', header=None)

feature = data.drop([48], 1)
label = data[48]

scaler = preprocessing.StandardScaler().fit(feature)
joblib.dump(scaler, 'scalers/scaler.pkl')
# clf = joblib.load('filename.pkl')
# feature = scaler.transform(feature)
feature = feature.values

# --------------Train----------------- #

train_x, test_x, train_y, test_y = train_test_split(feature, label, test_size=0.2)

train_y = to_categorical(train_y, 2)
test_y = to_categorical(test_y, 2)

validation_x = train_x[:2000]
train_x = train_x[2000:]
validation_y = train_y[:2000]
train_y = train_y[2000:]


model = Sequential()
model.add(Dense(128, activation='relu', input_dim=48))
model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
# model.add(Dense(40, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.25))
# model.add(Lambda(lambda x: x / 10))
model.add(Dense(2, activation='softmax'))

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy']) mean_squared_error
model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_x, train_y,
                epochs=100,
                batch_size=1024,
                shuffle=True,
                validation_data=(validation_x, validation_y))

results = model.evaluate(test_x, test_y)
print("\n")
print(results)

model.save('models/FNN_Name.h5')

