import keras
import winsound
import sys
import pandas
import tensorflow as tf
import numpy as np
import random

from keras.models import load_model
from keras.utils.np_utils import to_categorical


def advEval(model, alpha):
    testDataNum = 1000

    badDataFile = "YouDataFile.csv"
    data = pandas.read_csv(badDataFile, header=None)
    goodData = data.loc[data[48] == 0].reset_index().drop(['index'], axis=1).copy()
    indexList = random.sample(list(range(goodData.shape[0])), testDataNum)
    indexList.sort()
    goodData = goodData.iloc[indexList, :].reset_index().drop(['index'], axis=1).copy()

    goodDataFeature = goodData.drop([48], 1).copy().values

    # stupid attacker
    adversarial_example = goodDataFeature * alpha
    # stupid attacker

    adv_sample = np.asarray(adversarial_example)

    AML_result = model.evaluate(adv_sample.reshape(adv_sample.shape[0], adv_sample.shape[1]), to_categorical(np.asarray([1] * testDataNum)))
    print('Afer AML Test loss:', AML_result[0])
    print('After AML Test accuracy:', AML_result[1])
    print("the overall energy consumption is: ", np.sum(np.sum(adv_sample)) / testDataNum)

    record = open('You_Record_File.txt', 'a')
    recordString = "\n alpha:" + str(alpha) + " Accucacy:" + str(AML_result[1]) + \
                   " size:" + str(np.sum(np.sum(adv_sample)) / testDataNum) + "\n"
    record.write(recordString)
    record.close()



def startXero(model, u):
    testDataNum = 1000

    # stupid attacker
    adversarial_example = np.random.uniform(0, u, (testDataNum, 48))
    # stupid attacker

    adv_sample = np.asarray(adversarial_example)

    AML_result = model.evaluate(adv_sample.reshape(adv_sample.shape[0], adv_sample.shape[1]), to_categorical(np.asarray([1] * testDataNum)))
    print('Afer AML Test loss:', AML_result[0])
    print('After AML Test accuracy:', AML_result[1])
    print("the overall energy consumption is: ", np.sum(np.sum(adv_sample)) / testDataNum)

    record = open('You_Record_File.txt', 'a')
    recordString = "\n u:" + str(u) + " Accucacy:" + str(AML_result[1]) + \
                   " size:" + str(np.sum(np.sum(adv_sample)) / testDataNum) + "\n"
    record.write(recordString)
    record.close()


alpha = float(sys.argv[1])
u = float(sys.argv[2])

sess = tf.Session()
keras.backend.set_session(sess)

targetModel = load_model("models/XXX.h5")

record = open('You_Record_File.txt', 'a')
record.write("===============")
record.close()

advEval(targetModel, alpha)
print("================================================")
startXero(targetModel, u)

