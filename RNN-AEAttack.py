import keras
import winsound
import sys
import pandas
import tensorflow as tf
import numpy as np
import random

from keras.models import load_model
from keras.utils.np_utils import to_categorical


def map2Positive(inputArray):

    for i in range(inputArray.shape[0]):
        if inputArray[i] <= 0:
            inputArray[i] = 0

    return inputArray


def scaled_gradient(x, y, predictions, alpha):
    # loss: the mean of loss(cross entropy)
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predictions, labels=y)) - \
           alpha * tf.nn.l2_loss(x)
    grad, = tf.gradients(loss, x)
    # signed_grad = tf.sign(grad)
    return grad


def oneStepSearch(startX, size, label, sess, grad):

    gradient_value = sess.run(grad, feed_dict={x: startX.reshape(-1, startX.shape[0], 1),
                                               y: label.reshape(-1, 2),
                                               keras.backend.learning_phase(): 0})

    gradient_value = gradient_value.reshape(startX.shape[0])


    abs_valid_gradient = np.abs(gradient_value)
    max_gradient = max(abs_valid_gradient)

    if max_gradient == 0:
        return np.asarray([0] * startX.shape[0])
    else:
        pertur = gradient_value * (size / max_gradient )
        # pertur = gradient_value * size
        return np.asarray(pertur).reshape(startX.shape[0])


def AESearch(startFeature, step, size, label, sess, model, grad):

    count = 0
    # startFeature = startFeature.reshape(startFeature.shape[0], 1)
    while count < step + 1:
        # print("count", count)
        # print(startFeature)
        if np.not_equal(np.argmax(model.predict(startFeature.reshape(1, startFeature.shape[0], 1)).reshape(2)),
                        np.argmax(label)):
            return startFeature

        pertur = oneStepSearch(startFeature, size, label, sess, grad)
        # print(pertur)
        pertur = pertur.astype(float)

        if np.argmax(pertur) == 0:
            print("aaa")
            return map2Positive(startFeature)
        else:
            startFeature += pertur
            startFeature = map2Positive(startFeature)
        count += 1

    return startFeature

# give a set of attack target features, given back adversarial examples
def advAttack(features, steps, size, label, sess, model, grad):

    advExample = []

    for counter in range(features.shape[0]):
        print("counter = ", counter)
        # if counter > 0 and counter % 100 == 0:
        #     print("counter = ", counter)
        selectedFeature = features[counter].copy()
        tunedFeature = AESearch(selectedFeature, steps, size, label, sess, model, grad)
        advExample.append(tunedFeature.reshape(48,))

    return advExample


def advEval(model, size, steps, sess, grad):
    testDataNum = A_NUMBER_OF_TEST

    badData = pandas.DataFrame(np.random.normal(loc=0.0, scale=0.0001, size=(testDataNum, 48)))
    badData[48] = pandas.DataFrame(np.asarray([1] * testDataNum))

    badDataFeature = badData.drop([48], 1).copy().values
    badDataLabel = badData[48].copy()
    badDataLabel = to_categorical(badDataLabel, 2)


    adversarial_example = advAttack(badDataFeature, steps, size, np.asarray([0, 1]),
                                        sess, model, grad)

    adv_sample = np.asarray(adversarial_example)

    AML_result = model.evaluate(adv_sample.reshape(adv_sample.shape[0], adv_sample.shape[1], 1), badDataLabel)
    print('Afer AML Test loss:', AML_result[0])
    print('After AML Test accuracy:', AML_result[1])

    print("the overall energy consumption is: ", np.sum(np.sum(adv_sample)) / testDataNum)

    record = open('RECORD_FILE', 'a')
    recordString = "\n step:" + str(steps) + " Accucacy:" + str(AML_result[1]) + \
                   " size:" + str(np.sum(np.sum(adv_sample)) / 200.0) + "\n"
    record.write(recordString)
    record.close()


configCPU = tf.ConfigProto(device_count={'CPU': 1, 'GPU': 0})
sess = tf.Session(config=configCPU)
keras.backend.set_session(sess)

targetModel = load_model("RNN_MODEL.h5")

size = 0.01
steps = int(sys.argv[1])
alpha = int(sys.argv[2])

record = open('RECORD_FILE', 'a')
record.write("-------" + "alpha: " + sys.argv[2] + " ---------------")
record.close()

x = tf.placeholder(tf.float32, shape=(None, 48, 1))
y = tf.placeholder(tf.float32, shape=(None, 2))


prediction = targetModel(x)
grad = scaled_gradient(x, y, prediction, alpha)

advEval(targetModel, size, steps, sess, grad)
# winsound.Beep(1000, 3000)

