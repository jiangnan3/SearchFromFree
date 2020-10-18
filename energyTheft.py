import numpy as np
import random

def fractionReduce(inputArray):
    return inputArray * random.uniform(0.1, 0.8)

def varyReduce(inputArray):
    return np.multiply(inputArray, np.random.uniform(0.1, 0.8, (inputArray.shape[0], inputArray.shape[1])))


def rangeZero(inputArray):
    oneMatrix = []
    for row in range(inputArray.shape[0]):
        rowLine = [1] * inputArray.shape[1]
        zeroNum = np.random.randint(8, 49)

        if zeroNum == inputArray.shape[1]:
            zeroStartIndex = 0
        else:
            zeroStartIndex = np.random.randint(0, inputArray.shape[1] - zeroNum)

        for i in range(zeroNum):
            rowLine[zeroStartIndex + i] = 0
        oneMatrix.append(rowLine)

    return np.multiply(inputArray, np.asarray(oneMatrix))


def flatMean(inputArray):
    theftMatrix = []
    for row in range(inputArray.shape[0]):
        rowLine = [np.mean(inputArray[row])] * inputArray.shape[1]
        theftMatrix.append(rowLine)
    return np.array(theftMatrix)


def varyFlatMean(inputArray):
    return np.multiply(varyReduce(inputArray), np.random.uniform(0.1, 0.8, (inputArray.shape[0], inputArray.shape[1])))


def reverseOrder(inputArray):
    reverseMatrix = []
    for row in range(inputArray.shape[0]):
        rowLine = []
        for col in range(inputArray.shape[1]):
            rowLine.append(inputArray[row][inputArray.shape[1]-col-1])
        reverseMatrix.append(rowLine)
    return np.array(reverseMatrix)
