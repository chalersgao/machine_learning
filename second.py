import numpy as np
import python_speech_features as mfcc
from sklearn import preprocessing

def calculate_delta(array):
    rows, cols = array.shape
    deltas = np.zeros((rows, 0))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i - j < 0:
                first = 0
            else:
                first = i - j
            if i + j >rows - 1:
                second = rows -1
            else:
                second = i + j
            index.append((second, first))
            j += 1
        deltas[i] = (array[index[0][0]] - array[index[0][1]] + (2 * (array[index[1][0]])))
    return deltas
def extract(audio, rate):
    mfcc_feat = mfcc.mfcc(audio, rate, 0.025, 0.01, appendEnergy=True)
    mfcc_feat = preprocessing.scale(mfcc_feat)
    delta = calculate_delta(mfcc_feat)
    combined = np.hstack((mfcc_feat, delta))
    return combined