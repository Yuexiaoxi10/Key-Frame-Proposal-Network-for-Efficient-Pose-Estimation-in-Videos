import os
import numpy as np
import random
from math import isclose
import torch
import matplotlib.pyplot as plt
from modelZoo.DyanOF import OFModel, fista
from torch.autograd import Variable
import torch.nn


def gridRing(N):
    # epsilon_low = 0.25
    # epsilon_high = 0.15
    # rmin = (1 - epsilon_low)
    # rmax = (1 + epsilon_high)

    epsilon_low = 0.25
    epsilon_high = 0.15
    rmin = (1 - epsilon_low)
    rmax = (1 + epsilon_high)

    thetaMin = 0.001
    thetaMax = np.pi / 2 - 0.001
    delta = 0.001
    # Npole = int(N / 4)
    Npole = int(N/2)
    Pool = generateGridPoles(delta, rmin, rmax, thetaMin, thetaMax)
    M = len(Pool)

    idx = random.sample(range(0, M), Npole)
    P = Pool[idx]
    Pall = np.concatenate((P, -P, np.conjugate(P), np.conjugate(-P)), axis=0)

    return P, Pall


## Generate the grid on poles
def generateGridPoles(delta, rmin, rmax, thetaMin, thetaMax):
    rmin2 = pow(rmin, 2)
    rmax2 = pow(rmax, 2)
    xv = np.arange(-rmax, rmax, delta)
    x, y = np.meshgrid(xv, xv, sparse=False)
    mask = np.logical_and(np.logical_and(x ** 2 + y ** 2 >= rmin2, x ** 2 + y ** 2 <= rmax2),
                          np.logical_and(np.angle(x + 1j * y) >= thetaMin, np.angle(x + 1j * y) <= thetaMax))
    px = x[mask]
    py = y[mask]
    P = px + 1j * py

    return P

def getRowSparsity(inputDict):
    rowNum = inputDict.shape[0]
    L = inputDict.shape[1]
    count = 0
    for i in range(0, rowNum):
        dictRow = inputDict[i,:].unsqueeze(0)
        if len(dictRow.nonzero()) <= round(0.6*L):
            count+=1
        else:
            continue
    rowSparsity = count
    return rowSparsity


def get_recover_fista(D, y, key_set, param, gpu_id):
    if type(D) is np.ndarray:
        D = torch.Tensor(D)

    D_r = D[key_set]

    if len(y.shape)==3:
        y_r = y[:,key_set]
    else:
        y_r = y[key_set]

    if D.is_cuda:
        c_r = fista(D_r, y_r, param, 100, gpu_id)
        y_hat = torch.matmul(D, c_r)
    else:
        c_r = fista(D_r.cuda(gpu_id), y_r, param, 100, gpu_id)
        y_hat = torch.matmul(D.cuda(gpu_id), c_r)

    return y_hat

