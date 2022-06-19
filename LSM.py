from sklearn import datasets  
import numpy as np  
import matplotlib.pyplot as plt
from numba import jit
n = 400
np.random.seed(0)  
XX, labels = datasets.make_moons(n,noise=.1)
YY = labels.reshape(n)

import math
def rho(u): 
    return 1/(1 + np.exp(-u))
def rho_derivative(u): 
    return rho(u)*(1-rho(u))

# To run on GPU
@jit
def forwardpass(x,wh,wo):
    zh = np.dot(wh, x)
    xh = rho(zh)
    zo = np.dot(wo.T, xh)
    xo = rho(zo)[0]
    return zh, xh, zo, xo

# To run on GPU
@jit
def backwardpass(x,y,zh,xh,zo,xo, wo, wh):
    dE_dx0  = xo - y 
    dx0_dz0 = rho_derivative(zo)[0]
    dz0_dwo = np.reshape(xh, (np.shape(xh)[0], 1))
    dE_dwo  = dE_dx0 * dx0_dz0 * dz0_dwo

    dz0_dxh = wo.T
    dxh_dzh = np.reshape(rho_derivative(zh), (np.shape(xh)[0], 1))
    dzh_dwh = np.reshape(x, (1,2))
    dE_dwh  = dz0_dxh @ dxh_dzh @ dzh_dwh
    return dE_dwo, dE_dwh

# To run on GPU
@jit
def SGD(wh,wo,dE_dwo,dE_dwh,rate):
    #print('---------', np.shape(wh), np.shape(dE_dwh), np.shape(wo), np.shape(dE_dwo))
    wh = wh - rate * dE_dwh
    wo = wo - rate * dE_dwo.T
    return wh,wo

m = 15
rate = .1
iter = 200000

# To run on GPU
@jit
def train(x, y):
    n = np.shape(x)[0]
    d = np.shape(x)[1]
    ypred = np.zeros((n, iter))
    error = []
    wh = np.random.rand(m, d) # xh [m] = rho( wh   [m , d] * x [d, 1])
    wo = np.random.rand(m, 1) # x0 [1] = rho( wo.T [1 , m] * xh [m])

    for iterations in range(0, iter):

        for i in range(0, n):
            x_  = x[i,:].T
            y_  = y[i]
            zh, xh, zo, xo = forwardpass(x_, wh, wo)
            dE_dwo, dE_dwh = backwardpass(x_,y_,zh,xh,zo,xo, wo, wh)
            wh, wo         = SGD(wh,wo,dE_dwo,dE_dwh,rate)
            ypred[i, iterations] = xo

        if iterations % 1000 == 0:
            #print(np.shape(wo), np.shape(wh))
            err = (1/2) * (y - ypred[:,iterations])**2 
            err = np.mean(err)
            error = np.append(error, err)
            print('Error in iteration {} is {}'.format(iterations, err))
            
        if iterations % 10000 == 0:
            print(wo)
            print(wh)

    return wo, wh, error

wo, wh, error = train(XX, YY)

np.save('/home/apdl006/LSM/wo.npy', wo)
np.save('/home/apdl006/LSM/wh.npy', wh)
np.save('/home/apdl006/LSM/error.npy', error)

print(wo, wh)
print('\n')
print(error)
