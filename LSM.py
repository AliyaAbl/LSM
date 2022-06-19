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
    return 1/(1 + math.exp(-u))
def rho_derivative(u): 
    return rho(u)*(1-rho(u))

# To run on GPU
@jit
def forwardpass(x,wh,wo):
    #print(np.shape(x), np.shape(wh), np.shape(wo))
    zh = wh @ x
    rho_v = np.vectorize(rho) 
    xh = rho_v(np.array(zh))
    zo = wo.T @ xh
    zo = zo[0][0]
    xo = rho(zo)
    return zh, xh, zo, xo

# To run on GPU
@jit
def backwardpass(x,y,zh,xh,zo,xo, wo):
    dE_dx0  = - (y - xo)
    dE_dx0  = np.reshape(dE_dx0,  (1,1))
    dx0_dz0 = rho_derivative(zo) 
    dx0_dz0 = np.reshape(dx0_dz0, (1,1))
    dz0_dwo = xh.T
    #print(np.shape(dE_dx0), np.shape(dx0_dz0), np.shape(dz0_dwo))
    dE_dwo  = dE_dx0 @ dx0_dz0 @ dz0_dwo
    #print(np.shape(dE_dwo))

    dz0_dxh = wo.T
    rho_derivative_v = np.vectorize(rho_derivative)
    dxh_dzh = rho_derivative_v(zh)
    dzh_dwh = x.T
    #print('---',np.shape(dE_dx0), np.shape(dx0_dz0), np.shape(dz0_dxh), np.shape(dxh_dzh), np.shape(dzh_dwh))
    dE_dwh  = dE_dx0 @ dx0_dz0 @ dz0_dxh @ dxh_dzh @ dzh_dwh
    #print(np.shape(dE_dwh))
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
            x_  = np.reshape(x_, (d, 1))
            zh, xh, zo, xo = forwardpass(x_, wh, wo)
            dE_dwo, dE_dwh = backwardpass(x_,y_,zh,xh,zo,xo, wo)
            wh, wo         = SGD(wh,wo,dE_dwo,dE_dwh,rate)
            ypred[i, iterations] = xo

        if iterations % 100 == 0:
            err = (1/n) * (y - ypred[:,iterations])**2
            err = np.mean(err)
            error = np.append(error, err)
            print('Error in iteration {} is {}'.format(iterations, err))
            #print(ypred[1:5, iterations], ypred[1:5, iterations-10] )
    return wo, wh, error

wo, wh, error= train(XX, YY)
print(wo, wh)
print('\n')
print(error)
