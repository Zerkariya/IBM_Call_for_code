import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import random
from sklearn.preprocessing import StandardScaler

## open the file and read in the data
def ReadInData(filename):
    with open(filename, encoding='utf-8') as f:
        data = np.loadtxt(filename, dtype=str, delimiter=',')

    useless_attributes = [0, 1, 2, 4, 5, 6, 8, 9, 10,11, 14, 15, 16, 17, 18, 19, 20,
                           21, 22, 23, 24, 25,26, 30, 31, 32, 33, 34, 35, 36,37, 38, 39, 40]
    ## remove the first line of attributes
    data = np.delete(data, 0, axis=0)
    ## remove the attributes such as DATES and SENTENCE_COURT_FACILITY
    data = np.delete(data, useless_attributes, axis=1)

    ## loop over each element and remove the quotes sign and the blank space
    for i in range(0, data.shape[0]):
        for j in range(0, data.shape[1]):
            data[i][j] = eval(data[i][j]).strip()

    return data
## Normalize the data to let mean = 0 and std = 1
def Normalize_Data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)
## get the X attributes and Y response
def Get_predictor_response(data):
    ## y_response represents the DISPOSITION_CHARGED_CLASS
    ## it means how serious the suspicious is charged
    y_response = data[:, 13:14]
    ## the D represents the rest of data points after removed the y response
    D = np.delete(data,13,axis=1)
    ## augment the data and let the first column serve as the bias term
    const_1 = np.ones(D.shape[0])
    D = np.c_[const_1, D]
    return D, y_response

## separate the data in training dataset, validation, and test dataset
def Separate_Data(data):
    train_data = data[0:20000]
    validation = data[20000: 25000]
    test_data = data[25000:45000]
    return train_data, validation, test_data

## using the Ridge Regression: Stochastic Gradient Descent
def train_and_solve_w(D_train, Y_train, alpha, eta, eps, maxiter):
    t = 0
    w_t = np.ones(D_train.shape[1]).reshape(-1,1)
    converged = False
    while (not converged) and t < maxiter:
        gradient_w = - D_train.T @ Y_train \
                     + (D_train.T @ D_train) @ w_t \
                     + alpha * w_t
        w_new_t = w_t - (eta * gradient_w)
        t += 1
        converged = (np.linalg.norm(w_new_t - w_t)) <= eps
        w_t = w_new_t
    ## return weights and whether the algo converged
    return w_t, converged

## get SSE
def Compute_SSE(D, Y, w_t):
    observed_y = Y
    predicted_y = D @ w_t.reshape(-1,1)
    return np.linalg.norm(observed_y - predicted_y)**2

## get TSS
def Compute_TSS(Y):
    y_mean = np.mean(Y)
    TSS = 0
    for i in range(Y.shape[0]):
            TSS += (Y[i][0] - y_mean)**2
    return TSS

## return SSE if the iterations converged
def validation(D_vali, Y_vali, w_t, converged):
    if (not converged):
        return 10**6
    else:
        SSE = Compute_SSE(D_vali, Y_vali, w_t)
        return SSE

## return SSE, TSS, R_square for test data
def test(D_test, Y_test, w_t):
    SSE = Compute_SSE(D_test, Y_test, w_t)
    TSS = Compute_TSS(Y_test)
    R_square = (TSS - SSE)/TSS
    return SSE, TSS, R_square

## do the output
def output(D_vali, Y_vali, D_test, Y_test , w_t, check):
    print("The w_t is \n{}".format(w_t))
    print("The SSE for validation dataset is {}".format(validation(D_vali, Y_vali, w_t, check)))
    print("The SSE for test dataset is {}".format(test(D_test, Y_test, w_t)[0]))
    print("The TSS for test dataset is {}".format(test(D_test, Y_test, w_t)[1]))
    print("The R square for test dataset is {}".format(test(D_test, Y_test, w_t)[2]))

## Used to generate the best alpha and eta
def compute_best(D_train, Y_train,D_vali, Y_vali, eps, maxiter):
    ## eps is set as 0.00001
    i = 350
    j = 0.00001
    a = []
    final_i = 0
    final_j = 0
    final_sse = 100000
    ## set up a list for alphas
    while (i <= 450):
        a.append(i)
        i += 1
    ## loop and find the smallest SSE
    for p in a:
        w_t, check = train_and_solve_w(D_train, Y_train, p,  j, eps, maxiter)
        sse = Compute_SSE(D_vali, Y_vali, w_t)
        if sse < final_sse and check:
            final_sse = sse
            final_i = p
            final_j = j
    ## return alpha, eta, and sse
    return final_i,final_j, sse


if __name__ == '__main__':
    ## read in and get the data
    filename = 'Sentencing.csv'
    alpha = 378
    eta = 1.2
    eps = 0.001
    maxiter = 1000
    data = ReadInData(filename)

    ## Normalize the data for mean = 0 and std = 1
    data = Normalize_Data(data)

    ## Get the predictor and response Y
    D, Y_response = Get_predictor_response(data)

    ## seperate the data
    D_train, D_vali, D_test = Separate_Data(D)
    Y_train, Y_vali, Y_test = Separate_Data(Y_response)
    w_t, check = train_and_solve_w(D_train, Y_train, alpha, eta, eps, maxiter)

    print("The given parameters are:\nAlpha: {}\nEta: {}\nEps: {}\nMaxiter: {}".format(alpha, eta, eps, maxiter))
    output(D_vali, Y_vali, D_test, Y_test , w_t, check )


