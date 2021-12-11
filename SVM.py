import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import random
from sklearn.preprocessing import StandardScaler

## open the file and read in the data

def ReadiInData(filename):
    ## get the first column dates
    dates = np.loadtxt(filename, dtype=str, delimiter=',', skiprows=1, usecols=0, encoding='utf-8')
    with open(filename, encoding='utf-8') as f:
        data = np.loadtxt(filename, dtype=str, delimiter=',')

    ## get the attributes
    attributes = data[0]
    data = np.delete(data, 0, axis=0)
    data = np.delete(data, [0, -1], axis=1)
    # clean the data
    # loop over each row of the data from second row
    for i in range(0, len(data)):
        # loop each column of the data from the second column
        for j in range(0, len(data[i])):
            ## remove the quotes sign and the blank space
            data[i][j] = eval(data[i][j]).strip()
    ## convert the data type from string to float
    data = data.astype(np.float_)
    return dates, attributes, data

## Normalize the data to let mean = 0 and std = 1
def Normalize_Data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)

## get the X attributes and Y response
def Get_predictor_response(data):
    D = data[:, 1:]

    Y_response = data[:, 0]
    for i in range(len(Y_response)):
        if Y_response[i] <= 50:
            Y_response[i] = 1
        else:
            Y_response[i] = -1
    return D, Y_response

## Separate data in training dataset, validation, and test dataset
def Separate_Data(data):
    train_data = data[0:5000]
    validation = data[5000:5000 + 2000]
    test_data = data[7000:7000+5000]
    return train_data, validation, test_data

## Calculate the norm square between two points
def Compute_2_norm_square(x1, x2):
    return (np.linalg.norm(x1 - x2))**2

## Set up the Linear Kernel Matrix
def Compute_Linear_Kernel_Matrix(data1, data2):
    l = len(data1)
    m = len(data2)
    Kernel_Matrix = np.zeros((l,m))

    ## Construct Linear Kernel Matrix
    for i in range(l):
        for j in range(m):
            Kernel_Matrix[i][j] = data1[i] @ data2[j]
    return Kernel_Matrix

## Set up the Gaussian Kernel Matrix
def Compute_Gaussian_Kernel_Matrix(data1, data2, sigma_2):
    l = len(data1)
    m = len(data2)
    Kernel_Matrix = np.zeros((l,m))

    ## Construct Kernel Matrix
    for i in range(l):
        for j in range(m):
            norm_ = Compute_2_norm_square(data1[i], data2[j])
            k = 2 * sigma_2
            Kernel_Matrix[i][j] = math.exp((-1) * norm_/k)

    return Kernel_Matrix
## Dual SVM Algorithm: SGA
def SVM_dual(K, loss, C, y_response, eps, maxiter):
    K_aug = K+1
    eta = [] ## in form of [(index, 1/K(xk,xk)].....]
    n = K.shape[0]
    index = []
    for k in range(n):
        eta.append(1/K_aug[k][k])
        index.append(k)
    eta = np.array(eta)
    t = 0
    a_t = np.zeros(n)
    y_response = np.array(y_response)
    while t <= maxiter:
        a_new = a_t.copy()
        random.shuffle(index)
        ## loop over eta, so we use the first value in each eta's element to get the random index
        for k in index:
            temp = a_new * y_response
            temp = temp @ K_aug[:,k]
            # for i in range(n):
            #     temp += a_new[i]*y_response[i]*K[i][index]
            a_new[k] = a_new[k] + eta[k]*(1-y_response[k]*temp)
            if a_new[k] < 0:
                a_new[k] = 0
            if a_new[k] > C:
                a_new[k] = C
        if np.linalg.norm(a_new - a_t) <= eps:
            return a_new
        else:
            a_t = a_new
            t += 1
    return a_t


## Compute the number of SV
def Compute_SV(a_t, C_value):
    sv = 0
    for i in range(len(a_t)):
        if a_t[i] > 0 and a_t[i] < C_value:
            sv += 1
    return sv


## Compute the accuracy on the validation data set
def Compute_vali_accuracy(a, K, y_train_response, y_vali_response):
    K_aug = K+1
    y_hat = np.zeros(K.shape[1])
    ## looping all of the columms
    # for j in range(K.shape[1]):
    #     y_hat_temp = 0
    #     for i in range(K.shape[0]):
    #         if a[i] > 0:
    #             y_hat_temp += a[i]*y_train_response[i]*K_aug[i][j]
    #     y_hat[j] = y_hat_temp
    for j in range(K_aug.shape[1]):
        y_hat_temp = a*y_train_response
        y_hat_temp = y_hat_temp @ K_aug[:,j]
        y_hat[j] = y_hat_temp
    # for i in range(K.shape[0]):
    #     temp = sum(K[i])
    #     y_hat += a[i]*y_train_response[i]*temp
    missclassified = 0
    for i in range(len(y_hat)):
        if y_hat[i] * y_vali_response[i] < 0 or y_hat[i] == 0:
            missclassified += 1
    error = missclassified/len(y_vali_response)


    return 1-error


if __name__ == '__main__':
    ## read in and get the dates, attributes' name, and data
    filename = sys.argv[1]
    C = float(sys.argv[2])
    eps = float(sys.argv[3])
    maxiter = int(sys.argv[4])
    Kernel = sys.argv[5]
    Kernel_param = float(sys.argv[6])

    dates, attributes, data = ReadiInData(filename)[0], \
                              ReadiInData(filename)[1], \
                              ReadiInData(filename)[2]
    ## 1. separate the data set into three categories.
    ## Get the predictor and response Y
    D, Y_response = Get_predictor_response(data)
    ## Normalize the data for mean = 0 and std = 1
    D = Normalize_Data(D)
    ## Separate the data
    D_train, D_vali, D_test = Separate_Data(D)
    Y_train, Y_vali, Y_test = Separate_Data(Y_response)
    ## 2. Compute Linear and Gaussian Kernel Matrix for each sub dataset.
    ## 2.1 Linear Kernel Matrix
    if Kernel == "linear":
        Linear_D_train_Kernel = Compute_Linear_Kernel_Matrix(D_train, D_train) # 5000x5000
        Linear_D_vali_Kernel = Compute_Linear_Kernel_Matrix(D_train, D_vali) # 5000x2000
        Linear_D_test_Kernel = Compute_Linear_Kernel_Matrix(D_train, D_test) # 5000x5000

        ## compute accuracy and sv number
        a = SVM_dual(Linear_D_train_Kernel, 'hinge', C, Y_train, eps, maxiter)
        print("Accuracy for validation: ", Compute_vali_accuracy(a, Linear_D_vali_Kernel, Y_train, Y_vali))
        print("Accuracy for test: ", Compute_vali_accuracy(a, Linear_D_test_Kernel, Y_train, Y_test))
        print("SV number: ", Compute_SV(a, C))
        ## Augmented train dataset for w and b
        const_1 = np.ones(D_train.shape[0])
        D_train = np.c_[const_1, D_train]
        temp = a * Y_train
        w = np.ones(D_train.shape[1])
        for i in range(D_train.shape[0]):
            w += temp[i] * D_train[i]
        b = w[0]
        print("The bias is :", b)
        print("The weight vector is :", w)

        ## Used to find the best C for best accuracy
        '''
        i = 0.001
        while i <= 0.1:
            print("C = ",i)
            a = SVM_dual(Linear_D_train_Kernel, 'hinge', i, Y_train, eps, maxiter)
            print("Accuracy: ", Compute_vali_accuracy(a, Linear_D_vali_Kernel, Y_train, Y_vali))
            print("SV number: ",Compute_SV(a))
            i += 0.005
        '''
    ## print(Compute_vali_accuracy(a, Linear_D_test_Kernel, Y_train, Y_test))
    ## 2.2 Gaussian Kernel Matrix
    elif Kernel == "gaussian":
        Gaussian_D_train_Kernel = Compute_Gaussian_Kernel_Matrix(D_train, D_train, Kernel_param)
        Gaussian_D_vali_Kernel = Compute_Gaussian_Kernel_Matrix(D_train, D_vali, Kernel_param)
        Gaussian_D_test_Kernel = Compute_Gaussian_Kernel_Matrix(D_train, D_test, Kernel_param)
        a = SVM_dual(Gaussian_D_train_Kernel, 'hinge', C, Y_train, eps, maxiter)
        print("Accuracy: ", Compute_vali_accuracy(a, Gaussian_D_vali_Kernel, Y_train, Y_vali))
        print("Accuracy: ", Compute_vali_accuracy(a, Gaussian_D_test_Kernel, Y_train, Y_test))
        print("SV: ", Compute_SV(a, C))

        ## Used to find the best C and spread value for best accuracy
        '''
        c_value = [18,19,20]
        spread_value = [630,640,650,660]
        for c in c_value:
            for spread in spread_value:
                print("C: {}, Spread: {}".format(c,spread))
                Gaussian_D_train_Kernel = Compute_Gaussian_Kernel_Matrix(D_train, D_train, spread)
                Gaussian_D_vali_Kernel = Compute_Gaussian_Kernel_Matrix(D_train, D_vali, spread)
                a = SVM_dual(Gaussian_D_train_Kernel, 'hinge', c, Y_train, eps, maxiter)
                print("Accuracy: ", Compute_vali_accuracy(a, Gaussian_D_vali_Kernel, Y_train, Y_vali))
                print("SV: ", Compute_SV(a, c))
                ## print(Compute_vali_accuracy(a, Gaussian_D_test_Kernel, Y_train, Y_test, Kernel))
        '''
