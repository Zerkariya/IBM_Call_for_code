import time
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import random
import time
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.linalg import fractional_matrix_power
from sklearn.cluster import KMeans


## open the file and read in the data
## the first column info (dates)
## the first row info (attributes)
## the rest 27 columns of data (in float forms)
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

def class_separation(raw_data):
    for i in raw_data:
        if i[0] <= 40:
            i[0] = 0
        elif i[0] <= 60:
            i[0] = 1
        elif i[0] <= 100:
            i[0] = 2
        elif i[0] > 100:
            i[0] = 3

def compute_N_points(n, raw_data):
    x_ = raw_data[:,1:]
    y_ = raw_data[:,0:1].reshape(-1)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=n / x_.shape[0])
    points = None
    for train_index, test_index in sss.split(x_, y_):
        points = test_index
    partialData = np.zeros((n, raw_data.shape[1]))
    for i in range(len(points)):
        partialData[i] = raw_data[points[i]]
    return partialData

def separate_x_and_y(raw_data):
    x_predictor = raw_data[:,1:]
    y_response = raw_data[:,0]
    return x_predictor, y_response

## Calculate the norm square between two points
def Compute_2_norm_square(x1, x2):
    return (np.linalg.norm(x1 - x2))**2

def compute_gaussian_kernel_matrix(data1, data2, sigma_2):
    l = len(data1)
    m = len(data2)
    Kernel_Matrix = np.zeros((l,m))
    ## construct the kernel matrix
    for i in range(l):
        for j in range(m):
            norm_ = Compute_2_norm_square(data1[i], data2[j])
            true_spread = 2 * sigma_2
            Kernel_Matrix[i][j] = math.exp((-1)*norm_/true_spread)
    return Kernel_Matrix

def compute_U(B, k):
    eigenvalue, eigenvec = np.linalg.eig(B)
    eigenvalue = np.real(eigenvalue)
    eigenvec = np.real(eigenvec)
    eigenvalue_sort_index = np.argsort(eigenvalue)
    k_smallest = eigenvalue_sort_index[:k]
    U = np.zeros((eigenvec.shape[0], 1))
    for i in range(k):
        U = np.hstack([U, eigenvec[:,k_smallest[i]].reshape(-1,1)])
    U = np.delete(U,0,axis=1)
    return U

def compute_Y(U,k):
    each_row_coe = 1/np.sqrt(np.sum(np.square(U), axis=1))
    return each_row_coe.reshape(-1,1) * U

def spectral_clustring_algo(D, k, obj):
    ## get x predictors and y responses
    D_x, D_y = separate_x_and_y(D)
    ## set up the Adjacency Maxtrix
    AdjacencyMatrix = compute_gaussian_kernel_matrix(D_x, D_x, spread)
    degree_matrix_diagonal = np.sum(AdjacencyMatrix, axis=1)
    ## compute degree matrix
    degree_matrix = np.multiply(np.eye(AdjacencyMatrix.shape[0]), degree_matrix_diagonal)
    ## compute L
    L = degree_matrix - AdjacencyMatrix
    B = None
    if obj == 'ratio':
        B = L.copy()
    elif obj == 'asymmetric':
        L_a = np.linalg.pinv(degree_matrix) @ L
        B = L_a.copy()
    elif obj == 'symmetric':
        L_s = fractional_matrix_power(degree_matrix,-0.5) @ L @ fractional_matrix_power(degree_matrix, -0.5)
        B = L_s.copy()
    ## After we get B, we can compute U and Y
    U = compute_U(B, k)
    Y = compute_Y(U, k)
    kmeans = KMeans(n_clusters=k).fit(Y)
    Cs = kmeans.labels_
    return Y, Cs

def compute_cluster_size(D, k, prediction):
    ## where D_y is the true clustering labels
    D_x, D_y = separate_x_and_y(D)
    ## we need to building up the
    D_y = D_y.astype(int)
    binary_classifier = np.zeros((max(k,4), max(k,4)))
    for i in range(len(prediction)):
        binary_classifier[prediction[i]][D_y[i]] += 1
    print("The size of each cluster is classified as follows:")
    for i in range(k):
        print("The size for class {} is {}".format(i, np.sum(binary_classifier[i])))

def compute_F_score(D,k,prediction):
    ## where D_y is the true clustering labels
    D_x, D_y = separate_x_and_y(D)
    ## we need to building up the
    D_y = D_y.astype(int)
    binary_classifier = np.zeros((max(k, 4), max(k, 4)))
    for i in range(len(prediction)):
        binary_classifier[prediction[i]][D_y[i]] += 1

    precisions = np.zeros(k)
    for i in range(k):
        if (np.sum(binary_classifier[i]) == 0):
            precisions[i] = 0
        else:
            precisions[i] = np.max(binary_classifier[i])/np.sum(binary_classifier[i])

    recalls = np.zeros(k)
    for i in range(k):
        maxT = binary_classifier[i][np.argmax(binary_classifier[i])]
        recalls[i] = maxT / np.sum(binary_classifier[:,np.argmax(binary_classifier[i])])

    F = 0
    for i in range(k):
        if (precisions[i] + recalls[i] == 0):
            F += 0
        else:
            F += (2*precisions[i]*recalls[i])/(precisions[i]+recalls[i])
    F /= k
    return F

if __name__ == '__main__':
    ## read in and get the dates, attributes' name, and data
    filename = sys.argv[1]
    k = int(sys.argv[2])
    n = int(sys.argv[3])
    spread = float(sys.argv[4])
    obj = sys.argv[5]
    dates, attributes, data = ReadiInData(filename)[0], \
                              ReadiInData(filename)[1], \
                              ReadiInData(filename)[2]
    ## separate the data depends on their appliance
    class_separation(data)
    ## using stratified sampling to sample from each class
    partialData = compute_N_points(n, data)
    ## doing the spectral clustring algorithm
    Y, Cs = spectral_clustring_algo(partialData,k,obj)
    ## compute the size of clusters
    print("The obj is {}".format(obj))
    compute_cluster_size(partialData,k,Cs)
    print("The F measure is {}".format(compute_F_score(partialData,k,Cs)))

    ######
    ## the following part is to choose the best spread value and compute the highest f-measure
    # rec = 0
    # f = 0
    # partialData = compute_N_points(n, data)
    # for i in tqdm(range(500,2000)):
    #     spread = i
    #     ## doing the spectral clustring algorithm
    #     Y, Cs = spectral_clustring_algo(partialData, k, obj)
    #     F_score = compute_F_score(partialData,k,Cs)
    #     if F_score > f:
    #         f = F_score
    #         rec = i
    # print(f, rec)

    ## test
    # for i in range(10):
    #     partialData = compute_N_points(n, data)
    #     ## doing the spectral clustring algorithm
    #     Y, Cs = spectral_clustring_algo(partialData, k, obj)
    #     ## compute the size of clusters
    #     print("The obj is {}".format(obj))
    #     compute_cluster_size(partialData, k, Cs)
    #     print("The F measure is {}".format(compute_F_score(partialData, k, Cs)))
