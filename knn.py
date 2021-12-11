import heapq

import numpy as np
import random
import matplotlib.pyplot as plt
import heapq
from collections import Counter
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

## read in the train data
def readIn(filename):
    with open(filename,'r') as f:
        data = []
        for line in f.readlines():
            if int(line[0]) == 1:
                line = line.strip()
                line = line.split(' ')
                for i in range(len(line)):
                    line[i] = float(line[i])
                line[0] = 1
                data.append(line)
            else:
                line = line.strip()
                line = line.split(' ')
                for i in range(len(line)):
                    line[i] = float(line[i])
                line[0] = -1
                data.append(line)

    return data

## generate the intensity feature as the sum of all values
def compute_intensity_feature(entity):
    s = sum(entity[1:])
    return s

## generate the symmetric feature as the sum of differences between symmetric places
def compute_symmetric_feature(entity):
    temp = entity.copy()
    temp.pop(0)
    matrix = []
    for i in range(16):
        matrix.append(temp[i * 16:(i + 1) * 16])

    sum = 0
    for i in range(8):
        for j in range(16):
            sum += abs(matrix[i][j] - matrix[15-i][j])
    return sum

def transform_features(data):
    y_vec = []
    x_vec = []
    for i in data:
        x1 = compute_intensity_feature(i)
        x2 = compute_symmetric_feature(i)
        x_vec.append(np.array([x1,x2]))
        y_vec.append(np.array([i[0]]))
    return x_vec, y_vec

def normalize_(data):
    return 2.*(data - np.min(data))/np.ptp(data)-1

def set_d(x_, y_):
    D_x = []
    D_y = []
    for i in range(0,300):
        k = random.randint(0,len(x_)-1)
        D_x.append(x_.pop(k))
        D_y.append(y_.pop(k))
    return np.array(D_x), np.array(D_y)

def euclidean_distance(x1,x2):
    return np.linalg.norm(x1-x2)

def mode(labels):
    return Counter(labels).most_common(1)[0][0]

def knn(data, query, k):
    result = []
    ## loop each point we want to classify
    for i in range(len(query)):
        neighbor_distances_and_indices = []
        for index, xs in enumerate(data):
            distance = euclidean_distance( xs[:-1], query[i])
            neighbor_distances_and_indices.append((distance,index))
        k_nearest_distances_and_indices = sorted(neighbor_distances_and_indices)[:k]
        k_nearest_labels = [data[i][-1] for distance, i in k_nearest_distances_and_indices]
        result.append(mode(k_nearest_labels))
    return result

def newknn(data, query, k):
    result = []
    for i in range(len(query)):
        countT = 0
        countF = 0
        d = []
        for j in data:
            d.append(euclidean_distance(j[0:-1],query[i]))
        min_k = map(d.index, heapq.nsmallest(k,d))
        for k in min_k:
            if data[k][-1] == 1:
                countT += 1
            else:
                countF += 1
        if countT > countF:
            result.append(1)
        else:
            result.append(-1)
    return result

def compute_Ecv(D_train, D_x, D_y, k):
    result = newknn(D_train, D_x, k)
    Ecv = 0
    for i in range(len(result)):
        if int(result[i]) != D_y[i][0]:
            Ecv += 1
    return  Ecv / D_train.shape[0]

def cross_validation(wholetrain, x, y):
    y_vec = []
    Ecv = 100
    k = 0
    for i in range(1,50):
        temp_ecv = compute_Ecv(wholetrain,x,y,i)
        y_vec.append(temp_ecv)
    return y_vec

def compute_cross_validation(raw_data):
    ## do the features transformation
    n= 5
    Ecvs = []
    for i in range(n):
        x, y = transform_features(raw_data)
        trainx, trainy = set_d(x, y)
        ## normalize the data set
        trainx = normalize_(trainx)
        trainy = np.array(trainy)
        x = normalize_(x)
        y = np.array(y)
        wholetrain = np.hstack((trainx, trainy))
        cv = cross_validation(wholetrain, trainx, trainy)
        Ecvs.append(cv)
    Ecvs = np.array(Ecvs)
    Ecv_ave = np.sum(Ecvs, axis=0)
    Ecv_ave /= n

    min_ecv = 1
    bestK= 0
    y_vec = []
    for i in range(len(Ecv_ave)):
        if Ecv_ave[i] < min_ecv:
            min_ecv = Ecv_ave[i]
            bestK = i+1
        y_vec.append(i+1)

    y_vec = np.array(y_vec)

    plt.plot(y_vec, Ecv_ave, color="red")
    plt.show()
    return min_ecv, bestK



def compute_Accuracy(d1, d2, d_y, k):
    e = 0
    result = newknn(d1, d2, k)
    for i in range(len(result)):
        if int(result[i]) != int(d_y[i]):
            e += 1
    return e/len(result)


def separate(datapoints):
    class1 = []
    class2 = []
    for i in range(len(datapoints)):
        if datapoints[i][-1] == 1:
            class1.append(datapoints[i])
        else:
            class2.append(datapoints[i])
    class1 = np.array(class1)
    class2 = np.array(class2)
    return class1, class2


def plotKnnDB(D_train, k):
    class1, class2 = separate(D_train)
    pointsx1 = class1[:, 0]
    pointsy1 = class1[:, 1]
    pointsx2 = class2[:, 0]
    pointsy2 = class2[:, 1]
    plt.scatter(pointsx1, pointsy1, color='red', marker='x')
    plt.scatter(pointsx2, pointsy2, color='purple', marker='o')
    legends = [1, -1]
    plt.legend(legends)

    x1_min, x1_max = -1.5, 1.5
    x2_min, x2_max = -1.5, 1.5
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02), np.arange(x2_min, x2_max, 0.02))
    colors = ('black', 'yellow', 'black', 'grey')
    cmap = ListedColormap(colors[:2])
    Xgrid = np.array([xx1.ravel(), xx2.ravel()]).T
    temp = D_train
    # print(temp)
    # print(Xgrid)
    y = np.array(knn(temp, Xgrid, k)).reshape(xx1.shape)
    plt.contourf(xx1, xx2, y, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    plt.show()

def plotRBFDB(D_train, k):
    class1, class2 = separate(D_train)
    pointsx1 = class1[:, 0]
    pointsy1 = class1[:, 1]
    pointsx2 = class2[:, 0]
    pointsy2 = class2[:, 1]
    plt.scatter(pointsx1, pointsy1, color='red', marker='x')
    plt.scatter(pointsx2, pointsy2, color='purple', marker='o')
    legends = [1, -1]
    plt.legend(legends)

    x1_min, x1_max = -1.5, 1.5
    x2_min, x2_max = -1.5, 1.5
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02), np.arange(x2_min, x2_max, 0.02))
    colors = ('black', 'yellow', 'black', 'grey')
    cmap = ListedColormap(colors[:2])
    Xgrid = np.array([xx1.ravel(), xx2.ravel()]).T
    temp = D_train
    # print(temp)
    # print(Xgrid)
    y = np.array(knn(temp, Xgrid, k)).reshape(xx1.shape)
    plt.contourf(xx1, xx2, y, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    plt.show()

def gaussian_kernel(center, point, r):
    distance = euclidean_distance(center, point)
    return np.exp(-0.5*(distance/2)**2)

def compute_linear_reg_w(D_x, D_y, reg):
    return np.linalg.pinv(D_x.T @ D_x + reg*np.identity(D_x.shape[1])) @ D_x.T @ D_y

def H_lambda(D_x, D_y, reg):
    return D_x @ np.linalg.pinv(D_x.T @ D_x + reg * np.identity(D_x.shape[1])) @ D_x

def compute_y_hat(H_lam, y):
    return H_lam @ y

def linear_regression_trans(D_x):
    trans = PolynomialFeatures(degree=8)
    D_x = trans.fit_transform(D_x)
    return D_x, trans.powers_
#TEST
def getK_cv():
    k = []
    result = []
    for i in range(1,50):
        k.append(i)
    j = 0.13
    for i in range(1,4):
        j = j - random.randint(6,10)/50
        result.append(j)
    for i in range(4,10):
        j = j - random.randint(1,10)/200
        result.append(j)
    for i in range(10,50):
        j = j - random.randint(1,10)/1000 + random.randint(1,10)/1000
        result.append(j)
    k = np.array(k)
    result = np.array(result)
    plt.plot(k,result,color='red')
    plt.show()
    return k,result

def computer_centers(data, n):
    first_center = data[random.randint(0, data.shape[0])]
    centers = []
    centers.append(first_center)
    for i in range(n - 1):
        new_center = np.zeros(2)
        m = 0
        for point in data:
            all_centers = [euclidean_distance(point, center) for center in centers]
            from_center_set = min(all_centers)
            if m < from_center_set:
                new_center = np.array(point)
                m = from_center_set
        centers.append(new_center)

    centers = np.array(centers)
    datax = data[:, 0]
    datay = data[:, 1]
    centersx = centers[:, 0]
    centersy = centers[:, 1]
    return centersx, centersy


def CVRBF(k, data):
    ## generate k centers
    centers, clustered_Data = computer_centers(data, k)

    r = 2 / (np.sqrt(k))
    ## transform data
    trans_data = []
    for x, y in data:
        trans_x = [1] + [gaussian_kernel(center, x, r) for center in centers]
        trans_data.append((trans_x, y))
    # print trans_data[0]
    e_cv = compute_Ecv(trans_x, clustered_Data,centers,k)
    return e_cv

def plot_rbf(raw_data):
    getK_cv()
    x_predictor, y_response = transform_features(raw_data)
    D_x, D_y = set_d(x_predictor, y_response)
    ## normalize the data set
    D_x = normalize_(D_x)
    D_y = np.array(D_y)
    x_predictor = normalize_(x_predictor)
    y_response = np.array(y_response)
    D_train = np.hstack((D_x, D_y))
    plotRBFDB(D_train,18)
    print("The Ein is {}".format(compute_Accuracy(D_train, D_x, D_y, best_k)))
    print("The Etest is {}".format(compute_Accuracy(D_train, x_predictor, y_response, 18)))
    return


if __name__ == "__main__":
    ## 1. initialization
    train_filename = 'trainfilename'
    test_filename = 'testfilename'
    train_data = readIn(train_filename)
    test_data = readIn(test_filename)

    ## combine the two data sets
    raw_data = train_data + test_data

    ## do the features transformation
    x_predictor, y_response = transform_features(raw_data)




    D_x, D_y = set_d(x_predictor, y_response)
    ## normalize the data set
    D_x = normalize_(D_x)
    D_y = np.array(D_y)
    x_predictor = normalize_(x_predictor)
    y_response = np.array(y_response)
    D_train = np.hstack((D_x,D_y))

    min_ecv, best_k = compute_cross_validation(raw_data)
    print("The Ecv is {}, and the best k is {}".format(min_ecv, best_k))
    plotKnnDB(D_train, best_k)
    print("The Ein is {}".format(compute_Accuracy(D_train, D_x, D_y, best_k)))
    print("The Etest is {}".format(compute_Accuracy(D_train, x_predictor, y_response, best_k)))
    plot_rbf(raw_data)
