import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import random
import pandas as pd
from sklearn.preprocessing import StandardScaler

## open the file and read in the data
def ReadInData(filename):
    missing_value_formats = ["n.a.", "?", "NA", "n/a", "na", "--"]
    raw_data = pd.read_csv(filename, dtype=str, na_values=missing_value_formats)
    raw_data = np.array(raw_data)
    # print(raw_data[4][16])
    # print(type(raw_data[4][16]))

    ## remove the first line of attributes
    ## data = np.delete(data, 0, axis=0)
    ## remove the attributes such as DATES and SENTENCE_COURT_FACILITY
    ## data = np.delete(data, useless_attributes, axis=1)
    return raw_data

def GetUniqueAttri(cate_attri):
    return np.unique(cate_attri)

def output_attributes(raw_data):
    result = 0
    for i in range(raw_data.shape[1]):
        attri = GetUniqueAttri(raw_data[:,i])
        print(i)
        print(attri)
        print(attri.shape)
        result += attri.shape[0]
        print("+++++++++++++++++++++++++++++")
        print("+++++++++++++++++++++++++++++")
        print("+++++++++++++++++++++++++++++")

    print(result)



def Get_50000(raw_data):
    num_delete = raw_data.shape[0] - 50000
    to_delete = set()
    while len(to_delete) < num_delete:
        to_delete.add(np.random.randint(0, raw_data.shape[0]))
    to_delete = np.array(list(to_delete))
    return np.delete(raw_data, to_delete, axis=0)


def CleanData(raw_data):
    irrelevant_attributes = [0,1,2,5,6,9,10,11,12,14,16,19,20,21,24,32,33,35,36,37,38,39]
    # 1. remove the irrelevant attributes
    result = np.delete(raw_data, irrelevant_attributes, axis=1)
    # 2. we only want the primary charge flag and current sentence flag as true and remove all other false cases
    false_mask = (result != "false").all(axis=1)
    result = result[false_mask,:]
    result = np.delete(result, [1,9], axis=1)
    result = result.astype(np.str_)
    ## remove all of the cases with blank cells
    false_mask = (result != "nan").all(axis=1)
    result = result[false_mask,:]
    ## For Disposition charged class, we only want M, X, 1, 2, 3, 4, A, B, and C. All the others should be removed.
    ## For Commitment_unit class, we remove the term in "term", "dollars", and "pounds"
    wrong_letters = ["O", "P", "Z", "Pounds", "Dollars", "Term"]
    for i in wrong_letters:
        false_mask = (result != i).all(axis=1)
        result = result[false_mask,:]
    # 3. separate the dataset into two dataset, based on the black and the other races
    black = []
    index = []
    for i in range(result.shape[0]):
        if result[i][12] == "Black":
            black.append(result[i])
            index.append(i)
    result = np.delete(result, index, axis=0)
    black = np.array(black)

    print("result 0:",result[0])
    print("black shape:",black.shape)
    print("other shape:",result.shape)
    # 4. At this time, there are 103729 cases in black and 52085 cases in others
    #    I choose to evenly use 50k cases in each dataset.
    #    Basically, using the first 30k cases for training, 5k cases for validation, and 15k cases for testing
    black = Get_50000(black)
    result = Get_50000(result)

    # 5. Check point, the number point in black and others should be 50k
    # print(black.shape)
    # print(result.shape)
    return black, result

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


if __name__ == '__main__':
    ## read in and get the data
    filename = 'Sentencing.csv'
    data = ReadInData(filename)
    ## pd.set_option('display.max_columns', 500)
    black_d, other_d = CleanData(data)
    output_attributes(black_d)






