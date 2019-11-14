import math
import pandas as pd
import numpy as np
import os

from sampen import sampen2, normalize_data
from scipy.signal import argrelmin, argrelmax, butter, lfilter, freqz
from sklearn import svm
from sklearn.model_selection import train_test_split



ACCELEROMETER_FILE_NAME = '/acce.txt'
GYROSCOPE_FILE_NAME = '/gyro.txt'
MAGNETOMETER_FILE_NAME = '/magnet.txt'
PILOT = 'ariel'
LENGTH_OF_VECTOR = 250
PATH = 'datum'





def generate_datasets (path_list):
   # read the csv files
   dataset = []

   for path in path_list:

      accdata = pd.read_csv(path + ACCELEROMETER_FILE_NAME,sep=" ",header=1, usecols=[1,2,3],names = ['ax','ay','az'])
      accdata = pd.DataFrame(accdata)
      accdata['index'] = accdata.index


      gyrdata = pd.read_csv(path + GYROSCOPE_FILE_NAME, sep=" ",header=1, usecols=[1,2,3], names = ['gx','gy','gz'])
      gyrdata = pd.DataFrame(gyrdata)
      gyrdata['index'] = gyrdata.index


      magdata = pd.read_csv( path + MAGNETOMETER_FILE_NAME,sep=" ",header=1, usecols=[1,2,3],names=['mx', 'my', 'mz'])
      magdata = pd.DataFrame(magdata)
      magdata['index'] = magdata.index


      dataframe = pd.merge (accdata, gyrdata)
      dataframe = pd.merge(dataframe, magdata)

      if PILOT in path:  dataframe['label'] = 1
      else: dataframe['label'] = 0
      dataset.append(dataframe)
      # if len(dataset) < 250:
      #  raise RuntimeError(path + "File is too short")

   return dataset


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n



def get_pathnames (path) :

   folders = []
   for root, dirs, files in os.walk(path):
      folders.append (root)

   return folders[1:]



def butter_mag (dataset):

   temp = []
   for data in dataset :
      for c in ['ax', 'ay', 'az', 'gx','gy','gz','mx','my','mz']:
          column_c = data[c].tolist()
          data[c] = pd.Series(moving_average(column_c,3))


      data['am'] = (data['ax'] ** 2 + data['ay'] ** 2 + data['az'] ** 2) ** 0.5
      data['gm'] = (data['gx'] ** 2 + data['gy'] ** 2 + data['gz'] ** 2) ** 0.5
      data['mm'] = (data['mx'] ** 2 + data['my'] ** 2 + data['mz'] ** 2) ** 0.5

      index = ['index', 'ax', 'ay', 'az', 'am', 'gx', 'gy', 'gz', 'gm', 'mx', 'my', 'mz', 'mm', 'label']
      temp.append(data[index])
   return temp



# from numpy.lib.stride_tricks import as_strided
#
#
# def windowed_view(arr, window, overlap):
#     arr = np.asarray(arr)
#     window_step = window - overlap
#     new_shape = arr.shape[:-1] + ((arr.shape[-1] - overlap) // window_step, window)
#     new_strides = (arr.strides[:-1] + (window_step * arr.strides[-1],) + arr.strides[-1:])
#     return as_strided(arr, shape=new_shape, strides=new_strides)
#
#
#
# def windows(dt, ws, ol):
#     r = np.arange(len(dt))
#     s = r[::ol]
#     z = list(zip(s, s + ws))
#     f = '{0[0]}:{0[1]}'.format
#     g = lambda ol: dt.iloc[ol[0]:ol[1]]
#     return pd.concat(map(g, z), keys=map(f, z))








def window_data (results):
    len_of_window = int(LENGTH_OF_VECTOR/5 + LENGTH_OF_VECTOR/15)
    len_of_step = int(LENGTH_OF_VECTOR/5)
    i = 0
    all_features = []

    while i+len_of_window <= len(results):
        all_features.extend(get_features_from_window(results[i:i+len_of_window]))
        i += len_of_step
    return all_features




def get_features_from_window(arr):
    features = []

    for i in range(len(arr[0])):
        # 1 - Max of each n
        mx, mxindex = get_max(arr, i)
        features.append(mx)

        # 2 - min of each n
        mn, mnindex = get_min(arr, i)
        features.append(mn)

        # 3 - mean of each n
        mean = get_mean(arr, i)
        features.append(mean)

        # 4 - variance of each n
        variance = get_variance(arr, i, mean)
        features.append(variance)

        # 5 - kurtosis of each n
        kurtosis = get_kurtosis(arr, i, mean, math.sqrt(variance))
        features.append(kurtosis)

        # 6 - skewness of each n
        skew = get_skew(arr, i, mean, math.sqrt(variance))
        features.append(skew)

        # 7 - peak to peak signal
        spp = mx - mn
        features.append(spp)

        # 8 - peak to peak time
        tpp = mxindex + mnindex
        features.append(tpp)

        # 9 - peak to peak slope
        if tpp == 0:
            features.append(spp)
        else:
            spps = spp / tpp
            features.append(spps)

        # 10 - ALAR
        if mx == 0:
            features.append(0)
        else:
            features.append(mxindex/mx)

        # 11 - Energy
        features.append(get_energy(arr, i))

        # 12 - Entropy
        features.append(get_sampen(arr,i)[1][1])


    return features


def get_max(arr, i):
    curr_max = 0
    index = 0

    for x in range(len(arr)):
        if arr[x][i] > curr_max:
            curr_max = arr[x][i]
            index = x

    return curr_max, index


def get_min(arr, i):
    curr_min = 9999999999999
    index = 0

    for x in range(len(arr)):
        if arr[x][i] < curr_min:
            curr_min = arr[x][i]
            index = x

    return curr_min, index


def get_mean(arr, i):
    total = 0
    size = 0

    for data in arr:
        total += data[i]
        size += 1

    return total/size


def get_variance(arr, i, mean):
    variance = 0

    for data in arr:
        variance += (data[i] - mean) ** 2

    return variance/len(arr)


def get_kurtosis(arr, i, mean, std_dev):
    kurtosis = 0

    for data in arr:
        kurtosis += (data[i] - mean) ** 4 / len(arr)

    return kurtosis / (std_dev ** 4)


def get_skew(arr, i, mean, std_dev):
    skew = 0

    for data in arr:
        skew += (data[i] - mean) ** 3 / len(arr)

    return skew / (std_dev ** 3)


def get_energy(arr, i):
    energy = 0

    for data in arr:
        energy += data[i] ** 2

    return energy


def get_sampen(arr, i):
    sampen = []

    for data in arr:
        if data[i] == 0 : print(data)
        sampen.append(data[i])

    sampen = normalize_data(sampen)
    return sampen2(sampen)





def extract_features (dat) :


    # for every iteration in the loop, all the features from every column will be stored in this variable
    all_features = []

    for data in dat:
        sel = data[['ax', 'ay', 'az', 'am', 'gx', 'gy', 'gz', 'gm', 'mx', 'my', 'mz', 'mm']].values.tolist()
        temp = window_data(sel)
        all_features.append(temp)


    return all_features







##### INSTRUCTIONS #####



daym = butter_mag(generate_datasets(get_pathnames(PATH)))
daym[9].to_csv('kaka.csv')

t=extract_features(daym)
print(t)

train, test = train_test_split(t, test_size=0.3)



# print("training model...")
#
# model = svm.SVC(kernel= "rbf", C=100,gamma='scale')
# model.fit(train, test)
#
# prediction = model.predict(test)
#
# print("Prediction: ")
# print(prediction)

# true_positive, false_positive = get_performance_of_prediction(prediction, [0, 0, 0, 0, 1, 1, 1, 1, 1])
# print("True positives: " + str(true_positive) + " over 3")
# print("False positives: " + str (false_positive) )

