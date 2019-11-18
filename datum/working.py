import math
import pandas as pd
import sklearn
from sklearn import svm
from sampen import sampen2, normalize_data
import pickle

ACCELEROMETER_FILE_NAME = 'acce.txt'
GYROSCOPE_FILE_NAME = 'gyro.txt'
MAGNETOMETER_FILE_NAME = 'magnet.txt'

LENGTH_OF_VECTOR = 250


# Verify files are of the correct length
def verify_all_files(prefix):
    if sum(1 for line in open(prefix + ACCELEROMETER_FILE_NAME)) < LENGTH_OF_VECTOR:
        raise RuntimeError(prefix + "Acce - File is too short")

    if sum(1 for line in open(prefix + GYROSCOPE_FILE_NAME)) < LENGTH_OF_VECTOR:
        raise RuntimeError(prefix + "Gyro - File is too short")

    if sum(1 for line in open(prefix + MAGNETOMETER_FILE_NAME)) < LENGTH_OF_VECTOR:
        raise RuntimeError(prefix + "Magnet - File is too short")


# #########################################
# precomputing files to correct data
# assumes files are formatted according to DataCollect2
# #########################################
def collect_data(folder_names):
    dataset = []

    for folder in folder_names:

        dataset.append(parse_data_of_folder(folder))

    return dataset


def parse_data_of_folder(prefix):
    verify_all_files(prefix)
    curr_res = parse_data_into_features(parse_data_into_12d_array(prefix))
    return curr_res


# returns features as 12d array, adding magnitude to the acce, gyro and magnet
def parse_data_into_12d_array(prefix):
    # Open files
    with open(prefix + ACCELEROMETER_FILE_NAME) as a, \
            open(prefix + GYROSCOPE_FILE_NAME) as g, \
            open(prefix + MAGNETOMETER_FILE_NAME) as m:

        # instanciate result and ignores timestamp
        result = []
        a.readline()
        g.readline()
        m.readline()

        # Assuming data range of 250 lines minimum at 50Hz, which results in 5 seconds of data
        for i in range(LENGTH_OF_VECTOR):
            accel_data = a.readline().split()
            ax = float(accel_data[1])
            ay = float(accel_data[2])
            az = float(accel_data[3])
            am = magnitude(ax, ay, az)

            gyro_data = g.readline().split()
            gx = float(gyro_data[1])
            gy = float(gyro_data[2])
            gz = float(gyro_data[3])
            gm = magnitude(gx, gy, gz)

            magnet_data = m.readline().split()
            mx = float(magnet_data[1])
            my = float(magnet_data[2])
            mz = float(magnet_data[3])
            mm = magnitude(mx, my, mz)

            result.append([ax, ay, az, am, gx, gy, gz, gm, mx, my, mz, mm])

    # closes files
    print(len(result))
    return result


# Simple method to calculate the magnitude of x,y,z coordonate
def magnitude(x, y, z):
    return math.sqrt(x * x + y * y + z * z)


# Takes the 12d array and creates the necessary features
# Divides into 5 different overlapping windows: 75 rows at a time
def parse_data_into_features(results):
    len_of_window = int(LENGTH_OF_VECTOR/5 + LENGTH_OF_VECTOR/15)
    len_of_step = int(LENGTH_OF_VECTOR/5)
    i = 0
    all_features = []

    while i+len_of_window <= len(results):
        all_features.extend(get_features_from_window(results[i:i+len_of_window]))
        i += len_of_step
    return all_features


# Creates features from a 2d array
# returns 1d array of size n * 12
# for each n, the different features are added
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
        sampen.append(data[i])


    return sampen2(normalize_data(sampen))



def toast():
    # test get max
    arr = [[1, 2, 3], [1, 2, 3], [1, 5, 6]]
    assert get_max(arr, 1) == (5, 2)

    # test get min
    arr = [[1, 1, 3], [1, 2, 3], [1, 5, 6]]
    assert get_min(arr, 1) == (1, 0)


# ###################################
# statistics for prediction
# ###################################
def get_performance_of_prediction(prediction, actual_data):
    if len(prediction) != len(actual_data):
        raise RuntimeError("Both predictions should have the same size")

    positives = 0
    true_positives = 0
    false_positives = 0

    for i in range(len(prediction)):
        if actual_data[i] == 1:
            positives += 1

            if prediction[i] == 1:
                true_positives += 1

        if actual_data[i] == 0:

            if prediction[i] == 1:
                false_positives += 1

    return true_positives, false_positives


print("getting data...")
training_folder_names = ["jeremie1/", "jeremie3/", "jeremie5/", "jeremie7/", "jeremie9/",
                         "kaan1/", "kaan3/", "kaan5/", "kaan7/", "kaan9/",
                         "ariel1/", "ariel5/", "ariel3/", "ariel11/", "ariel10/",
                         "jeremie0/", "jeremie2/", "jeremie4/", "jeremie6/", "jeremie8/",
                         "kaan0/", "kaan2/", "kaan4/", "kaan8/", "kaan6/",
                         "ariel0/", "ariel2/", "ariel4/", "ariel6/", "ariel8/",  "ariel15/" ,  "ariel16/"]



testing_folder_names = [ "jeremie10/", "jeremie11/",
                        "kaan10/", "kaan11/", "ariel7/",  "ariel9/" ,  "ariel12/" ,  "ariel13/" ,  "ariel14/" ]
train_test_folder_label = [0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0,
                           1, 1, 1, 1, 1,
                           0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0,
                           1, 1, 1, 1, 1, 1, 1]

train_data = collect_data(training_folder_names)
test_data = collect_data(testing_folder_names)


print("training model...")

model = svm.SVC(kernel= "rbf", C=100,gamma='scale')
model.fit(train_data, train_test_folder_label)

prediction = model.predict(test_data)

print("Prediction: ")
print(prediction)

true_positive, false_positive = get_performance_of_prediction(prediction, [0, 0, 0, 0, 1, 1, 1, 1, 1])
print("True positives: " + str(true_positive) + " over 3")
print("False positives: " + str (false_positive) )



x = [2,3,4,5,'NaN','NaN']
print(normalize_data(x))
pickle.dump(model, open("svm.p", "wb"))