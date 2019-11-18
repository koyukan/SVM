import math
import pandas as pd
import numpy as np
import os

from sampen import sampen2, normalize_data
from scipy.signal import argrelmin, argrelmax, butter, lfilter, freqz, find_peaks
from sklearn import svm, datasets
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from IPython.display import display, HTML
from sklearn.utils.multiclass import unique_labels
import pickle
ACCELEROMETER_FILE_NAME = '/acce.txt'
GYROSCOPE_FILE_NAME = '/gyro.txt'
MAGNETOMETER_FILE_NAME = '/magnet.txt'
PILOT = 'ariel'
PATH = 'datum'
ROW_SIZE = 252



def get_pathnames (path) :

   folders = []
   for root, dirs, files in os.walk(path):
       folders.append (root)


   return folders[1:]

def generate_datasets (path_list):
   # read the csv files
   dataset = []

   for path in path_list:

       accdata = pd.read_csv(path + ACCELEROMETER_FILE_NAME,sep=" ",header=1, usecols=[1,2,3],names = ['ax','ay','az'], nrows=ROW_SIZE)
       gyrdata = pd.read_csv(path + GYROSCOPE_FILE_NAME, sep=" ",header=1, usecols=[1,2,3], names = ['gx','gy','gz'], nrows=ROW_SIZE)
       magdata = pd.read_csv( path + MAGNETOMETER_FILE_NAME,sep=" ",header=1, usecols=[1,2,3],names=['mx', 'my', 'mz'], nrows=ROW_SIZE)

       frames = [accdata,gyrdata,magdata]
       dataframe = pd.concat(frames, axis=1, sort=False)


       if PILOT in path: dataframe.insert(9,'lab',1)
       else: dataframe.insert(9,'lab',0)
       dataset.append(dataframe)

   return dataset



def smooth_mag (dataset):

    temp = []

    for data in dataset :
        for c in ['ax', 'ay', 'az', 'gx','gy','gz','mx','my','mz']:
            data.loc[:,[c]] = data.loc[:,[c]].rolling(3).mean()


        data = data.dropna()
        data.reset_index(drop=True, inplace=True)

        am =np.sqrt(np.square(data.loc[:, ['ax','ay','az']]).sum(axis=1))
        data.insert(3,'am', am)
        gm = np.sqrt(np.square(data.loc[:, ['gx', 'gy', 'gz']]).sum(axis=1))
        data.insert(7,'gm', gm)
        mm =np.sqrt(np.square(data.loc[:, ['mx','my','mz']]).sum(axis=1))
        data.insert(11,'mm', mm)


        temp.append(data)


    return temp


def get_windows(dt, ws, ol):
    r = np.arange(len(dt))
    s = r[::ol]
    z = list(zip(s, s + ws))
    f = '{0[0]}:{0[1]}'.format
    g = lambda ol: dt.iloc[ol[0]:ol[1]]
    return pd.concat(map(g, z), keys=map(f, z))



def get_features_from_window(window):

    features = []
    cols = ['ax', 'ay', 'az', 'am', 'gx', 'gy', 'gz', 'gm', 'mx', 'my', 'mz', 'mm']

    for c in cols:
        col = window.loc[:,[c]]


        # 1 - Max
        mx = col.max()
        mxindex = col.idxmax()
        features.append(mx)

        # 2 - min of each n
        mn = col.min()
        mnindex = col.idxmin()
        features.append(mn)

        # 3 - mean of each n
        mean = col.mean()
        features.append(mean)

        # 4 - variance of each n
        variance = col.var()
        features.append(variance)

        # 5 - kurtosis of each n
        kurtosis = col.kurt()
        features.append(kurtosis)

        # 6 - skewness of each n
        skew = col.skew()
        features.append(skew)

        # 7 - peak to peak signal
        spp = mx - mn
        features.append(spp)

        # 8 - peak to peak time
        tpp = mxindex + mnindex
        features.append(tpp)

        # 9 - peak to peak slope

        if int(tpp) == 0:
            features.append(spp)
        else:
            spps = spp / tpp
            features.append(spps)

        # 10 - ALAR
        if int(mx) == 0:
            features.append(0)
        else:
            features.append(mxindex/mx)

        # 11 - Energy

        energy = np.einsum('ij,ij->j',col,col)
        features.append(energy[0])

        # 12 - Entropy
        normalized = normalize(col)
        features.append(sampen2(normalized)[1][1])

    return features



def extract_features (dataset):
    features=[]
    #Iterate through dataset
    for data in dataset:
        data_feature = []
        #For each data create windowed view
        windowed = get_windows(data,50,25)
        #Iterate through windows
        for window in windowed.index.get_level_values(level=0).unique():
            #Get features for each window
            feature = get_features_from_window( windowed.loc[window])
            #Apeend the features to features list
            data_feature.extend(feature)

        features.append([data_feature, data['lab'][0]])


    return features





def svc_param_selection (t, nfolds):

    train_data, train_labels = zip(*t)
    Cs = [0.0000001,0.00000001,0.0000001,0.000001,0.00001,0.0001, 0.001,  0.01, 0.1, 1, 10]
    param_grid = [{'C': Cs, 'kernel':['linear']}, {'C': Cs, 'kernel':['rbf']}]
    grid_search = GridSearchCV(svm.SVC(gamma='scale'), param_grid, cv=nfolds, scoring = 'f1')
    grid_search.fit(train_data, train_labels)




    # Plot non-normalized confusion matrix
    plot_confusion_matrix(train_labels, grid_search.predict(train_data),
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plot_confusion_matrix(train_labels, grid_search.predict(train_data), normalize=True,
                          title='Normalized confusion matrix')




    print("Best parameters set found on development set:")
    print()
    print(grid_search.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = grid_search.cv_results_['mean_test_score']
    stds = grid_search.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()



    return grid_search.best_params_




def classification_report_with_accuracy_score(y_true, y_pred):

    print (classification_report(y_true, y_pred)) # print classification report
    return accuracy_score(y_true, y_pred) # return accuracy score



def show_scores(t, outer_cv):
    test_data, test_labels = zip(*t)
    model = svm.SVC(kernel='linear', C=1e-05, gamma='scale')

    # Nested CV with parameter optimization
    nested_score = cross_val_score(model, X=test_data, y=test_labels, cv=outer_cv,
                                   scoring=make_scorer(classification_report_with_accuracy_score))



    print("Accuracy: %0.2f CI: (+/- %0.2f)" % (nested_score.mean(), nested_score.std() * 2))
    print()


    return nested_score, nested_score.mean(), nested_score.std()






def plot_confusion_matrix(y_true, y_pred,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):

    np.set_printoptions(precision=2)
    if y_true[0] == 1:
        classes = [1, 0]
    else:
        classes = [0, 1]

    print ('classes are')
    print (classes)


    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax



# Draw Plot
def plot_df(df, x, y, title="", xlabel='Date', ylabel='Value', dpi=100):

    plt.figure(figsize=(16,5), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()


def quick_plot (t) :
    train, test = train_test_split(t, test_size=0.5)
    test_data, test_labels = zip(*test)
    train_data, train_labels = zip(*test)
    model = svm.SVC(kernel='linear', C=1e-05, gamma='scale')
    model.fit(train_data, train_labels)

    # Plot non-normalized confusion matrix
    plot_confusion_matrix(test_labels, model.predict(test_data),
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plot_confusion_matrix(test_labels, model.predict(test_data), normalize=True,
                          title='Normalized confusion matrix')
    plt.show()


##### INSTRUCTIONS #####

#Generate raw dataset
kaan = generate_datasets(get_pathnames(PATH))

p = kaan[0]['ax']
plot_df(p, x=p.index, y=p, title='Acceleration on X axis, without filtering.')

#Plot raw Acc x
daym = smooth_mag(kaan)
q= daym[0]['ax']
plot_df(p, x=q.index, y=q, title='Acceleration on X axis, after filtering.')
t=extract_features(daym)




# train, test = train_test_split(t, test_size=0.3)
svc_param_selection(t,5)
show_scores(t,5)
quick_plot(t)


# pickle.dump(model, open("svm.p", "wb"))

# dff = pd.DataFrame(t[0][1])
# Html_file= open("filename","w")
# Html_file.write(dff.to_html())
# Html_file.close()

# train, _ = train_test_split(t, test_size=0.5)

#TSNA PCA: Reduce the dimension of the data.

