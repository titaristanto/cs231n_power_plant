from PIL import Image
import numpy as np
import os, glob, itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

import csv

def parse_filenames(folder_name):
    files = glob.glob(os.path.join("./", folder_name, '*.tif'))
    return files

def get_data(filenames):
    m = len(filenames)
    X = np.zeros((m, 75, 75, 3))
    Y = np.zeros((m))
    dict_label_id = {}
    dict_id_label = {}
    label_int_count = 0
    dict_id_count = {}
    dict_label_count = {}

    for i in range(m):
        img = Image.open(filenames[i])
        X[i] = np.array(img)[:75,:75,:3]

        label = filenames[i].split('_')[-1][:-4]
        if (label not in dict_label_id):
            dict_label_id[label] = label_int_count
            dict_id_label[label_int_count] = label
            label_int_count += 1
        Y[i] = int(dict_label_id[label])
        if (Y[i] not in dict_id_count):
            dict_id_count[Y[i]] = 0
        dict_id_count[Y[i]] += 1
        dict_label_count[dict_id_label[Y[i]]] = dict_id_count[Y[i]]
        
    print (X.shape)
    print dict_label_id
    print dict_id_count
    print dict_label_count
    print sorted(((v,k) for k,v in dict_label_count.iteritems()), reverse=True)
    print np.histogram(Y)
    
    # Convert labels
#     with open("PLANTS.csv", 'rb') as csvfile:
#         csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')

    return X, Y

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    '''
    :param cm: a confusion matrix output from sklearn func, which takes on actual label and prediction
    :param classes: a list of all the labels
    :param normalize: if True, performs row normalization
    :param title: title of the plot
    :param cmap: color choices
    '''
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')

    print(cm)

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('Power Plant Classification Confusion Matrix.png')
    
def main():
    X_train, Y_train = get_data(parse_filenames(folder_name='uspp_landsat'))
    
    # Show confusion Matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    class_names = list([,]) # list of all the labels
    plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')

if __name__ == '__main__':
    main()
