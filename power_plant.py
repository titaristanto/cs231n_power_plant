from PIL import Image
import numpy as np
import os, glob, itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

def parse_filenames(folder_name):
    main_dir = os.chdir('C:\\Users\\E460\\Documents\\Stanford\\Courses\\Spring 17\\CS 231N\\Project')
    files = glob.glob(os.path.join(main_dir, folder_name, '*.tif'))
    return files

def get_data(filenames):
    m = len(filenames)
    arr_input_X = np.zeros((m, 75, 75, 3))
    #arr_input_Y = np.zeros((m,1))

    for i in range(m):
        img = Image.open(filenames[i])
        arr_img = np.array(img)
        if arr_img.shape[0] > 75:
            arr_img = arr_img[:-1,:,:]
        if arr_img.shape[1] > 75:
            arr_img = arr_img[:,:-1,:]

        arr_input_X[i] = arr_img

        #print (arr_input_X.shape)
        #arr_input_Y[i,:] = arr['y']

    return arr_input_X

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
    x_train = get_data(parse_filenames(folder_name='uspp_landsat'))
    
    
    
    
    
    # Show confusion Matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    class_names = list([,]) # list of all the labels
    plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')

if __name__ == '__main__':
    main()
