from PIL import Image
import numpy as np
import os, glob, itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.applications.resnet50 import ResNet50
import csv
import cv2 as cv

def parse_filenames(folder_name):
    files = glob.glob(os.path.join("./", folder_name, '*.tif'))
    return files

def get_data(filenames):

    # Read from CSV file, find abbreviated label (alabel) to full label (flabel) mapping
    csv_rows = []
    with open("PLANTS.csv", 'rbU') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', dialect=csv.excel_tab)
        for row in csvreader:
            csv_rows.append(row)
   
    for col_idx, col_title in enumerate(csv_rows[1]):
        if "PLPRMFL" in col_title:
            abbr_idx = col_idx
        if "PLFUELCT" in col_title:
            full_idx = col_idx
        if "Imagery Status" in col_title:
            imagery_idx = col_idx

    dict_alabel_flabel = {}
    dict_flabel_id = {}
    dict_alabel_id = {}
    dict_id_flabel = {}
    flabel_id_count = 0

    for row in csv_rows:
        if "YES" in row[imagery_idx]:
            if row[full_idx] not in dict_flabel_id:
                dict_flabel_id[row[full_idx]] = flabel_id_count
                dict_id_flabel[flabel_id_count] = row[full_idx]
                flabel_id_count += 1
            if row[abbr_idx] not in dict_alabel_flabel:
                dict_alabel_flabel[row[abbr_idx]] = row[full_idx]
                dict_alabel_id[row[abbr_idx]] = dict_flabel_id[row[full_idx]]
    print (dict_alabel_flabel)
    print (dict_flabel_id)

    m = len(filenames)
    X = np.zeros((m, 75, 75, 3))
    Y = np.zeros((m))
    dict_id_count = {}
    dict_flabel_count = {}

    for i in range(m):
        img = Image.open(filenames[i])
        X[i] = np.array(img)[:75,:75,:3]

        alabel = filenames[i].split('_')[-1][:-4]
        if (alabel not in dict_alabel_flabel):
            print ("WARNING: unseen a-label found in filename.")
        Y[i] = int(dict_alabel_id[alabel])
        if (Y[i] not in dict_id_count):
            dict_id_count[Y[i]] = 0
        dict_id_count[Y[i]] += 1
        dict_flabel_count[dict_id_flabel[Y[i]]] = dict_id_count[Y[i]]
        
    print (X.shape)
    print (dict_id_count)
    print (dict_flabel_count)
    print (sorted(((v,k) for k,v in dict_flabel_count.iteritems()), reverse=True))
    print (np.histogram(Y))
    
    return X, Y

def resize_image(X, size):
    X_new = np.zeros((X.shape[0], *size,3))
    for i in range(X.shape[0]):
        X_new[i, :, :, :] = cv.resize(X[i, :, :, :], size, interpolation=cv.INTER_LINEAR)
    return X_new

def run_model(x_train, y_train, x_dev, y_dev, epochs=50, batch_size=200):
    model = Sequential()
    model.add(Dense(activation='relu', input_dim=x_train.shape[1], output_dim=64))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=12, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    history = model.fit(x_train, y_train,
                        epochs=epochs, batch_size=batch_size,
                        validation_data=(x_dev, y_dev))
    return model, history

def plot_history(history):
    '''
    Plots train and val loss and accuracy given history. From: history = model.fit(...)
    '''
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]

    if len(loss_list) == 0:
        print('Loss is missing in history')
        return

    ## As loss always exists
    epochs = range(1,len(history.history[loss_list[0]]) + 1)

    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))

    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    ## Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

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
    X_raw, Y_raw = get_data(parse_filenames(folder_name='uspp_landsat'))

    # Resize images
    X = resize_image(X_raw, size=(200, 200))
    m, h, w, c = X.shape

    # Convert labels into one-hot-encoding format
    encoder = LabelBinarizer()
    Y = encoder.fit_transform(Y_raw)

    # Compute the output of last layer of training set using pretrained classifier (clf:Resnet50 - weights:imagenet)
    resnet50_model = ResNet50(include_top=False, weights='imagenet', input_shape=(h, w, c))
    X_features = resnet50_model.predict(X)
    m_features, h_features, w_features, c_features = X_features.shape
    X_features_reshaped = np.reshape(X_features,
                                           (m_features, h_features*w_features*c_features))

    # Split into train, dev, and test set
    x_traindev, x_test, y_traindev, y_test = train_test_split(X_features_reshaped, Y,
                                                              test_size=0.1,
                                                              shuffle=True)
    x_train, x_dev, y_train, y_dev = train_test_split(x_traindev, y_traindev,
                                                              test_size=0.1,
                                                              shuffle=True)



    # Run model
    model, history = run_model(x_train, y_train, x_dev, y_dev, epochs=50, batch_size=200)

    # Make prediction on test set
    y_test_pred = model.predict(x_test)

    # Plot train and val accuracy & loss
    plot_history(history)
    
    # Show confusion Matrix
    #cnf_matrix = confusion_matrix(y_test, y_pred)
    #class_names = list([,]) # list of all the labels
    #plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')

if __name__ == '__main__':
    main()
