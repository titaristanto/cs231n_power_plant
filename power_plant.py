from PIL import Image
import numpy as np
import os, glob
import matplotlib.pyplot as plt

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

def main():
    x_train = get_data(parse_filenames(folder_name='uspp_landsat'))


if __name__ == '__main__':
    main()
