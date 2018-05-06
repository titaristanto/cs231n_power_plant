from PIL import Image
import numpy as np
import os, glob
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

def parse_filenames(folder_name):
    main_dir = os.getcwd('C:\\Users\\E460\\Documents\\Stanford\\Courses\\Spring 17\\CS 231N\\Project')
    files = glob.glob(os.path.join(main_dir, folder_name, '*'))
    return files

def get_data(filenames):
    m = len(filenames)
    arr_input_X = np.zeros((m, 76, 76, 3))
    #arr_input_Y = np.zeros((m,1))

    for i in range(m):
        img = Image.open(filenames[i])
        arr_img = np.array(img)
        arr_input_X[i]= arr_img

        #print (arr_input_X.shape)
        #arr_input_Y[i,:] = arr['y']

    return arr_input_X

def main():
    x_train = get_data(parse_filenames(folder_name='uspp_landsat'))


if __name__ == '__main__':
    main()
