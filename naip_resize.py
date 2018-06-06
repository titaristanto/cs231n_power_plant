# Standalone python script for resizing NAIP imagery files to specified size.
# Usage: put this .py file immediately outside the uspp_naip folder and run
# $ python naip_resize.py

# Author: Zhilin Jiang (zjiang23@)

from PIL import Image
import numpy as np
import os, glob
import cv2 as cv

def parse_filenames(folder_name):
    files = glob.glob(os.path.join("./", folder_name, '*.tif'))
    return files

def resize_single_image(X, size):
    X_new = cv.resize(X, size, interpolation=cv.INTER_LINEAR)
    return X_new

def batch_resize(folder_name, target_size):
    filenames = parse_filenames(folder_name)
    m = len(filenames)

    for i in range(m):
        print (filenames[i])
        img = Image.open(filenames[i])
        # Crop out some 1115-th row/column, and remove the infrared channel
        X = np.array(img)[:1114,:1114,:3]
        new_filename = os.path.join("./", folder_name, '*.tif')
        img_new = Image.fromarray(resize_single_image(X, target_size))
        img_new.save("./" + folder_name + "_resized/" + filenames[i][(len(folder_name)+2):])

def main():
    batch_resize('uspp_naip', target_size=(300,300))

if __name__ == '__main__':
    main()
