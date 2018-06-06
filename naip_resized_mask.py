# Standalone python script for applying mask on
# Non-building part will be painted black and the building part remaining the original colors.
# Usage: put this .py file immediately outside the uspp_naip_resized folder and run
# $ python naip_resized_mask.py
#
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

def batch_masking(img_folder_name, mask_folder_name):
    filenames = parse_filenames(img_folder_name)

    m = len(filenames)

    for i in range(m):
        print (filenames[i])
        img = Image.open(filenames[i])
        img_X = np.array(img)
        entry_id = filenames[i].split('_')[-3]
        mask_path = "./" + mask_folder_name + "/bilabels_" + entry_id + ".png"
        mask = Image.open(mask_path)
        mask_X = np.array(mask)[:1114,:1114]
        mask_resized_X = resize_single_image(mask_X, size=(300,300))
        mask_resized_X = np.expand_dims(mask_resized_X, axis=2)
        mask_resized_X = np.where(mask_resized_X > 0, 1, 0).astype(np.uint8)
        img_new_X = img_X * mask_resized_X
        img_new = Image.fromarray(img_new_X)
        img_new.save("./" + img_folder_name + "_masked" + filenames[i][(len(img_folder_name)+2):])

def main():
    batch_masking('uspp_naip_resized', 'annotations/binary')

if __name__ == '__main__':
    main()
