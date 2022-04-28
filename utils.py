import os
import cv2
import numpy as np
import tensorflow as tf

training_images = os.path.join('data_road', 'training', 'image_2')
training_labels = os.path.join('data_road', 'training', 'gt_image_2')
processed_images = os.path.join('data_road', 'processed_images')

def preprocessdata(label_img):
    '''Transform a label image into an array label'''
    output = np.zeros(label_img.shape[1])
    for i in range(label_img.shape[1]):
        # start from the top because (0,0) is 
        # actually the top left of the image
        for j in range(label_img.shape[0]-1, -1, -1):
            pixel = (255, 0, 255)
            if pixel == tuple(label_img[j,i,:]):
                output[j,i] = 1
            else:
                break
    # need to flip the line because (0,0) is at the top of the image
    return np.abs(output - label_img.shape[0])

def preprocessdataseg(label_img):
    '''Transform a label image into an array label'''
    output = np.zeros((384,1248,2))
    for i in range(label_img.shape[1]):
        for j in range(label_img.shape[0]):
            pixel = (255, 0, 255)
            if pixel == tuple(label_img[j,i,:]):
                output[j,i] = np.array([0,1])
            else:
                output[j,i] = np.array([1,0])

    # need to flip the line because (0,0) is at the top of the image
    return output

def getfilelist():
    '''Return a 2d array of the images and their labels'''
    imagefiles = np.array([x for x in os.listdir(training_images)])
    filelist = np.zeros((imagefiles.shape[0], 2), dtype='object')
    for i, file in enumerate(imagefiles):
        file_ :str = file
        name = file_.split('_')
        labelname = os.path.join(training_labels, f'{name[0]}_road_{name[1]}')
        filename = os.path.join(training_images, file_)
        if not os.path.exists(labelname):
            raise ValueError('label does not exist')

        filelist[i, 0] = filename
        filelist[i, 1] = labelname
    return filelist

def getprocessedfilelist():
    x = set()
    y = set()
    image_files = []
    label_files = [os.path.join(processed_images, x) for x in os.listdir(processed_images) if x.endswith('.npy')]
    def getimagefile(labelfile):
        basename = os.path.basename(labelfile)
        e = basename.split('_')
        suffix = e[-1].split('.')[0]
        imgfile = f'{e[0]}_{e[2]}_{suffix}.jpg'
        return imgfile

    image_files = np.array([os.path.join(processed_images, getimagefile(x)) for x in label_files])
    labels = np.array([np.load(x) for x in label_files])
    return image_files, labels
