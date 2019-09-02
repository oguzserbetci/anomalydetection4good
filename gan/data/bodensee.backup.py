# CIFAR10 Downloader

import logging
import pickle
import math
import os
import errno
import tarfile
import shutil
import numpy as np
import urllib3
from sklearn.model_selection import train_test_split
from utils.adapt_data import adapt_labels_outlier_task

logger = logging.getLogger(__name__)

def get_train(label=-1, centered=True, normalize=True):
    return _get_adapted_dataset("train", label, centered, normalize)

def get_test(label=-1, centered=True, normalize=True):
    return _get_adapted_dataset("test", label, centered, normalize)

def get_valid(label=-1, centered=True, normalize=True):
    return _get_adapted_dataset("valid", label, centered, normalize)
    
def get_shape_input():
    return (None, 32, 32, 3)

def get_shape_input_flatten():
    return (None, 32*32*3)

def get_shape_label():
    return (None,)

def num_classes():
    return 2

def get_anomalous_proportion():
    return 0.9

def _unpickle_file(filename):
    logger.debug("Loading pickle file: {}".format(filename))

    with open(filename, mode='rb') as file:
        data = pickle.load(file, encoding='bytes')

    # Reorder the data
    img = data[b'data']
    img = img.reshape([-1, 3, 32, 32])
    img = img.transpose([0, 2, 3, 1])
    # Load labels
    lbl = np.array(data[b'labels'])

    return img, lbl

def _get_dataset(split, centered=False, normalize=False):
    '''
    Gets the adapted dataset for the experiments
    Args : 
            split (str): train or test
            normalize (bool): (Default=True) normalize data
            centered (bool): (Default=False) data centered to [-1, 1]
    Returns : 
            (tuple): <training, testing> images and labels
    '''
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from glob import glob
    from PIL import Image
    
    PATH = '/notebooks/userdata/teamE/MrBank/Pipistrel/'
    
    if split == 'train':
        path = PATH + "Train/nature/*.png"
    elif split == 'test':
        path = "/notebooks/data/datasets/pipistrel/Hackathon/SingleFrame_ObjectProposalClassification/test/*/*.png"
    elif split == 'valid':
        path = PATH + "Validation/Nature/*.png"

    from PIL import ImageOps
    data_list = []
    label_list = []
    filename_list = []
    for img_i, img_path in enumerate(glob(path), 1):
        img = Image.open(img_path)
        img = img.resize((32,32))
        data_list.append(np.array(img))
        label_list.append(int("boat" in img_path.lower()))
        filename_list.append(img_path.split('/')[-1])
        if split in ['train', 'valid']:
            img = ImageOps.mirror(img)
            data_list.append(np.array(img))
            label_list.append(int("boat" in img_path.lower()))
            filename_list.append(img_path.split('/')[-1])

    if split == 'train':
        for patch_i, patch_path in enumerate(glob("/notebooks/userdata/teamE/NATURE_PATCHES/*"), 1):
            img = Image.open(patch_path)
            img = img.resize((32,32))
            data_list.append(np.array(img))
            label_list.append(0)
            filename_list.append(patch_path.split('/')[-1])
            if patch_i > 3000: break
            
    if split == 'train':
        for patch_i, patch_path in enumerate(glob("/notebooks/userdata/teamE/OCEAN_PATCHES_TRAIN/*"), 1):
            img = Image.open(patch_path)
            img = img.resize((32,32))
            data_list.append(np.array(img))
            label_list.append(0)
            filename_list.append(patch_path.split('/')[-1])
        
    if split == 'test':
        for patch_i, patch_path in enumerate(glob("/notebooks/userdata/teamE/OCEAN_PATCHES_TEST/*"), 1):
            img = Image.open(patch_path)
            img = img.resize((32,32))
            data_list.append(np.array(img))
            label_list.append(0)
            filename_list.append(patch_path.split('/')[-1])
    
    data = np.array(data_list)
    labels = np.array(label_list)

    if split in ['train', 'valid']:
        shuffle_ind = np.random.permutation(len(data))
        data = data[shuffle_ind]
        labels = labels[shuffle_ind]

    # Convert images to [0..1] range
    if normalize:
        data = data.astype(np.float32)/255.0
    if centered:
        data = data.astype(np.float32)*2. - 1.
        
    print(split, data.shape, labels, sum(labels))
    return data.astype(np.float32), labels, filename_list

def _get_adapted_dataset(split, label=None, centered=False, normalize=False):
    """
    Gets the adapted dataset for the experiments
    Args : 
            split (str): train or test
            mode (str): inlier or outlier
            label (int): int in range 0 to 10, is the class/digit
                         which is considered inlier or outlier
            rho (float): proportion of anomalous classes INLIER
                         MODE ONLY
            centered (bool): (Default=False) data centered to [-1, 1]
    Returns : 
            (tuple): <training, testing> images and labels
    """
    dataset = {}
    dataset['x_train'], dataset['y_train'], dataset['i_train'] = _get_dataset('train', centered=centered, normalize=normalize)
    dataset['x_test'], dataset['y_test'], dataset['i_test'] = _get_dataset('test', centered=centered, normalize=normalize)
    dataset['x_valid'], dataset['y_valid'], dataset['i_valid'] = _get_dataset('valid', centered=centered, normalize=normalize)

    key_img = 'x_' + split
    key_lbl = 'y_' + split
    key_ind = 'i_' + split

    if label != -1:

        if split in ['train', 'valid']:
            dataset[key_lbl] = adapt_labels_outlier_task(dataset[key_lbl],
                                                         label)
            return (dataset[key_img], dataset[key_lbl], dataset[key_ind])
        else:
            dataset[key_lbl] = adapt_labels_outlier_task(dataset[key_lbl],
                                                         label)

            return (dataset[key_img], dataset[key_lbl], dataset[key_ind])
