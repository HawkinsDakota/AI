"""
Script to detect objects in an image using AlexNet output along with an SVM
trained on the top 1000 principle components from HOG analysis.

To use script issue the following command in the terminal:
     > python object_localization.py /path/to/your/image

Author: Dakota Hawkins
Class: CS640
"""


from PIL import Image
import cv2
from skimage.io import imread
import numpy as np
import pickle
import sklearn
from sklearn.externals import joblib
from sklearn import svm
from sklearn.decomposition import PCA

from torchvision.models import alexnet
from torchvision import transforms
from torch.autograd import Variable
import torch.nn  as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader

from matplotlib import pylab as plt
from matplotlib import patches
from matplotlib import colors as mcolors

import sys

svm_model = joblib.load('pca_svm.pkl')
classes = svm_model.classes_
pca = joblib.load('pca.pkl')
alex_nn = alexnet(pretrained=True)
alex_nn.eval()


preprocessFn = transforms.Compose([transforms.Scale(256), 
                                   transforms.CenterCrop(224), 
                                   transforms.ToTensor(), 
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                        std=[0.229, 0.224, 0.225])])

def get_windows(image, n_windows):
    """
    Get window size and window shifts for an image with a given shape.

    Arguments:
        image (numpy.ndarray): image data in [H x W x C] format. 
        n_windows (int): number of evenly sized windows desired.
    Returns:
        (list, dict): list containing dictionaries marking the boundaries for
            each window.
    """
    n_rows, n_cols = image.shape[0:2]
    div = n_windows // 2
    dims = n_rows // div, n_cols // div
    shift = dims[0] // 2, dims[1] // 2
    windows = []
    for top_y in range(0, n_rows - dims[0] + 1, shift[0]):
        for top_x in range(0, n_cols - dims[1] + 1, shift[1]):
            window_dict = {'ymin': top_y, 'ymax': top_y + dims[0],
                           'xmin': top_x, 'xmax': top_x + dims[1]}
            windows.append(window_dict)
    
    return windows

def plot_window(image, window):
    """
    Plot current window of an image.

    Arguments:
        image (numpy.ndarray): image data in [H x W x C] format.
        window (dict, int): dictionary of window dimensions. Output from
            get_windows()
    """
    w_image = cv2.rectangle(image.copy(), (window['xmin'], window['ymin']),
                            (window['xmax'], window['ymax']), (255, 0, 0), 3)
    plt.axis('off')
    plt.imshow(w_image)
    plt.pause(0.001)


def get_hog_vector(image):
    """Get HOG features for provided image."""
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (224, 224))
    window_size = (64, 128)
    block_size = (64, 128)
    block_stride = (8, 16)
    cell_size = (8, 16)
    n_bins = 9
    hog = cv2.HOGDescriptor(_winSize=window_size, _blockSize=block_size,
                            _blockStride=block_stride, _cellSize=cell_size,
                            _nbins=n_bins)
    return hog.compute(img).flatten()

def get_top_windows(detections):
    """
    Find the five most probable and unique objects detected in the image.
    """
    probs = np.array([each['p'] for each in detections])
    tops = np.argsort(-1 * np.array(probs))
    best_windows = []
    objects = []
    i = 0
    while len(best_windows) < 5 and i < len(tops):
        window = detections[tops[i]]
        if window['label'] not in objects:
            objects.append(window['label'])
            best_windows.append(window)
        i += 1
    return best_windows

def plot_detections(image, best_windows):
    """
    Plot detections in provided image. 

    Arguments:
        image (numpy.ndarray): image with detected objects.
        best_windows (list, dict): list containing meta-data for each detectin.
    """
    plt.figure()
    img = image.copy()
    plt.axis('off')
    colors = np.random.randint(10, 255, (5, 3))
    object_patches = []
    for i, each in enumerate(best_windows):
        window = each['window']
        img = cv2.rectangle(img, (window['xmin'], window['ymin']),
                            (window['xmax'], window['ymax']), colors[i].tolist(), 3)
        hex_color = mcolors.to_hex(colors[i]/255)
        object_patches.append(patches.Patch(color=hex_color, label=each['label']))
    plt.legend(handles=object_patches)
    plt.imshow(img)
    plt.show()

def locate_objects(image_file):
    """
    Detects objects in an image using AlexNet output along with an SVM trained
    on the top 1000 principle components from HOG analysis.
    """
    pil_image = Image.open(image_file).convert('RGB')
    np_image = np.array(pil_image)
    windows = get_windows(np_image, 4)
    windows += get_windows(np_image, 8)
    windows += get_windows(np_image, 16)
    detections = []
    for w in windows:
        plot_window(np_image, w)
        pil_sub = pil_image.crop((w['xmin'], w['ymin'],
                                  w['xmax'], w['ymax']))
        np_sub = np_image[w['ymin']:w['ymax'], w['xmin']:w['xmax']]
        nn_out = alex_nn(Variable(preprocessFn(pil_sub).unsqueeze(0))).data.numpy()
        hog = get_hog_vector(np_sub)
        features = np.hstack((nn_out, pca.transform(hog.reshape(1, -1))))
        probs = svm_model.predict_proba(features)[0]
        label = svm_model.predict(features)
        max_idx = int(np.where(classes==label)[0])
        detections.append({'window': w, 'p': probs[max_idx], 'label': classes[max_idx]})
        plt.cla()
    best_detections = get_top_windows(detections)
    plot_detections(np_image, best_detections)

        

if __name__ == "__main__":
    args = sys.argv
    if len(args) != 2:
        print("Expected single file path input.")
        sys.exit(1)
    locate_objects(args[1])




