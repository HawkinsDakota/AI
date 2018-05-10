from parse_data import *
import pickle

with open('img_annotations.pkl', 'rb') as f:
    data = pickle.load(f)

attribute_set = set()

for each in data:
    for obj in each['data']:
        attribute_set.add(obj['attrs'])

# plot bounding box
import numpy as np
from skimage.io import imread
import cv2

img_test = imread(data[0]['img'])
test = data[0]['data']
boxes = []
for each in test:
    xmin, xmax, ymin, ymax = np.inf, 0, np.inf, 0
    for item in each['pts']:
        if item[0] < xmin:
            xmin = item[0]
        if item[0] > xmax:
            xmax = item[0]
        
        if item[1] < ymin:
            ymin = item[1]
        if item[1] > ymax:
            ymax = item[1]
    boxes.append([(xmin, ymin), (xmax, ymax)])

for each in boxes:
    img_test = cv2.rectangle(img_test, each[0], each[1], (255, 0, 0), 2)