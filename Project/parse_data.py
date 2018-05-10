"""
This module is used to interact and organize the crowdsourced dataset for the
BU CS640 semester project.

Author: Dakota Hawkins
"""

import xml.etree.ElementTree as et
import os
import re
import numpy as np
import pickle
from matplotlib import pylab as plt
from cv2 import drawContours


from torchvision.models import alexnet
from torchvision import transforms
from torch.autograd import Variable
import torch.nn  as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader


def get_img_data(root_dir):
    """
    Get file names and annotation data for each annotated image in the dataset.

    Arguments:
        root_dir (string): top-most directory in dataset directories.

    Returns:
        (list, dict): list of dictionaries containing image file and xml file
            locations along with annotation data for each labelled image.
    """
    img_files = []
    for root, dirs, files in os.walk(root_dir):
        for loc in dirs:
            current_loc = os.path.join(root, loc)
            files = os.listdir(current_loc)
            file_dict = {}
            for f in files:
                if re.search('(?=scene)', f) and f.endswith('.jpg'):
                    file_dict['img'] = os.path.join(current_loc, f)
                if f.endswith('.xml'):
                    file_dict['xml'] = os.path.join(current_loc, f)
                    file_dict['data'] = get_polygon_pts(file_dict['xml'])
            if len(file_dict) > 0:
                img_files.append(file_dict)
    return img_files



def get_polygon_pts(xml_file):
    """
    Get all points for all polygons annotated in a provided XML file.

    Get all points for all polygons annotated in a provided XML file. XML files
    should be formatted accordinging to LabelMe standards.

    Arguments:
        xml_file (string): location of xml file of choice.

    Returns:
        (list, dict): list of dictionaries containing meta-data for each polygon.
    """

    tree = et.parse(xml_file)
    root = tree.getroot()
    objects = []

    for entry in root.findall('object'):
        data = {}
        data['name'] = entry.find('name').text
        data['id'] = entry.find('id').text
        data['attrs'] = entry.find('attributes').text
        pts = []
        polygons = entry.findall('polygon')
        for obj in polygons:
            points = obj.findall('pt')
            for pt in points:
                pts.append([int(pt.find('x').text), int(pt.find('y').text)])

        data['pts'] = pts
        objects.append(data)
    
    return objects

def box_from_points(points):
    """
    Get edge points to form bounding box from points in a contour.
    """
    xmin, xmax, ymin, ymax = np.inf, 0, np.inf, 0
    box_edges = {'xmin': np.inf, 'xmax': 0,
                 'ymin': np.inf, 'ymax': 0}
    for item in points:
        if item[0] < box_edges['xmin']:
            box_edges['xmin'] = item[0]
        if item[0] > box_edges['xmax']:
            box_edges['xmax'] = item[0]
        
        if item[1] < box_edges['ymin']:
            box_edges['ymin'] = item[1]
        if item[1] > ymax:
            box_edges['ymax'] = item[1]
    return box_edges

def get_object_data(anno_data):
    """
    Separate segmented objects from labelled images.
    """
    object_list = []
    for img_file in anno_data:
        for obj in img_file['data']:
            edges = box_from_points(obj['pts'])
            try:
                label = obj['attrs'].split(',')[0]
            except:
                label = obj['attrs']
            obj_dict = {'file': img_file['img'],
                    'edges': edges,
                    'label': label}
            object_list.append(obj_dict)
    return object_list




def img_data_to_contours(objects):
    """
    Convert polygon points for each labelled object to opencv contours.


    Arguments:
        objects (list, dict): list of dictionaries containing parsed information
            from xml files. Output from `get_polygon_pts`.

    Returns:
        (list, np.array): list of numpy arrays containing edge points for
            labelled objects.
    """
    contours = []
    for each in objects:
        contours.append(np.array(each['pts'], int))
    return contours


def plot_annotations(image, anno_data):
    """
    Plot polygon annotations associated with a given image.

    Arguments:
        image (numpy.array): annotated image.
        anno_data (dict): dictionary containing annotation data. Output from
            `get_polygon_pts()`.
    """
    if not isinstance(image, np.ndarray):
        try:
            image = np.array(image)
        except:
            return None

    contours = img_data_to_contours(anno_data)
    anno_img = cv2.drawContours(image, contours, -1, (255, 0, 0), 2)
    plt.axis('off')
    plt.imshow(anno_img)
    plt.pause(0.001)

if __name__ == "__main__":
    data = get_img_data('dataset')
    with open('img_annotations.pkl', 'wb') as f:
        pickle.dump(data, f)
