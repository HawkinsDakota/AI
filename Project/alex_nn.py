"""
This script generates an AlexNet and Histogram of Oriented Gradients dataset
for annotated images.

Author: Dakota Hawkins
CS640 Project
"""

# Dakota imports
from parse_data import *

from PIL import Image

import numpy as np
import cv2

from skimage.io import imread
from skimage import transform
import pickle

from torchvision.models import alexnet
from torchvision import transforms
from torch.autograd import Variable
import torch.nn  as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader

from matplotlib import pyplot as plt

import json



class IkeaDataset(Dataset):
    """Annotated Ikea Dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Arguments:
            root_dir (string): Path to top-level directory containing annotated
                image data.
        """
        self.image_data = get_img_data(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        image = Image.open(self.image_data[idx]['img']).convert('RGB')
        annotations = self.image_data[idx]['data']
        sample = {'image': image, 'annotations': annotations}

        if self.transform:
            sample['image'] = self.transform(sample['image'])
            # sample = self.transform(sample)
        return sample

def hog_from_file(file_name):
    """Retrieve hog features from an image."""
    img = imread(file_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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

def hog_from_tensor(tensor_img):
    """Retrieve HOG measurements from tensor image."""
    arr = tensor_img_to_numpy(tensor_img)
    grayscale = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    return get_hog(grayscale)

def tensor_img_to_numpy(tensor_img):
    """Convert a tensor image to a numpy image."""
    arr = tensor_img.numpy().transpose((1, 2, 0))
    arr = transfrom.Scale()
    return arr
    

class IkeaSubImageDataset(Dataset):
    """Annotated Ikea Dataset with annotated objects separated."""

    def __init__(self, object_dict, transform=None):
        """
        Arguments:
            object_dict (list, dict): list of dictionaries containing
                metadata for annotated Ikea images.
            transform (function): functions to be applied to each object image.
        """
        self.image_data = object_dict
        self.transform = transform

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        image = Image.open(self.image_data[idx]['file']).convert('RGB')
        edges = self.image_data[idx]['edges']
        image = image.crop((edges['xmin'], edges['ymin'],
                            edges['xmax'], edges['ymax']))
        label = self.image_data[idx]['label']
        sample = {'image': image, 'label': label,
                  'file': self.image_data[idx]['file']}

        if self.transform:
            sample['image'] = self.transform(sample['image'])
            # sample = self.transform(sample)
        return sample


preprocessFn = transforms.Compose([transforms.Scale(256), 
                                   transforms.CenterCrop(224), 
                                   transforms.ToTensor(), 
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                        std=[0.229, 0.224, 0.225])])

if __name__ == "__main__":
    model = alexnet(pretrained=True)

    with open('img_annotations.pkl', 'rb') as f:
        data = pickle.load(f)


    class_dict = {}
    with open('imagenet_class_index.json', 'r') as f:
        for (i, x) in json.load(f).items():
            class_dict[int(i)] = x[-1]
    object_data = get_object_data(data)
    obj_dataset = IkeaSubImageDataset(object_data, transform=preprocessFn)
    ikea_labels = list(set([each['label'] for each in object_data]))
    label_to_idx = {each:i for i, each in enumerate(ikea_labels)}



    model.eval()
    # 1000 NN out, 84672 HOG out, 1 label out
    alex_columns = ['alex{}'.format(i) for i in range(1000)]
    hog_columns = ['hog{}'.format(i) for i in range(84672)]
    columns = ",".join(np.array(alex_columns + hog_columns + ['label']))
    f_handle = open("sub_img_data.csv", 'w')
    f_handle.write(columns + '\n')
    for i in range(len(obj_dataset)):
        print(i)
        try:
            each = obj_dataset[i]
            input_var = Variable(each['image'].unsqueeze(0))
            nn_out = model(input_var).data.numpy()[0]
            hog = hog_from_file(each['file'])
            combined = np.hstack((nn_out, hog, label_to_idx[each['label']]))
            line = ",".join([str(each) for each in combined]) + '\n'
            f_handle.write(line)
        except:
            pass
    f_handle.close()

                                                    

