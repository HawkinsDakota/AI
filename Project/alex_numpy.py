class Rescale(object):
    """
    Rescale image in a sample to a given size.

    Arguments:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size while maintaing aspect ratio.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, annos = sample['image'], sample['annotations']

        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > 2:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h

        else: 
            new_h, new_w = self.output_size
        
        new_h, new_w = int(new_h), int(new_w)
        
        img = transform.resize(image, (new_h, new_w))
        for i, anno_dict in enumerate(annos):
            scaled = np.array(anno_dict['pts']) * [new_w / w, new_h / h]
            annos[i]['pts'] = scaled.astype(int).tolist()
        
        return {'image': img, 'annotations': annos}


class Standardize(object):
    """Standardize and scale pixel values in an image."""

    def __init__(self):
        self.NORM_MEAN = np.array([0.485, 0.456, 0.406])
        self.NORM_STD = np.array([0.229, 0.224, 0.225])
    
    def __call__(self, sample):
        image = sample['image']
        img = image/255
        img = (img - self.NORM_MEAN) / self.NORM_STD
        
        return {'image': img, 'annotations': sample['annotations']}


class ToTensor(object):
    """Convert ndarray images to sample Tensors"""

    def __call__(self, sample):
        image, annos = sample['image'], sample['annotations']

        # numpy image H x W x C -> C x H X W
        image = image.transpose((2, 0, 1))
        image_tensor = torch.from_numpy(image).double()
        return {'image': image_tensor, 'annotations': annos}


fig = plt.figure()
for i in range(4):
    sample = ikea_dataset[i]
    print(i, sample['image'].shape)
    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title("Sample #{}".format(i))
    ax.axis('off')
    plot_annotations(sample['image'], sample['annotations'])