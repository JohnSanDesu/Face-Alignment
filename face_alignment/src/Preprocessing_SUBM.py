"""
This py file contains the classes used for preprocessing the training data which is fetched to Convolutional Neural Network.
The file includes the class
1. FacialKeypointsDataset: loading the dataset and transforming it
2. Normalize: Normalize and grayscale the data
3. Resize: Resize the image and adjust the keypoints
4. ToTensor: Convert the data into tensor
"""

import os
from torch.utils.data import Dataset
import cv2
import torch
import numpy as np

class FacialKeypointsDataset(Dataset):
    """
    Facial keypoints dataset for loading and transforming facial landmarks data.
    This class takes the dataset and the transformation process. It splits the dataset into images section and keypoints section.
    it then applies the transformation to the dataset.

    Attributes:
        images (numpy.ndarray): The numpy array which represents the image section of the given dataset.
        keypoints (numpy.ndarray): The numpy array which represents the keypoints section of the given dataset
        transform (callable, optional): Transformation which is done to the dataset
    """

    def __init__(self, npz_file, transform=None):
        """
        Initialize the FacialKeypointsDataset with data and optional transformation.

        Args:
            npz_file (str): Path to the numpy zip file which contains the dataset
            transform (callable, optional): A set of transformation process to be applied on dataset.

        Raises:
            FileNotFoundError: In the case which the file is not given to the class.
        """
        if not os.path.exists(npz_file):
            raise FileNotFoundError(f"No file found at {npz_file}")
        with np.load(npz_file) as data:
            self.images = data['images']
            self.keypoints = data['points'] if 'points' in data else None
        self.transform = transform

    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.images)

    def __getitem__(self, idx):
        """
        Getting an image and corresponding keypoints referring to the index. Then apply any transformations

        Args:
            idx (int): The index of the sample

        Returns:
            dict: A dictionary which contains the images and correspoinding keypoints:
                  {'image': image array (transformed), 'keypoints': keypoints array}
.
        """
        image = self.images[idx]

        # unify the channel of the image
        if (image.shape[2] == 4):
            image = image[:, :, 0:3]

        key_pts = self.keypoints[idx].reshape(-1, 2)
        key_pts = key_pts.astype('float').reshape(-1, 2)
        sample = {'image': image, 'keypoints': key_pts}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Normalize(object):
    """
    Normalize and grayscale the image.

    Attributes:
        None
    """

    def __call__(self, sample):
        """
       Applying the normalization and grayscale transformations to the dataset

        Args:
            sample (dict): A dhictionary contains "image" and "keypoints"
                           'image' shall be 3d array (H, W, C)
                           'keypoints' shall be a 2D numpy array (N, 2).

        Returns:
            dict: The processed sample's 'image' and 'keypoints''
        """
        image, key_pts = sample['image'], sample['keypoints']
        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)
        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)
        image_copy = image_copy / 255.0
        key_pts_copy = (key_pts_copy - 100) / 50.0

        return {'image': image_copy, 'keypoints': key_pts_copy}


class Resize(object):
    """
    Resize the image, and adjust the keypoint to the resized image

    Attributes:
        output_size (int or tuple): The desired output size. If an integer, the smallest
                                    dimension of the image will be matched to this size,
                                    maintaining aspect ratio. If tuple, resize to this
                                    exact size (may distort image).
    """

    def __init__(self, output_size):
        """
        Args:
            output_size (int or tuple): indecates the postprocessing size. if it is int, it represents both width and height. Otherwise, it represents (height, width)
        """
        assert isinstance(output_size, (int, tuple)), "output_size must be an int or tuple"
        self.output_size = output_size

    def __call__(self, sample):
        """
        Process of resizing here
        Args:
            sample (dict): A dictionary which contains 'image' and 'keypoints'. The image
                           shall be a 2d or 3d numpy array.
        Returns:
            dict: The resized sample includes resized "image" and adjusted "keypoints"
        """
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            new_size = (self.output_size, self.output_size)
        else:
            new_size = self.output_size

        img = cv2.resize(image, new_size)

        # keypoints scaling
        if h != w:
            key_pts = key_pts * [new_size[1] / w, new_size[0] / h]
        else:
            key_pts = key_pts * (self.output_size / w)

        return {'image': img, 'keypoints': key_pts}

class ToTensor(object):
    """
    Convert nd numpy arrays in a sample to Tensors.
    Dimention adjustment of the numpy array happens in the process of Tensor conversion
    Attributes:
        None
    """

    def __call__(self, sample):
        """
        Run Tensor conversion

        Args:
            sample (dict):  A dictionary containing :
                            'image' (shape: (Height, Width), Color: Grayscale, Colored) and '
                             keypoints' (shape: 2d numpy array (N. 2))
        Returns:
            dict: tensored image and keypoints
                  {'image': tensor image, 'keypoints': tensor keypoints}
        """
        image, key_pts = sample['image'], sample['keypoints']

        if (len(image.shape) == 2):
            image = image.reshape(image.shape[0], image.shape[1], 1)

        image = image.transpose((2, 0, 1))

        return {'image': torch.from_numpy(image),
                'keypoints': torch.from_numpy(key_pts)}


"""
Code tried in the investigation, but not applied in the submission version.
"""

# class EqualizeHistogram(object):
#         image, key_pts = sample['image'], sample['keypoints']
#         equalized_image = cv2.equalizeHist((image * 255).astype(np.uint8))
#         return {'image': equalized_image, 'keypoints': key_pts}


# class GaussianBlur(object):
#     def __init__(self, kernel_size=5):
#         def __init__(self, kernel_size=5):
#         assert kernel_size % 2 == 1, "Kernel size must be odd."
#         self.kernel_size = kernel_size
#     def __call__(self, sample):
#         image, key_pts = sample['image'], sample['keypoints']
#         blurred_image = cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), 0)
#         return {'image': blurred_image, 'keypoints': key_pts}