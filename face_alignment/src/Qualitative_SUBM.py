import os
import torch
import torch.nn as nn
from Network_SUBM import Net
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from Preprocessing_SUBM import FacialKeypointsDataset
from Preprocessing_SUBM import Normalize, ToTensor, Resize
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch.optim as optim


data_transform = transforms.Compose([Resize(224),
                                     Normalize(),
                                     ToTensor()
                                     ]
                                    )#
# load and transform the dataset
transformed_dataset = FacialKeypointsDataset(npz_file=r"C:\Users\umaib\Downloads\CV_training_images.npz",
                                             transform=data_transform)

batch_size = 10
train_loader = DataLoader(transformed_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

train_data = np.load(r'C:\Users\umaib\Downloads\CV_training_images.npz')

net = Net(1,20,88)
net.load_state_dict(torch.load(r"C:\Users\umaib\PycharmProjects\CV_Submission\keypoints_model_16.pth"))
net.eval()

criterion = nn.MSELoss()
all_errors = []

with torch.no_grad():
    for data in train_loader:
        images = data["image"]
        key_pts = data["keypoints"]
        key_pts = key_pts.view(key_pts.size(0), -1)

        key_pts = key_pts.type(torch.FloatTensor)
        images = images.type(torch.FloatTensor)

        output_pts = net(images)

        loss = criterion(output_pts, key_pts)
        all_errors.append(loss.item())

all_errors = np.array(all_errors)
def plot_cdf(errors, save_path=None):
    sorted_errors = np.sort(errors)
    cdf = np.arange(len(sorted_errors)) / float(len(sorted_errors))
    plt.plot(sorted_errors, cdf)
    plt.xlabel('Error')
    plt.ylabel('Cumulative Distribution')
    plt.title('Cumulative Error Distribution')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


output_dir = ""
cdf_save_path = os.path.join(output_dir, 'cumulative_error_distribution.png')
plot_cdf(all_errors, save_path=cdf_save_path)

output_dir = r'C:\Users\umaib\PycharmProjects\CV_Submission\Diagrams\Ver16'
os.makedirs(output_dir, exist_ok=True)
