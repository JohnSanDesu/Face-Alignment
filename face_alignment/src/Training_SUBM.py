"""
This file is used for training CNN model.
The overview of the functionality of the file is following:
1. Prepare Dataset
2. Initializing Model
3. Train the model
4. Save the model
"""
import os
import torch
import torch.nn as nn
from Network_SUBM import Net
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from Preprocessing_SUBM import FacialKeypointsDataset
from Preprocessing_SUBM import Normalize, ToTensor, Resize
import matplotlib.pyplot as plt
import torch.optim as optim

# 1. Prepare dataset
# create a pipeline for data transformation
data_transform = transforms.Compose([Resize(224),
                                     Normalize(),
                                     ToTensor()
                                     ]
                                    )#
# load and transform the dataset
transformed_dataset = FacialKeypointsDataset(npz_file=r"C:\Users\umaib\Downloads\CV_training_images.npz",
                                             transform=data_transform)

# Add the dataset into loader
batch_size = 10
train_data_loader = DataLoader(transformed_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

# 2. Initialize the network
net = Net(1,20,88)

# Define the loss and optimer
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters())

# function to calculate the euclid distance between the output and the actual keypoints
def euclid_dist(pred_points, actual_points):
    dist = torch.sqrt(torch.sum((pred_points - actual_points) ** 2, dim=2))
    return dist

# 3. Train the model
def train_net(n_epochs):
    """
        Train a CNN model for determined epochs

        Within the epoch, the dataloader forwardpasses the batched train data to the model, and the model makes the predictions.
        The predections are evaluated by the MSE loss function and optimized by the optimizer which applies Adam.

        Args:
            n_epochs (int): The number of epochs

        Returns:
            None
        """
    # prepare the net for training
    net.train()

    avg_loss_list = []
    avg_euclid_list = []

    # loop over epoch
    for epoch in range(n_epochs):

        running_loss = 0.0
        running_euclid = 0.0

        # feeding data in batch over loop
        for batch_i, data in enumerate(train_data_loader):
            # split the data into image and keypoint
            images = data['image']
            key_points = data['keypoints']

            # flatten keypoints
            key_points = key_points.view(key_points.size(0), -1)

            # to evaluate in regression loss, convert the data into Float tensor
            key_points = key_points.type(torch.FloatTensor)
            images = images.type(torch.FloatTensor)

            # forward pass the data to the model to get outputs
            output_points = net(images)

            # calculate the loss between predicted and actual keypoints
            loss = criterion(output_points, key_points)

            # make the parameter gradients zero
            optimizer.zero_grad()

            # backward pass to calculate the weight gradients
            loss.backward()

            # update the weights
            optimizer.step()

            # tracking the performance
            running_loss += loss.item()
            output_points_2ded = output_points.view(output_points.size(0), -1, 2)
            key_points_2ded = key_points.view(key_points.size(0), -1, 2)
            dist = euclid_dist(output_points_2ded, key_points_2ded)
            avg_dist = dist.mean().item()
            running_euclid += avg_dist
            if batch_i % 280 == 279:
                avg_loss_list.append(running_loss / 10)
                avg_euclid_list.append(running_euclid / 10)
            if batch_i % 10 == 9:  # Log every 10 batches
                print(
                    f'Epoch: {epoch + 1}, Batch: {batch_i + 1}, Avg. Loss: {running_loss / 10}, Avg. RMSE: {running_euclid / 10}')
                running_loss = 0.0
                running_euclid = 0.0

        print('Finished Training')

    print('Finished Entire Training')
    # Plotting Loss and the Euclid distance
    xaxis_loss_euc = [i for i in range(n_epochs)]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xaxis_loss_euc, avg_euclid_list, label='Euclid Distance', marker='o')
    ax.plot(xaxis_loss_euc, avg_loss_list, label='Loss', marker='x')

    # formatting the graph
    ax.set_title('Euclid distance and Loss Progress over Epochs')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True)

    #Save the graph
    fig.savefig('euc_loss_progress.png')

    # Show the graph
    plt.show()



# Define the epoch
n_epochs = 35

train_net(n_epochs)

# 4. Save the model
def save_model(network, location):
    """
        Save the state of the CNN

        Args:
            network (torch.nn.Module): The state of the neural network
            location (str): The location of the saved model

        Returns:
            None
    """
    file_path = os.path.join(location, "keypoints_model_16.pth")
    torch.save(network.state_dict(), file_path)
    print("model saved to {}".format(file_path))


save_model(net, r'C:\Users\umaib\PycharmProjects\CV_Submission')



