"""
This file is used to get the images for qualitative comparison
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import pickle
from Network_SUBM import Net

# load example image data
example_data = np.load(r"C:\Users\umaib\Downloads\CV_examples.npz")
example_img = example_data['images']

# load CNN network
net = Net(1,20,88)
net.load_state_dict(torch.load(r"C:\Users\umaib\PycharmProjects\CV_Submission\keypoints_model_16.pth"))
net.eval()


def visualize_output(test_image, test_outputs, gt_pts=None):
    """
    Display the image along with the predicted keypoints.

    This function converts a Torch tensor image and the corresponding keypoints back to a numpy format,
    applies denormalization to keypoints, and visualizes the results by plotting keypoints on the image.

    Args:
        test_image (Torch tensor): The image data wrapped in a Torch tensor.
        test_outputs (Torch tensor): The predicted keypoints from the neural network, also wrapped in a tensor.
        gt_pts (numpy array, optional): Ground truth keypoints for comparison (not used in this function).

    Returns:
        tuple: A tuple containing the numpy array of the image and the denormalized keypoints.

    The function performs the following steps:
    - Convert the Torch tensor image back to a numpy array and adjust the channel order for visualization.
    - Denormalize the predicted keypoints (transformed by the model into a normalized space for training).
    - Plot the keypoints on the image.
    - Optionally, plot the ground truth keypoints if provided (currently commented out).
    """

    # Un-transform the image data
    image = test_image[0].data  # Extract the image from its Torch tensor wrapper
    image = np.array(image)     # Convert to a numpy array
    image = np.transpose(image, (1, 2, 0))  # Transpose to convert from Torch to numpy image format

    # Un-transform the predicted key_pts data
    pred_key_pts = test_outputs.data
    pred_key_pts = pred_key_pts.numpy()  # Convert to numpy array
    pred_key_pts = pred_key_pts.reshape(-1, 2)  # Reshape to coordinate pairs
    # Undo normalization of keypoints
    pred_key_pts = pred_key_pts * 50 + 100  # Scale and shift keypoints back to their original scale
    pred_key_pts = np.squeeze(pred_key_pts)  # Remove unnecessary dimensions

    # Display the image in grayscale
    plt.imshow(np.squeeze(image), cmap='gray')
    # Plot the predicted keypoints
    plt.scatter(pred_key_pts[:, 0], pred_key_pts[:, 1], s=20, marker='+', c='r')

    plt.axis('off')  # Hide axes
    plt.show()

    return image, pred_key_pts

test_results = {"images":[],"points":[]}


for img in example_img:
    plt.imshow(img)
    plt.show()
    ## Convert the face region from RGB to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    plt.imshow(img_gray)
    plt.axis('off')
    plt.title("gray")
    plt.show()

    ## Normalize the grayscale image so that its color range falls in [0,1] instead of [0,255]
    img_gray = img_gray / 255
    plt.imshow(img_gray)
    plt.axis('off')
    plt.title("normalized")
    plt.show()


    ## Rescale the detected face to be the expected square size for your CNN (224x224, suggested)
    img_resized = cv2.resize(img_gray, (224, 224))
    plt.imshow(img_resized)
    plt.axis('off')
    plt.title("resized")
    plt.show()

    ## Reshape the numpy image shape (H x W x C) into a torch image shape (C x H x W)
    img_resized = np.expand_dims(img_resized, axis=0)
    img_resized = np.expand_dims(img_resized, axis=0)

    ## Make facial keypoint predictions using your loaded, trained network
    ## perform a forward pass to get the predicted facial keypoints
    img_tensor = torch.from_numpy(img_resized)
    img_tensor = img_tensor.type(torch.FloatTensor)

    ## forward pass to get net output
    output_pts = net(img_tensor)

    ## reshape to batch_size x 68 x 2 pts
    output_pts = output_pts.view(output_pts.size()[0], 88, -1)

    img_save, pts_save = visualize_output(img_resized, output_pts)
    test_results["images"].append(img_save)
    test_results["points"].append(pts_save)

# save the test result as a pickel file
with open('test_result_16.pkl', 'wb') as f:
    pickle.dump(test_results, f)
    print("result data saved")






    # create and save a graph shows change in loss/rmse of the model over epoch




