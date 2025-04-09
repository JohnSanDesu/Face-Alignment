"""
This file is used to test the CNN model with the training dataset
Overview
1. Preprocessing the dataset
2. Fetch to the CNN model
3. Save the results as csv file

"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from Network_SUBM import Net

# load the testing dataset
test_data = np.load(r"C:\Users\umaib\Downloads\CV_test_images.npz",
                    allow_pickle=True)
# extract the image data from test data just in case
testing_images = test_data['images']

# load the CNN Model
net = Net(1, 20, 88)
net.load_state_dict(torch.load(r"C:\Users\showm\Face Alignment\CV_Submission\CV_Submission\CNNModels\keypoints_model.pth"))
net.eval()


def get_output(test_outputs, gt_pts=None):

    # Un-transform the predicted key_pts data
    predicted_key_pts = test_outputs.data
    predicted_key_pts = predicted_key_pts.numpy()
    predicted_key_pts = predicted_key_pts.reshape(-1, 2)

    # Undo normalization of keypoints
    predicted_key_pts = predicted_key_pts * 50 + 100
    predicted_key_pts = np.squeeze(predicted_key_pts)

    return predicted_key_pts


pts_list = []

for i, img in enumerate(testing_images):

    """
    Procedure:
    1. Preprocess the data
    2. Fetch the data to the model, and get the output
    3. list up the keypoints as a preliminary of saving them as csv
    4. Save the image with model output
    """
    # 1. preprocess the data
    # gray scaling
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # normalization
    img_norm = img_gray / 255
    # resizing
    img_resized = cv2.resize(img_norm, (224, 224))
    # reshaping
    img_resized = np.expand_dims(img_resized, axis=0)
    img_resized = np.expand_dims(img_resized, axis=0)
    # convert to tensor
    img_tensor = torch.from_numpy(img_resized)
    img_tensor = img_tensor.type(torch.FloatTensor)


    # 2. Fetch the data to the model, and get the output
    output_pts = net(img_tensor)

    # 33. list up the keypoints as a preliminary of saving them as csv
    output_pts = output_pts.view(output_pts.size()[0], 88, -1)
    pts_save = get_output(output_pts)
    pts_list.append(pts_save)



test_results = {"images":[],"points":[]}
def save_as_csv(points, location=r'C:\Users\umaib\Desktop\CV_SUBM'):

    """
    Save the points out as a .csv file
    :param points: numpy array of shape (no_test_images, no_points, 2) to be saved
    :param location: Directory to save results.csv in. Default to current working directory
    """
    assert points.shape[0] == 554, 'wrong number of image points, should be 554 test images'
    assert np.prod(points.shape[
                   1:]) == 44 * 2, 'wrong number of points provided. There should be 44 points with 2 values (x,y) per point'
    np.savetxt(location + '/results.csv', np.reshape(points, (points.shape[0], -1)), delimiter=',')
    # added by the candidate to make sure that the saving process is done
    print('Saved results to ' + location + '/results.csv')

pts_list = np.array(pts_list)
save_as_csv(pts_list)


