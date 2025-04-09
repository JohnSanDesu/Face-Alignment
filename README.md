# Facial Alignment 

## Overview

This project investigates a computer vision task using Convolutional Neural Networks (CNNs):

1. **Face Alignment**: Detecting and aligning facial keypoints.


Developed by **Keisuke Tsujie** on **16/05/2024** as part of a university module assignment.

---

### Objective

Detect and align facial keypoints (e.g., eyes, nose, mouth) from image data using a trained CNN model.

### Methodology

- **Data Preprocessing**
  - Input: NumPy-formatted facial image dataset.
  - Steps: Grayscaling, resizing, normalization.
  - Reference Dataset: [Facial-Landmark-Detection](https://github.com/vinayprabhu/facial-landmark-detection)

- **CNN Architecture**
  - Framework: PyTorch
  - Layers: Two convolutional layers + max pooling → Fully connected layer.
  - Output: 88 coordinates (44 facial landmarks in 2D)

- **Training Details**
  - Optimizer: Adam
  - Loss Function: Mean Squared Error (MSE)
  - Batch Size: 10
  - Activation Function: ReLU

### Results

- **Quantitative Analysis**
  - High accuracy on training data: ~90% within error margin of 0.0005–0.0015.
  - Potential overfitting observed.

- **Qualitative Analysis**
  - Success with front-facing, neutral expression images.
  - Failure with tilted angles, exaggerated facial expressions, or low-quality input.


## Dependencies

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- (Optional) OpenCV

---

## References

- [ReLU in Deep Learning – Kaggle](https://www.kaggle.com/code/dansbecker/rectified-linear-units-relu-in-deep-learning)  
- [Facial Landmark Detection GitHub Repo](https://github.com/vinayprabhu/facial-landmark-detection)  
- [PyTorch Deep Learning by mrdbourke](https://github.com/mrdbourke/pytorch-deep-learning)

---

## Author

**Keisuke Tsujie**  
University of Sussex  
May 2024
