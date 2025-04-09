# Facial Alignment and Lip/Eyes Color Modification

## 📘 Overview

This project investigates two computer vision tasks using Convolutional Neural Networks (CNNs):

1. **Face Alignment**: Detecting and aligning facial keypoints.
2. **Lip and Eye Color Modification**: Applying color changes based on aligned keypoints.

Developed by **Keisuke Tsujie** on **16/05/2024** as part of a university module assignment.

---

## 🧠 Task 1: Face Alignment

### 🎯 Objective

Detect and align facial keypoints (e.g., eyes, nose, mouth) from image data using a trained CNN model.

### ⚙️ Methodology

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

### 📈 Results

- **Quantitative Analysis**
  - High accuracy on training data: ~90% within error margin of 0.0005–0.0015.
  - Potential overfitting observed.

- **Qualitative Analysis**
  - Success with front-facing, neutral expression images.
  - Failure with tilted angles, exaggerated facial expressions, or low-quality input.

- **Improvement Suggestions**
  - Use **AdamW** optimizer (includes L2 regularization).
  - Apply **data augmentation** (e.g., rotation, flipping).
  - Update CNN architecture to better handle diverse data.

---

## 🎨 Task 2: Lip and Eye Color Modification

### 🎯 Objective

Modify the color of lips and eyes in facial images by segmenting regions using keypoints.

### ⚙️ Methodology

- Use facial keypoint indexes to create binary masks for lips and eyes.
- Apply color transformations (e.g., red tones) to the masked regions.

### 📈 Results

- **Success Cases**: Accurate red coloring of lips and eyes in aligned images.
- **Failure Cases**: Inaccurate coloring if facial alignment is off.

### 💡 Suggestions

- Train a dedicated CNN to detect lips and eyes more precisely.
- Improve accuracy of facial alignment for better downstream performance.

---

## 📦 Dependencies

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- (Optional) OpenCV

---

## 🔗 References

- [ReLU in Deep Learning – Kaggle](https://www.kaggle.com/code/dansbecker/rectified-linear-units-relu-in-deep-learning)  
- [Facial Landmark Detection GitHub Repo](https://github.com/vinayprabhu/facial-landmark-detection)  
- [PyTorch Deep Learning by mrdbourke](https://github.com/mrdbourke/pytorch-deep-learning)

---

## 👤 Author

**Keisuke Tsujie**  
University of Sussex  
May 2024
