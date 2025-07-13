# 🧠 Image Preprocessing & Classification Project

This project focuses on preprocessing image data and applying machine learning models like **Convolutional Neural Networks (CNN)** and **KMeans clustering** for visual analysis tasks. It includes scripts to standardize image dimensions, apply transformations, and organize data for training and testing.

---

## 📁 Folder Structure


```bash
Data_challenge/
├── x_train/ # Raw training images (.jpg/.png)
├── y_train/ # Corresponding labels (CSV or masks)
├── x_test/ # Raw test images
├── random_submission_example/ # Example submission file
├── supplementary_files/ # Additional scripts or config files
├── dataset_clean.py # Preprocessing script
├── CNN.py # CNN training and inference script 
├── KMeans.py # KMeans clustering script
├── Training_images/ # Preprocessed training images (auto-created)
└── Test_images/ # Preprocessed test images (auto-created)
```


---

## ⚙️ Dependencies

Install the required Python libraries:

```bash
pip install tensorflow scikit-learn opencv-python Pillow numpy pandas glob2
```

🚀 Getting Started
1. Place Your Data
Store all raw training images in x_train/

Store corresponding labels in y_train/

Store raw test images in x_test/

Optionally include an example submission file in random_submission_example/

Add any helper scripts in supplementary_files/

2. Preprocess Images
Run the preprocessing script to resize, crop, and transform the images:
```bash
python dataset_clean.py
```

🧠 Run the Models
CNN (Convolutional Neural Network)
```bash
python CNN.py
```

This script trains a CNN on the processed images and performs classification.

KMeans Clustering
```bash
python KMeans.py
```

This script clusters image features using KMeans for unsupervised analysis.

🔍 Features of Preprocessing
Automatic resizing: Images are resized to match the smallest original width.

Augmentation: Rotations, flips, and crops are applied based on filename patterns.

Format support: Works with .jpg, .jpeg, and .png formats.

