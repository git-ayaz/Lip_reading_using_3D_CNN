# Lip Reading - Cross Audio-Visual Recognition using 3D Convolutional Neural Networks

This project is developed as part of the **Deep Neural Networks course**.  
The aim is to perform **visual speech recognition (lip reading)** by recognizing words from only visual frames (no audio) using a 3D Convolutional Neural Network.

Dataset used: **Best Lip Reading Dataset (Kaggle)** containing 685 word sample folders.

---

## Dataset Format

The dataset contains folders named as:
a_1, a_2, cat_1, cat_2, hello_4, lips_6, you_8, etc.

Each folder contains:
- Frames of the lip movement (e.g. 0.png, 1.png, ..., N.png)
- No audio is used (visual only).

Total classes: 14  
Total samples: 685

---

## Model Architecture

The model is a simple **3D Convolutional Neural Network (3D CNN)**:
- 3x Conv3D + BatchNorm3D + ReLU layers
- MaxPooling along spatial dimensions
- AdaptiveAvgPool3D for pooling
- Fully connected layer for classification

---

## How to Run

1. Install dependencies:
    pip install torch torchvision scikit-learn matplotlib tqdm pillow

2. Place dataset in dataset/ folder.

3. Run the notebook or script:
    python main.py

4. During prediction, the program will ask:
    Enter folder path to predict (example: dataset/you_1):

5. A confusion matrix is generated at end of training.

---------------------------------------------------------

