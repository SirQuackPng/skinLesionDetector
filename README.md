# Skin Lesion Detector

This project uses a Convolutional Neural Network (CNN), built with my custom neural network library, [QuackNet](https://github.com/SirQuackPng/QuackNet), to classify skin lesions based on the HAM10000 dataset. The model focuses on the 3 largest classes due to significant class imbalance in the HAM10000 dataset.

## Project Overview

### Key Features:
-   **Custom Architecture:** Built and trained using QuackNet with convolutional and pooling layers.
-   **Dataset:** Utilised the HAM10000 dataset, preprocessed to a size of 64x64 
-   **Training Metrics:** Achieved 60.2% accuracy and a loss of 0.91

### Model Architecture:
| Layer Type             | Parameters                                                            |
|------------------------|-----------------------------------------------------------------------|
| Convolutional Layer    | number of kernel: 64, size of kernels: 3x3, stride: 1, padding: None  |
| Activation Layer       | activation function: Leaky Relu                                       |
| Max Pooling Layer      | size of grid: 2x2, stride: 2                                          |
| Convolutional Layer    | number of kernel: 128, size of kernels: 3x3, stride: 1, padding: None |
| Activation Layer       | activation function: Leaky Relu                                       |
| Max Pooling Layer      | size of grid: 2x2, stride: 2                                          |
| Global Average Pooling | -                                                                     |
| Fully Connected Layer  | size of layer: 128                                                    |
| Fully Connected Layer  | size of layer: 3                                                      |

### Training Details:
-   **Optimiser:** Adams optimiser with a learning rate of 0.00025
-   **Loss Function:** Cross-Entropy
-   **Batch Size:** 64
-   **Epochs:** 10