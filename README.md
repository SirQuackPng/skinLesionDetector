# Skin Lesion Detector

This project uses a Convolutional Neural Network (CNN), built with my custom neural network library, [QuackNet](https://github.com/SirQuackPng/QuackNet), to classify skin lesions based on the HAM10000 dataset. The model focuses on the three largest classes due to significant class imbalance in the HAM10000 dataset.

## Motivation

This project was developed to explore the capabilities of my neural network library, [QuackNet](https://github.com/SirQuackPng/QuackNet), by applying it to a real-world AI task. I chose to work on skin lesion classification because it's a practical example of machine learning being applied to healthcare, and it provided an opportunity to test my library with convolutional and pooling layers.

Rather than focusing solely on high accuracy, the aim was to understand the challenges involved in training on an imbalanced medical dataset like HAM10000. While the model currently achieves moderate performance, it serves as a strong foundation for future improvements and more advanced applications.

## HAM10000 dataset and Preprocessing

The HAM10000 dataset contains 10000+ dermatoscopic images across 7 classes. However the distribution is highly imbalanced:

| Class | Count  |
|----------------|
| nv    | 6705   |
| mel   | 1113   |
| bkl   | 1099   |
| bcc   | 514    |
| akiec | 327    |
| vasc  | 142    |
| df    | 115    |

To reduce bias and improve generalisation, I only used the top 3 classes - nv, mel and bkl - and balanced the dataset to contain 1099 images per class resulting in a balanced dataset of 3297 samples.

All images were resized to 64 x 64 and normalised. To improve generalisation and increase the size of the training set, simple augmentation was applied by horizontally and vertically flipping the images. This quadrupled the dataset size to 13188 images.

## Model Architecture:

| Layer Type             | Parameters                                                            |
|------------------------|-----------------------------------------------------------------------|
| Convolutional Layer    | number of kernels: 64, size of kernels: 3x3, stride: 1, padding: None  |
| Activation Layer       | activation function: Leaky ReLU                                       |
| Max Pooling Layer      | size of grid: 2x2, stride: 2                                          |
| Convolutional Layer    | number of kernels: 128, size of kernels: 3x3, stride: 1, padding: None |
| Activation Layer       | activation function: Leaky ReLU                                       |
| Max Pooling Layer      | size of grid: 2x2, stride: 2                                          |
| Global Average Pooling | -                                                                     |
| Fully Connected Layer  | size of layer: 128                                                    |
| Fully Connected Layer  | size of layer: 3                                                      |

## Training Details:

-   **Framework:** QuackNet (my own CNN/NN library)
-   **Optimiser:** Adam
-   **Learning Rate:** 0.00025
-   **Loss Function:** Cross-Entropy
-   **Batch Size:** 64
-   **Epochs:** 10

## Results:

After 10 epochs of training on 13188 balanced samples:
-   **Accuracy:** 60.2%
-   **Loss:** 0.91

The performance is moderate, but is strong considering the model was trained from scratch on a balanced subset and built entirely without major libraries.

## Highlights

-   Written entirely without libraries such as PyTorch/TensorFlow
-   Used custom convolution, pooling, and dense layers via QuackNet
-   Handles class imbalance by dataset reduction
-   Saves/loads weights to disk using npz format

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
