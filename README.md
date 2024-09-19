# Fruits Fresh vs. Rotten Classification 🍏🍎

This repository contains a machine learning project focused on classifying fruits as either **fresh** or **rotten** using a Convolutional Neural Network (CNN). The goal of the project is to build an accurate model that can predict the state of various fruits based on image inputs.

## Project Overview

In this project, a deep learning model is trained to classify images of fruits as either fresh or rotten. The project uses a dataset of fruit images and applies image preprocessing, model training, evaluation, and inference.

The project leverages the power of CNNs to extract features from the images and make accurate predictions. 

### Key Features:
- **Image Classification**: Distinguish between fresh and rotten fruits using a CNN-based approach.
- **Data Augmentation**: Techniques like rotation, flipping, and zooming are applied to enhance the dataset.
- **Transfer Learning**: A pre-trained model is used to improve performance and reduce training time.
- **Evaluation Metrics**: Accuracy, precision, recall, and confusion matrix are used to assess the model's performance.

## Installation

### Requirements

Make sure you have the following installed:

- Python 3.x
- TensorFlow / Keras
- NumPy
- Matplotlib
- OpenCV (optional for image processing)
- Jupyter Notebook (for running the notebook)
- Scikit-learn (for model evaluation)

### Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/fruits-fresh-vs-rotten-classification.git
    ```
2. Navigate to the project directory:
    ```bash
    cd fruits-fresh-vs-rotten-classification
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset

The dataset used in this project contains images of fresh and rotten fruits.

The images are split into training, validation, and test sets:
- **Training set**: Used to train the model.
- **Validation set**: Used to tune the model's hyperparameters and avoid overfitting.
- **Test set**: Used to evaluate the model's final performance.

If you'd like to use a different dataset, ensure the directory structure remains consistent with the expected format in the notebook.

## Model Architecture

The model is built using a Convolutional Neural Network (CNN). Here’s an overview of the architecture:
- Input Layer: Preprocessed fruit images.
- Convolutional Layers: Feature extraction layers with filters and ReLU activation.
- Pooling Layers: Downsample the feature maps.
- Fully Connected Layer: Dense layers to interpret the features and classify the image.
- Output Layer: A softmax activation function that outputs the probability of the image being fresh or rotten.

### Data Augmentation

Data augmentation techniques are applied to prevent overfitting and enhance the model's robustness:
- Rotation
- Horizontal/Vertical flipping
- Zooming
- Shifting

### Transfer Learning

To leverage existing knowledge, a pre-trained model such as **VGG16** or **ResNet50** is used as the base model, with additional custom layers added for the classification task.

## How to Run the Project

1. Open the notebook file `Fruits_Fresh_VS_Rotten_Classification.ipynb` using Jupyter Notebook:
    ```bash
    jupyter notebook Fruits_Fresh_VS_Rotten_Classification.ipynb
    ```
2. Execute the cells step-by-step to preprocess the data, build the model, and train it.

3. Evaluate the model's performance on the test dataset.

4. You can also use the trained model to make predictions on your custom fruit images by using the provided inference code.

## Results

- **Accuracy**: The model achieves [insert accuracy]% accuracy on the test set.
- **Confusion Matrix**: A confusion matrix is plotted to visualize the performance across different classes.
- **Precision & Recall**: Precision and recall metrics are calculated to measure the model’s robustness.

## Future Work

- Improve model accuracy by fine-tuning hyperparameters.
- Explore more advanced data augmentation techniques.
- Experiment with other pre-trained models for transfer learning.
- Build a web or mobile application for real-time fruit classification.

## Contributing

If you'd like to contribute to the project, feel free to submit a pull request or open an issue.

