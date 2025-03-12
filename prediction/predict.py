"""
prediction/predict.py

This script loads a trained model and makes a prediction for a new image.
"""

from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import numpy as np


def load_image(filename, target_size=(224, 224)):
    """
    Load and preprocess an image for prediction.
    
    Args:
        filename (str): Path to the image file.
        target_size (tuple): Desired image size.
    
    Returns:
        np.ndarray: Preprocessed image array.
    """
    # Load the image with the target size
    img = load_img(filename, target_size=target_size)
    # Convert the image to a numpy array
    img_array = img_to_array(img)
    # Reshape to add batch dimension
    img_array = img_array.reshape(1, *target_size, 3)
    # Convert pixel values to float32 and center using ImageNet means
    img_array = img_array.astype('float32')
    img_array -= [123.68, 116.779, 103.939]  # Subtract mean pixel values
    return img_array


def run_prediction(model_path, image_path):
    """
    Load a pre-trained model and predict the class of the provided image.
    
    Args:
        model_path (str): Path to the saved Keras model.
        image_path (str): Path to the image file.
    """
    # Load and preprocess the image
    img = load_image(image_path)
    # Load the trained model
    model = load_model(model_path)
    # Make a prediction
    result = model.predict(img)
    print("Prediction result:", result[0])


if __name__ == '__main__':
    # Example usage: update the model and image paths as needed
    MODEL_PATH = 'final_model.h5'
    IMAGE_PATH = 'sample_image.jpg'
    run_prediction(MODEL_PATH, IMAGE_PATH)
