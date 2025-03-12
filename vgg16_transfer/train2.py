"""
vgg16_transfer/train.py

This script performs transfer learning using the pre-trained VGG16 model.
It includes additional layers, batch normalization, dropout regularization,
data augmentation, early stopping, and model checkpointing.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


def define_model(input_shape=(224, 224, 3)):
    """
    Define a transfer learning model using VGG16 as the base.
    
    Args:
        input_shape (tuple): Shape of the input images.
        
    Returns:
        model: A compiled Keras model.
    """
    # Load pre-trained VGG16 model without top layers
    base_model = VGG16(include_top=False, input_shape=input_shape)
    
    # Freeze the layers of VGG16
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom classifier layers on top of the base model
    x = Flatten()(base_model.output)
    x = Dense(128, activation='relu', kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # Output layer for binary classification
    output = Dense(1, activation='sigmoid')(x)
    
    # Define the complete model
    model = Model(inputs=base_model.input, outputs=output)
    
    # Compile the model with SGD optimizer and additional metrics
    opt = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    return model


def run_training():
    """
    Load data, preprocess, and train the VGG16 transfer learning model with data augmentation.
    The best model is saved to disk.
    """
    # Load pre-saved data
    images = np.load('dogs_vs_cats_photos.npy')  # Expected shape: (N, 200, 200, 3)
    labels = np.load('dogs_vs_cats_labels.npy')
    
    # Resize images to (224, 224) for VGG16 compatibility
    images_resized = tf.image.resize(images, [224, 224]).numpy()
    images_resized = images_resized.astype('float32') / 255.0  # Normalize pixel values
    
    # Split dataset into training and validation sets (80/20 split)
    X_train, X_val, y_train, y_val = train_test_split(images_resized, labels, test_size=0.2, random_state=42)
    
    # Define data augmentation for training data
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Define the model
    model = define_model(input_shape=(224, 224, 3))
    
    # Define callbacks: model checkpoint, early stopping, and TensorBoard
    checkpoint = ModelCheckpoint('best_vgg16_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    tensorboard = TensorBoard(log_dir='logs', histogram_freq=1)
    
    # Train the model using data augmentation
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=64),
        validation_data=(X_val, y_val),
        epochs=50,
        callbacks=[checkpoint, early_stopping, tensorboard],
        verbose=2
    )
    
    print("Training complete. Best model saved as 'best_vgg16_model.h5'.")


if __name__ == '__main__':
    run_training()
