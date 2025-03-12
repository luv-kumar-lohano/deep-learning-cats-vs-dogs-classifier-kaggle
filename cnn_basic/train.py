"""
cnn_basic/train.py

This script defines and trains a basic CNN model for binary classification
using dropout regularization, weight decay (L2 regularization), data augmentation,
early stopping, and a learning rate scheduler. It uses Keras with TensorFlow backend.
"""

import sys
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def define_model(input_shape=(200, 200, 3)):
    """
    Define and compile a CNN model with dropout and L2 regularization.
    
    Args:
        input_shape (tuple): Shape of the input images.
        
    Returns:
        model: A compiled Keras model.
    """
    model = Sequential()
    # First convolutional block
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform',
                     padding='same', kernel_regularizer=l2(0.001), input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    
    # Second convolutional block
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform',
                     padding='same', kernel_regularizer=l2(0.001)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    
    # Third convolutional block
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform',
                     padding='same', kernel_regularizer=l2(0.001)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    
    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(0.001)))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification output
    
    # Compile model
    opt = Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def summarize_diagnostics(history, filename_prefix):
    """
    Plot training and validation loss and accuracy, and save the plot to a file.
    
    Args:
        history: Keras History object from model training.
        filename_prefix (str): Prefix for the saved plot filename.
    """
    plt.figure(figsize=(10, 8))
    
    # Plot loss
    plt.subplot(2, 1, 1)
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='Train Loss')
    plt.plot(history.history['val_loss'], color='orange', label='Validation Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(2, 1, 2)
    plt.title('Classification Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], color='orange', label='Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_plot.png")
    plt.close()


def run_training():
    """
    Load data, perform train-test split, augment training data, and train the CNN model.
    The trained model and diagnostic plot are saved to disk.
    """
    # Load pre-saved numpy arrays
    photos = np.load('dogs_vs_cats_photos.npy')  # Shape: (N, 200, 200, 3)
    labels = np.load('dogs_vs_cats_labels.npy')  # Shape: (N,)
    
    # Note: Images are expected to be in the range [0, 255]. Data augmentation generator will rescale them.
    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        photos, labels, test_size=0.2, random_state=42)
    
    # Data augmentation for training data (rescaling and transformations)
    train_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    
    # Only rescaling for validation data
    test_datagen = ImageDataGenerator(rescale=1.0/255.0)
    
    train_generator = train_datagen.flow(X_train, y_train, batch_size=64)
    validation_generator = test_datagen.flow(X_test, y_test, batch_size=64)
    
    # Define the CNN model
    model = define_model(input_shape=(200, 200, 3))
    
    # Define callbacks: early stopping and learning rate scheduler
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        epochs=20,
        callbacks=[early_stopping, lr_scheduler],
        verbose=2
    )
    
    # Evaluate the model on validation data
    loss, accuracy = model.evaluate(validation_generator, steps=len(validation_generator), verbose=0)
    print(f"Validation Accuracy: {accuracy * 100.0:.2f}%")
    
    # Save diagnostic plots
    summarize_diagnostics(history, filename_prefix='cnn_basic')
    
    # Save the trained model
    model.save('cnn_basic_model.h5')
    print("Model saved as 'cnn_basic_model.h5'.")


if __name__ == '__main__':
    run_training()
