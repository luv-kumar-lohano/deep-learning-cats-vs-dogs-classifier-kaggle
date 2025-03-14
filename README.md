# Deep Learning Model to classify Cats and Dogs:

This repository contains 2 deep learning models for classifying images of dogs and cats:

1. CNN Model (cnn_basic/):  
   - A custom CNN with dropout regularization, L2 weight decay, data augmentation, early stopping, and a learning rate scheduler.
   
2. Transfer Learning with VGG16 (vgg16_transfer/): 
   - A model that leverages the pre-trained VGG16 network with added layers, batch normalization, dropout, and callbacks such as model checkpointing and early stopping.
   

The file also contains:

1. Prediction Script (prediction/):  
   - A script to load a pre-trained model and make predictions on new images.


To access the dataset: https://www.kaggle.com/c/dogs-vs-cats/data



# Additional Notes:

Features added to improve the base model:
1. Dropout Regularization - Randomly sets x% of neurons to 0 (drop out) during training. Helps prevent overfitting.
2. Batch Normalization - Normalizes layer inputs during training to stabilize learning and improve speed.
3. Data Augmentation - Applies transformations like rotation, flipping, and zooming to training data to improve model generalization.
4. Validation Split - Separates part of the data for validation during training to monitor performance and avoid overfitting.
5. Early Stopping - Stops training if validation performance stops improving, saving training time.
6. Checkpoints - Saves the best model based on validation accuracy for future use.
7. TensorBoard - Provides a visual representation of metrics like accuracy and loss during training.
8. Fine-Tuning - Unfreezes some pre-trained model layers for additional training to better adapt to the dataset.
