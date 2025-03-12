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


## Repository Structure