#!/usr/bin/env python
"""
Info: This script uses the pretrained ResNet50 model as a feature extractor to predict sign language letters. It takes the pretrained ResNet50 model and adds new classification layers and trains this model on the ASL corpus consisting of images of English letters depicted in the American Sign Language.

Parameters:
    (optional) train_data: str <name-of-training-data>, default = "asl_alphabet_train_subset"
    (optional) test_data: str <name-of-test-data>, default = "asl_alphabet_test_subset"
    (optional) augment_data: str <perform-data-augmentation-true-false>, default = "False"
    (optional) batch_size: int <size-of-batches>, default = 32
    (optional) n_epochs: int <number-of-epochs>, default = 15
    (optional) output_filename: str <name-of-classification-report>, default = "classification_report.txt"

Usage:
    $ python cnn-asl.py
    
Output:
    - model_summary.txt: a summary of the model architecture.
    - model_architecture.png: a visual representation of the model architecture.
    - model_loss_accuracy_history.png: a plot showing the loss and accuracy learning curves of the model during training.
    - classification_report.txt: classification metrics of the model performance.
    - saved_model.json: the model saved as a JSON-file.
    - model_weights.h5: the model weights saved in the HDF5 format. 
"""

### DEPENDENCIES ###

# Core libraries
import os 
import sys
sys.path.append(os.path.join(".."))

# Matplotlib, numpy, OpenCV, pandas, glob, contextlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
import glob
from contextlib import redirect_stdout

# Scikit-learn
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer

# TensorFlow
import tensorflow as tf
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier # a wrapper for sci-kit learn that imports the KerasClassifier
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import (load_img,
                                                  img_to_array,
                                                  ImageDataGenerator)
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     Dropout)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, 
                                     MaxPooling2D, 
                                     GlobalAveragePooling2D,
                                     MaxPool2D,
                                     Activation, 
                                     Flatten, 
                                     Dense)
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from tensorflow.keras.models import model_from_json

# argparse
import argparse


### MAIN FUNCTION ###

def main():
    
    ### ARGPARSE ###
    
    # Initialize ArgumentParser class
    ap = argparse.ArgumentParser()
    
    # Argument 1: Training data
    ap.add_argument("-t", "--train_data",
                    type = str,
                    required = False, # the argument is not required 
                    help = "Name of training data folder",
                    default = "asl_alphabet_train_subset") # default is a subset of the training dataset
    
    # Argument 2: Validation data
    ap.add_argument("-te", "--test_data",
                    type = str,
                    required = False, # the argument is not required 
                    help = "Path to the training data",
                    default = "asl_alphabet_test_subset") # default is a subset of the training dataset
    
    # Argument 3: Augment data
    ap.add_argument("-a", "--augment_data",
                    type = str,
                    required = False, # the argument is not required 
                    help = "Specify whether you want to perform data augmentation: True/False.",
                    default = "False") # data augmentation is not performed by default
    
    # Argument 2: Batch size
    ap.add_argument("-b", "--batch_size",
                    type = int,
                    required = False, # the argument is not required 
                    help = "Define the size of the batches",
                    default = 32) # default batch size 
    
    # Argument 3: Number of epochs
    ap.add_argument("-n", "--n_epochs",
                    type = int,
                    required = False, # the argument is not required 
                    help = "Define the number of epochs",
                    default = 15) # default number of epochs
    
    # Argument 4: Output filename (classification report)
    ap.add_argument("-o", "--output_filename",
                    type = str,
                    required = False, # the argument is not required 
                    help = "Define the name of the output file (the model classification report)",
                    default = "classification_report.txt") # default output filename
    
    # Parse arguments
    args = vars(ap.parse_args())
    
    # Save input parameters
    train_data = os.path.join("..", "data", "subset_asl_sign_language", args["train_data"])
    test_data = os.path.join("..", "data", "subset_asl_sign_language", args["test_data"])
    augment_data = args["augment_data"]
    batch_size = args["batch_size"]
    n_epochs = args["n_epochs"]
    output_filename = args["output_filename"]
    
    # Create output directory if it does not already exist
    if not os.path.exists(os.path.join("..", "output")):
        os.mkdir(os.path.join("..", "output"))
    
    # Start message
    print("\n[INFO] Initializing...")
    
    # Instantiate the CNN_classifier class
    classifier = CNN_classifier(train_data, test_data)
    
    # Create list of label names from the directory names in the training data folder
    labels = classifier.list_labels()
    
    # Create training and test data (X) and labels (y)
    print("\n[INFO] Preparing training and validation data...")
    X_train, y_train = classifier.create_XY(train_data, labels, dimension=224)
    X_test, y_test = classifier.create_XY(test_data, labels, dimension=224)
    
    # Normalize images and binarize labels
    print("\n[INFO] Normalizing images and binarizing labels...")
    X_train_scaled, X_test_scaled, y_train_binarized, y_test_binarized = classifier.normalize_binarize(X_train, y_train, X_test, y_test)
    
    # Define ResNet50 model and add new classification layers
    print("\n[INFO] Loading the pretrained ResNet50 model and adding new classification layers...")
    model = classifier.build_ResNet50()
    
    # Save model summary to output directory
    print("\n[INFO] Saving model summary to 'output' directory...")
    classifier.save_model_summary(model)
    
    # Create new, artificial data with data augmentation if the user has specified so
    datagen = classifier.perform_data_augmentation(X_train_scaled, augment_data)
    
    # Train model
    print("\n[INFO] Training model...")
    model_history = classifier.train_model(model, batch_size, n_epochs, datagen, augment_data, X_train_scaled, X_test_scaled, y_train_binarized, y_test_binarized)
    
    # Plot loss/accuracy during training and saving to output
    print("\n[INFO] Plotting loss/accuracy history of model during training and saving plot to 'output' directory...")
    classifier.plot_training_history(model_history, n_epochs)
    
    # Evaluate model
    print(f"\n[INFO] Evaluating model and saving classification metrics as {output_filename} to 'output' directory...")
    classifier.evaluate_model(y_test_binarized, labels, output_filename, model, batch_size, X_test_scaled)
    
    # Save model as json-file to output directory
    print("\n[INFO] Saving model as JSON-file to 'output' directory...")
    classifier.save_model(model)

    # User message
    print("\n[INFO] Done! Results can be found in the 'output' directory. \n")
    
    
# Creating Neural network classifier class   
class CNN_classifier:
    
    def __init__(self, train_data, test_data):
        
        # Receive inputs: Image and labels 
        self.train_data = train_data
        self.test_data = test_data
   

    def list_labels(self):
        """
        This method defines the label names by listing the names of the folders within training directory without listing hidden files. 
        """
        # Create empty list
        labels = []
    
        # For every name in training directory
        for name in os.listdir(self.train_data):
            # If it does not start with . (which hidden files do)
            if not name.startswith('.'):
                labels.append(name)
                
        # Sort labels alphabetically           
        labels = sorted(labels)
            
        return labels

    
    def create_XY(self, data, labels, dimension=224):
        """
        Method creates trainX, trainY as well as testX and testY. It creates X, which is an array of images (corresponding to trainX and testX) and Y which is a list of the image labels (corresponding to trainY and testY). Hence, with this we can create the training and validation datasets. 
        """
        # Create empty array, X, for the images, and an empty list, y, for the image labels
        X = np.empty((0, dimension, dimension, 3))        
        y = []
    
        # For each artist name listed in label_names
        for name in labels:
            # Get all images for each artist
            images = glob.glob(os.path.join(data, name, "*.jpg"))
        
            # For each image in images 
            for image in images:
                # Load image
                loaded_img = cv2.imread(image)
        
                # Resize image to the specified dimension
                resized_img = cv2.resize(loaded_img, (dimension, dimension), interpolation = cv2.INTER_AREA) # INTER_AREA means that it is resizing using pixel-area relation which was a suggested method by Ross
        
                # Create array of image
                image_array = np.array([np.array(resized_img)])
        
                # Append to trainX array and trainY list
                X = np.vstack((X, image_array))
                y.append(name)
        
        return X, y
    
    
    def normalize_binarize(self, X_train, y_train, X_test, y_test):
        """
        Method that normalizes the training and validation data and binarizes the training and test labels. Normalizing is done by dividing by 255 to compress the pixel intensity values down between 0 and 1 rather than 0 and 255. Binarizing is performed using the LabelBinarizer function from sklearn. We binarize the labels to convert them into one-hot vectors. 
        """
        # Normalize training and test data
        X_train_scaled = (X_train - X_train.min())/(X_train.max() - X_train.min()).astype("float")
        X_test_scaled = (X_test - X_test.min())/(X_test.max() - X_test.min()).astype("float")
        
        # Binarize training and test labels
        label_binarizer = LabelBinarizer() # intialize binarizer
        y_train_binarized = label_binarizer.fit_transform(y_train) # binarizing training image labels
        y_test_binarized = label_binarizer.fit_transform(y_test) # binarizing validation image labels
        
        return X_train_scaled, X_test_scaled, y_train_binarized, y_test_binarized
    
    
    def build_ResNet50(self):
        """
        This method loads the pretrained ResNet50 model and adds new classifier layers 
        """
        # Clear existing session
        tf.keras.backend.clear_session()

        # Load pretrained ResNet50 model without classifier layers
        model = ResNet50(include_top=False, # this means that we are not including the fully-connected layer which is at the top of the network.
                         pooling='avg', # average pooling
                         input_shape=(224, 224, 3))
        
        # Mark loaded layers as not trainable, because we do not want to retrain the network and updating the weights. We just want to use the weights that have already been trained.
        for layer in model.layers:
            layer.trainable = False
            
        # Add new classifier layers to replace the fully-connected layer that we have not included
        # First we take the output of the final layer in the pretrained ResNet50 and use as input for the flattening layer
        flat1 = Flatten()(model.layers[-1].output) # -1 means that we are taking the output of the previous layer
        
        # Add class1 layer with 256 nodes and relu as activation function 
        class1 = Dense(256,
                       activation='relu')(flat1) # the flattening layer is the input layer
        
        drop1 = Dropout(0.4)(class1) # adding dropout layer to reduce overfitting 
        
        output = Dense(26, # we have 26 classes, i.e. 26 letters to predict 
                       activation='softmax')(drop1)
        
        # Define new model
        model = Model(inputs=model.inputs, 
                      outputs=output)
        
        # Compile new model
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        return model

        
    def save_model_summary(self, model):
        """
        This method simply takes the model, creates a summary and saves it to the output directory.
        """
        # Save model summary to output directory
        output_path_summary = os.path.join("..", "output", "model_summary.txt")
        
        with open(output_path_summary, 'w') as f:
            with redirect_stdout(f):
                model.summary()
                    
        # Visualize model architecture and save to output directory
        output_path_model = os.path.join("..", "output", "model_architecture.png")  
        plot = plot_model(model,
                          to_file = output_path_model,
                          show_shapes=True,
                          show_layer_names=True)

        
    def perform_data_augmentation(self, X_train_scaled, augment_data):
        """
        This method uses the TensorFlow DataGenerator to create new, artificial images based on the original data to increase the amount of data and prevent the model form overfitting the trainig data. I have tried to make the augmentation as "realistic" as possible to prevent generating data that the model would never encounter. 
        """
        if augment_data == "True":
            
            # User message
            print("\n[INFO] Performing data augmentation to create new data...")
            
            # Initialize the data augmentation object
            datagen = ImageDataGenerator(zoom_range = 0.15, # zooming
                                         width_shift_range = 0.2, # horizontal shift
                                         height_shift_range = 0.2, # vertical shift
                                         horizontal_flip=True) # mirroring image
            
            # Perform the data augmentation
            datagen.fit(X_train_scaled)
            
            return datagen
    
        
        # If the user has not specified that they want to perform data augmentation
        if augment_data == "False":
            return None

        
    def train_model(self, model, batch_size, n_epochs, datagen, augment_data, X_train_scaled, X_test_scaled, y_train_binarized, y_test_binarized):
        """
        This method trains the ResNet50 model with the new classifier layers on the scaled training images and validates the model on the scaled validation data.
        """
        # If the user has chosen to perform data augmentation
        if augment_data == "True":
            
            # Train model on the original and augmented data 
            model_history = model.fit(datagen.flow(X_train_scaled, y_train_binarized, batch_size = batch_size),
                                      validation_data=(X_test_scaled, y_test_binarized),
                                      epochs=n_epochs,
                                      verbose=1) # show progress bars to allow the user to follow along
            
            return model_history
        
        # If the user has not performed data augmentation
        if augment_data == "False":
            
            # Train model on the original and augmented data 
            model_history = model.fit(X_train_scaled, y_train_binarized, 
                                      validation_data=(X_test_scaled, y_test_binarized),
                                      batch_size = batch_size,
                                      epochs=n_epochs,
                                      verbose=1) # show progress bars to allow the user to follow along
            
            return model_history
        
                                  
    def plot_training_history(self, model_history, n_epochs):
        """
        This method plots the loss/accuracy curves of the model during training and saves the plot to the output directory. The code was developed for use in class and has been modified for this project. 
        """
        # Visualize performance using matplotlib
        plt.style.use("fivethirtyeight")
        plt.figure()
        plt.plot(np.arange(0, n_epochs), model_history.history["loss"], label="train_loss")
        plt.plot(np.arange(0, n_epochs), model_history.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, n_epochs), model_history.history["accuracy"], label="train_acc")
        plt.plot(np.arange(0, n_epochs), model_history.history["val_accuracy"], label="val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join("..", "output", "model_loss_accuracy_history.png"))
        
        
    def evaluate_model(self, y_test_binarized, labels, output_filename, model, batch_size, X_test_scaled):
        """
        This method evaluates the model performance, prints the classification report to the terminal, and saves it as a txt-file to the output directory. 
        """
        # Compute predictions
        y_predictions = model.predict(X_test_scaled, batch_size=batch_size)
        
        # Classification report
        classification_metrics = classification_report(y_test_binarized.argmax(axis=1), 
                                                       y_predictions.argmax(axis=1), 
                                                       target_names=labels)
        
        # Print classification report to the terminal
        print(classification_metrics)
        
        # Save classification report to output directory
        out_path = os.path.join("..", "output", output_filename) # Define output path
        
        # Save classification metrics to output directory
        with open(out_path, "w") as f:
            f.write(f"Below are the classification metrics for the classifier:\n\n{classification_metrics}")
            
            
    def save_model(self, model):
        """
        This method saves the model as a json-file as well as the model weights in the HDF5 format in the output directory. The HDF5 format contains the model weights grouped by layer names. This means that once the model has been trained with the new classification layer, it can be loaded from the saved files, which saves a lot of time. 
        """
        # Convert model to json
        model_json = model.to_json()
        
        # Save model as json-file
        out_model_path = os.path.join("..", "output", "saved_model.json")
        with open(out_model_path, "w") as json_file:
            json_file.write(model_json)
            
        # The weights of the model also need to be saved to the HDF5 format
        out_weights_path = os.path.join("..", "output", "model_weights.h5")
        model.save_weights(out_weights_path)
    
            
# Define behaviour when called from command line
if __name__=="__main__":
    main()