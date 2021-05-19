#!/usr/bin/env python
"""
Info: This script loads the model trained in the cnn-asl.py script and enables the user to use it for classifying unseen ASL letters. It also visualizes the feature map of the last convolutional layer of the network to enable the user to get an insight into exactly which parts of the original image that the model is paying attention to when classifying the image.

Parameters:
    (optional) model_name: str <name-of-the-model-to-load>, default = "saved_model.json"
    (optional) train_data: str <name-of-training-data>, default = "asl_alphabet_train_subset"
    (optional) unseen_image: str <name-of-unseen-image>, default = "unseen_img_test1.png"

Usage:
    $ python use-model.py
    
Output:
    - unseen_image_superimposed_heatmap.png: superimposed heatmap on unseen image.
    - unseen_image_prediction.txt: model prediction of unseen image.
"""

### DEPENDENCIES ### 

# Core libraries
import os 
import sys
sys.path.append(os.path.join(".."))

# Matplotlib, numpy, OpenCV
import matplotlib.pyplot as plt
import numpy as np
import cv2

# TensorFlow
import tensorflow as tf
from tensorflow.keras.preprocessing.image import (load_img, img_to_array)
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.models import model_from_json
from tensorflow.keras import backend as K

# argparse
import argparse


### MAIN FUNCTION ###

def main():
    
    ### ARGPARSE ###
    
    # Initialize ArgumentParser class
    ap = argparse.ArgumentParser()
    
    # Argument 1: Model name
    ap.add_argument("-m", "--model_name",
                    type = str,
                    required = False, # the argument is not required 
                    help = "Name of the model",
                    default = "saved_model.json") # default name
    
    # Argument 2: Training data
    ap.add_argument("-t", "--train_data",
                    type = str,
                    required = False, # the argument is not required 
                    help = "Name of training data folder",
                    default = "asl_alphabet_train_subset") # default is a subset of the training dataset
    
    # Argument 3: Input image
    ap.add_argument("-u", "--unseen_image",
                    type = str,
                    required = False, # the argument is not required 
                    help = "Name of the image the model should classify",
                    default = "unseen_img_test1.png") # default unseen image provided in the unseen_images folder
    
    # Parse arguments
    args = vars(ap.parse_args())
    
    # Save input parameters
    model_name = args["model_name"]
    train_data = os.path.join("..", "data", "subset_asl_sign_language", args["train_data"])
    unseen_image = args["unseen_image"]

    # Create output directory if it does not already exist
    if not os.path.exists(os.path.join("..", "output")):
        os.mkdir(os.path.join("..", "output"))
    
    # Start message
    print("\n[INFO] Initializing...")
    
    # Instantiate the class
    classifier = Loaded_model_classifier(train_data, unseen_image)
    
    # Create list of label names from the directory names in the training data folder
    labels = classifier.list_labels()
    
    # Load the model
    print(f"\n[INFO] Loading the CNN model, {model_name}, from 'output' directory...")
    model = classifier.load_model(model_name)
    
    # Classify input image
    print(f"\n[INFO] Using the model to predict the class of {unseen_image}...")
    label = classifier.classify_unseen_image(labels, model)
    
    # Visualize feature map of network for input image
    print(f"\n[INFO] Visualizing the feature map of the last convolutional layer of the network...")
    classifier.visualize_feature_map(model)
    
    # User message
    print(f"\n[INFO] Done! The {unseen_image} has been classified as {label} and the feature map of the last convolutional layer of the network has been visualized and saved as {unseen_image}_superimposed_heatmap.png in 'output' directory\n")

    
# Creating classifier class   
class Loaded_model_classifier:
    
    def __init__(self, train_data, unseen_image):
        
        # Receive inputs: train data and input image
        self.train_data = train_data
        self.unseen_image = unseen_image
        
    def list_labels(self):
        """
        This method defines the label names by listing the names of the folders within training directory without listing hidden files. It sorts the names alphabetically. 
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
    
    
    def load_model(self, model_name):
        """
        This method loads the model and the model weights that are saved in the output directory.
        """
        # Load JSON-file and create model
        model_path = os.path.join("..", "output", model_name)
        json_model = open(model_path, "r")
    
        # Read file
        loaded_file = json_model.read()
    
        # Create model
        loaded_model = model_from_json(loaded_file)
    
        # Load weights into new model
        loaded_model.load_weights(os.path.join("..", "output", "model_weights.h5"))
    
        # Compile model
        loaded_model.compile(loss='binary_crossentropy', 
                         optimizer='adam', 
                         metrics=['accuracy'])
    
        return loaded_model
    
    
    def classify_unseen_image(self, labels, model):
        """
        This method takes an unseen image, performs some preprocessing to prepare it for the model, and predicts the class of the image using the model. 
        """ 
        # Define path
        img_path = os.path.join("..", "data", "unseen_images", self.unseen_image)
            
        # Load unseen image
        image = load_img(img_path, target_size=(224, 224)) # using the same size as the images the model has been trained on
            
        # Convert the image to a numpy array
        image = img_to_array(image) 
            
        # Reshape the image, because the model expects a tensor of rank 4. The image goes from being 3-dimensional to 4-dimensional: (1, 224, 224, 3)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2])) 
            
        # Prepare the image for the ResNet50 model
        image = preprocess_input(image)
            
        # Predict the class of the image 
        prediction = np.argmax(model.predict(image))
            
        # Convert labels to be a dictionary which is needed to extract the label that corresponds to the prediction
        labels = dict(zip(labels, range(len(labels))))
            
        # Define function that finds the key (letter) that corresponds to the predicted value
        def find_key(dictionary, value):
            return {k for k, v in dictionary.items() if v == value}
            
        # Extract letter that corresponds to the predicted value from the label dictionary
        label = find_key(labels, prediction)
            
        # Print the predicted class to the terminal
        print(f"\nThe model predicts {self.unseen_image} to be the letter {label}")
            
        # Save prediction as txt-file to output directory
        with open(os.path.join("..", "output", f"{self.unseen_image}_prediction.txt"), "w") as f:
            f.write(f"The predicted class of the {self.unseen_image} made by the model is {label}")
            
        return label
                    
                    
    def visualize_feature_map(self, model):
        """
        This method visualizes the feature map of the last convolutional layer of the network. 
        """
        # Define path
        img_path = os.path.join("..", "data", "unseen_images", self.unseen_image)
            
        # Load image with dimensions corresponding to training images
        img = load_img(img_path, target_size=(224, 224))
            
        # Convert image to array
        x = img_to_array(img)
            
        # Convert to rank 4 tensor
        x = np.expand_dims(x, axis=0)
            
        # Preprocess to be in line with ResNet50 data 
        x = preprocess_input(x)
            
        # Create activation heatmap for final layer. This is done by taking advantage of how the model learns through gradient descent. We use the gradients that have been learned through training, and we go the opposite way (rather than minimizing we are maximizing). Essentially, we make use of the gradients in the final layer to highlight which regions are particularly informative when predicting a given class.     
        with tf.GradientTape() as tape:
            
            # Take the last convolutional layer in the network
            last_conv_layer = model.get_layer('conv5_block3_out')
                
            # Create a model that maps the input image to the activations of the last convolutional layer as well as the output predictions    
            iterate = tf.keras.models.Model([model.inputs], 
                                            [model.output, last_conv_layer.output])
                
            # Compute the gradient of the top predicted class for the input image with respect to the activations of the last conv layer
            # Take the gradients from the last layer
            model_out, last_conv_layer = iterate(x) 
                
            # Find the class that has been predicted by the model
            class_out = model_out[:, np.argmax(model_out[0])] 
                
            # Extract gradient of the output neuron of the last convolutional layer
            grads = tape.gradient(class_out, 
                                      last_conv_layer)
                
            # Vector of mean intensity of the gradient over a specific feature map channel
            pooled_grads = K.mean(grads, axis=(0, 1, 2))
            
            # Multiply each channel in the feature map array by "how important this channel is" with regard to the top predicted class. Then sum all the channels to obtain the heatmap class activation
            heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)
            heatmap = np.maximum(heatmap, 0)
            heatmap /= np.max(heatmap)
            heatmap = heatmap.reshape((7,7))
            plt.matshow(heatmap)
                
            # Load unseen image with OpenCV
            img = cv2.imread(img_path)
            
            # Make heatmap semi-transparent
            intensity = 0.5 
            
            # Resize the heatmap to be the original dimensions of the input 
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            
            # Apply colormap
            heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
            
            # Multiply heatmap by intensity and 'add' this on top of the original image
            superimposed = (heatmap * intensity) + img
            
            # Save the superimposed image to output directory
            cv2.imwrite(os.path.join("..", "output", f"{self.unseen_image}_superimposed_heatmap.png"), superimposed)
            
        # User message
        print(f"\n[INFO] The feature map has now been visualized and superimposed on {self.unseen_image}. Find image as {self.unseen_image}_superimposed_heatmap.png in 'output' directory...")


# Define behaviour when called from command line
if __name__=="__main__":
    main()