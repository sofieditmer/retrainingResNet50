# Self-Assigned Portfolio: Retraining ResNet50 to Classify the American Sign Language (ASL) Alphabet

### Description of Task: Retraining the pretrained ResNet50 model to classifiying sign language letters <br>
This project was the self-assigned portfolio assignment. For this project I chose to work with the American Sign Language (ASL) corpus available on [Kaggle](https://www.kaggle.com/grassknoted/asl-alphabet). This dataset consists of a collection of 87,000 images separated in 26 classes each corresponding to a single letter in the English alphabet depicted in sign language. For this self-assigned project, I aspired to demonstrate the use of deep learning in classifying sign language letters. In other words, I wanted to use a pretrained CNN model as a feature extractor and retrain it on a new classification task involving sign language letters. I chose to use the ResNet50 model as the pretrained model to work as a feature extractor and fine-tune its parameters to use in a new network. The ResNet50 model is a CNN with 50 layers, and the pretrained version of the network has been trained on more than a million images from the ImageNet database, which is a collection of more than 14 million images and around 20,000 categories. <br>
My project consisted in producing two main python scripts. The first script prepares the ASL data, retrains the pretrained ResNet50 model on the data, evaluates its performance, and saves the model as a json-file and its weights to the output directory.
The second script is then able to load the saved model and use it to classify an unseen image of a letter depicted in sign language. This script also visualizes the feature map of the final convolutional layer of the network, to enable the user to get an insight into exactly which parts of the original image that the model is paying attention to when classifying a letter in sign language. <br>
This is a task with several apparent cultural relevancies given that the ASL serves as the main form of communication of deaf communities in the anglophone communities, and progress within the field of deep learning for sign language recognition is undoubtedly going to entail major advancement in alleviating some of the difficulties faced by the deaf community. <br>

### Content and Repository Structure <br>
If the user wishes to engage with the code and reproduce the obtained results, this section includes the necessary instructions to do so. It is important to remark that all the code that has been produced has only been tested in Linux and MacOS. Hence, for the sake of convenience, I recommend using a similar environment to avoid potential problems. 
The repository follows the overall structure below. The two python scripts, ```cnn-asl.py``` and ```use-model.py```, are located in the src folder. The outputs produced when running the scripts can be found within the output folder. The data folder contains a subset of the full dataset. If the user wishes to obtain the full dataset it is available on [Kaggle](https://www.kaggle.com/grassknoted/asl-alphabet). To obtain the full dataset, I suggest downloading it from Kaggle and uploading it to the data folder as a zip-file and then unzipping it via the command line. Alternatively, I recommend setting up the Kaggle command-line which is explained in this [article](https://necromuralist.github.io/kaggle-competitions/posts/set-up-the-kaggle-command-line-command/).

| Folder | Description|
|--------|:-----------|
| ```data``` | A folder containing a subset of the full dataset as well as a folder called unseen_images which contains example images that can be used as input for the use-model.py script. 
| ```src``` | A folder containing the python scripts for the particular assignment.
| ```output``` | A folder containing the outputs produced when running the python scripts within the src folder.
| ```requirements.txt```| A file containing the dependencies necessary to run the python script.
| ```create_venv.sh```| A bash-file that creates a virtual environment in which the necessary dependencies listed in the ```requirements.txt``` are installed. This script should be run from the command line.
| ```LICENSE``` | A file declaring the license type of the repository.

### Usage and Technicalities <br>
If the user wishes to engage with the code and reproduce the obtained results, this section includes the necessary instructions to do so. First, the user will have to create their own version of the repository by cloning it from GitHub. This is done by executing the following from the command line: 

```
$ git clone https://github.com/sofieditmer/self-assigned.git
```

Once the user has cloned the repository, a virtual environment must be set up in which the relevant dependencies can be installed. To set up the virtual environment and install the relevant dependencies, a bash-script is provided, which automatically creates and installs the dependencies listed in the requirements.txt file when executed. To run the bash-script that sets up the virtual environment and installs the relevant dependencies, the user must execute the following from the command line. 

```
$ cd self-assigned
$ bash create_venv.sh 
```

Once the virtual environment has been set up and the relevant dependencies listed in the ```requirements.txt``` have been installed within it, the user is now able to run the two scripts, ```cnn-asl.py``` and ```use-model.py```, provided in the src folder directly from the command line. The user has the option of specifying additional arguments, however, this is not required to run the script. In order to run the script, the user must first activate the virtual environment in which the script can be run. Activating the virtual environment is done as follows.

```
$ source asl_cnn_venv/bin/activate
```

Once the virtual environment has been activated, the user is now able to run the two scripts, ```cnn-asl.py``` and ```use-model.py```

```
(asl_cnn_venv) $ python cd src

(asl_cnn_venv) $ python cnn-asl.py

(asl_cnn_venv) $ python use-model.py
```

For the ```cnn-asl.py``` script, the user is able to modify the following parameters, however, as mentioned this is not compulsory:

```
-t, --train_data: str <name-of-training-data>, default = "asl_alphabet_train_subset"
-te, --test_data: str <name-of-test-data>, default = "asl_alphabet_test_subset"
-a, --augment_data: str <perform-data-augmentation-true-false>, default = "False"
-b, --batch_size: int <size-of-batches>, default = 32
-n, --n_epochs: int <number-of-epochs>, default = 15
-o, --output_filename: str <name-of-classification-report>, default = "classification_report.txt"
```

For the ```use-model.py``` script the user is able to modify the following parameters, but once again, this is not necessary:

```
-m, --model_name: str <name-of-model-to-load>, default = "saved_model.json"
-t, --train_data: str <name-of-train-data>, default = "asl_alphabet_train_subset"
-u, --unseen_image: str <name-of-input-image>, default = "unseen_img_test1.png"
````

The abovementioned parameters allow the user to adjust the pipeline, if necessary, but because default parameters have been set, it makes the script run without explicitly specifying these arguments.  

### Output <br>
When running the ```cnn-asl.py```script, the following files will be saved in the ```output``` folder: 
1. ```model_summary.txt``` Summary of the model architecture.
2. ```model_architecture.png``` Visual representation of the model architecture.
3. ```model_loss_accuracy_history.png``` Plot showing the loss and accuracy learning curves of the model during training.
4. ```classification_report.txt``` Classification metrics of the model performance.
5. ```saved_model.json``` The model saved as a JSON-file.
6. ```model_weights.h5``` The model weights saved in the HDF5 format. ¨

When running the ```use-model.py```script, the following files will be saved in the ```output``` folder: 
1. ```unseen_image_superimposed_heatmap.png``` Superimposed heatmap on unseen image.
2. ```unseen_image_prediction.txt``` Model prediction of unseen image.

### Discussion of Results <br>
The retrained ResNet50 achieved a weighted average accuracy score of 4% on the ASL alphabet [see Classification Report](https://github.com/sofieditmer/self-assigned/blob/main/output/classification_report.txt). This is surprisingly low considering that a very deep, pretrained, convolution neural network model was used.
When assessing the loss/accuracy plot of the model, there are clear signs of underfit learning curves, suggesting that the model is not able to sufficiently learn from the dataset (see Figure 1). In particular this is indicated by the relatively high loss for both the training and validation data, as well as the fact that the loss curves remain relatively flat regardless of training. The slight gap between the training and validation loss indicate that the model is learning slightly from the training data, given that the training loss decreases more compared to the validation loss.  Furthermore, the training and validation accuracy curves remain low regardless of training, and do not increase at any point, once again suggesting that the model struggles with learning from the data. 
Taken together, the classification report as well as the learning curves of the model seem to suggest that the model is underfitting the data. One reason for this might be that the amount of data is simply not sufficient for the model to learn. Another reason might be the model specifications. Modifying the model parameters would potentially improve the performance. Making the model less restrictive might potentially enable it to learn more from the data. Grid search might be a potential solution to optimizing the model hyperparameters. 

<img src="https://github.com/sofieditmer/self-assigned/blob/main/output/model_loss_accuracy_history.png" width="500">
Figure 1: Loss/accuracy learning curves of the model. <br>

<br> Another aspect of the project was the visualization of the feature map of the last convolutional layer of the network. This method was implemented in order to provide the user with an insight into how the model is working “qualitatively”. In other words, visualizing the feature map of the last convolutional layer of the network enables the user to see exactly which parts of the original image the model is paying attention to when making its predictions. The activation heatmaps superimposed on 4 test images are shown below (see Figure 2). In all four instances, it becomes evident that the model is paying attention to the center of the image, suggesting that the most informative features are located here. However, it is important to remark that the weighted average accuracy obtained by the model was only 4%, suggesting that these activation heatmaps might not be very enlightening. 

<p float="left">
  <img src="https://github.com/sofieditmer/self-assigned/blob/main/output/unseen_img_test1_superimposed_heatmap.png" width="200" height = "200" />
  <img src="https://github.com/sofieditmer/self-assigned/blob/main/output/unseen_img_test2_superimposed_heatmap.png" width="200" height = "200" /> 
  <img src="https://github.com/sofieditmer/self-assigned/blob/main/output/unseen_img_test3_superimposed_heatmap.png" width="200" height = "200" />
  <img src="https://github.com/sofieditmer/self-assigned/blob/main/output/unseen_img_test4_superimposed_heatmap.png" width="200" height = "200" />
</p> 
Figure 2: Superimposed activation heatmaps on unseen images. <br>

### License <br>
This project is licensed under the MIT License - see the [LICENSE](https://github.com/sofieditmer/self-assigned/blob/main/LICENSE) file for details.

### Contact Details <br>
If you have any questions feel free to contact me on [201805308@post.au.dk](201805308@post.au.dk)
