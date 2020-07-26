# Deep learning framework for organelle fluorescent image classification

## Code reference:
Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning.
DOI: http://dx.doi.org/10.17632/rscbjbr9sj.2#file-b267c884-6b3b-43ff-a992-57960f740d0f

## Pre-trained model in Tensorflow for Transfer Learning:
"TensorFlow-Slim image classification model library" N. Silberman and S. Guadarrama, 2016.
https://github.com/tensorflow/models/tree/master/research/slim

## DenseNet weights reference:
Keras. Chollet, Francois and others. 2015.
https://github.com/fchollet/deep-learning-models/releases/tag/v0.8

## How to set up?
1) Install:
   Python 3.6, Tensorflow 1.14.0, Keras 2.2.4, pycharm

2) Clone this repository

3) Untar the downloaded file

## DataSet spec:
Dataset: DeepOrganelleDataset.zip. 
Before training, follow these data prep steps:
1) Divide the data set into training set, validation set and test set. 
2) Use "dataArgument.py" for data argument. What "dataArgument.py" does is divide each image into 4 parts, and rotate and flip them, which can be increase by 32 times. (TODO: different steps for data aug among training, val, test dataset.)

# Pre-trained model weights
Download the pre-training weights to the "pretrained" folder. The weights of VGG16, resnet_v2_50,
resnet_V2_101, resnet_v2_152 and mobilenet_v2 can be downloaded here: https://github.com/tensorflow/models/tree/master/research/slim.

The weights of densenet121, densenet169 and densenet201 can be downloaded here: https://github.com/fchollet/deep-learning-models/releases/tag/v0.8.

## Data augmentation:
Modify the path in "dataAugment.py". This method divides an image set into 4 parts, rotate and flip each part. Atfer data augmentation, the amount of data will be increased by 32 times.
```
python dataAugment.py
   --source_dir=/source_path
   --save_dir=/save_path
   --category=data_set ("train", "val" or "test")
```

## How to train:
Edit the "train_[model_name].py" file (such as "train_densenet_169.py") of the corresponding model, and change the "path" in line 27 accordingly, and then you should be able to train the model. 
In case you want to change the data set, please modify the "--images_dir" in the code.
```
python train_[model_name].py
   --path=/project_path
   --images_dir=/project_path/images
```

## Generate ROC curves and confusion_matrix
First edit "predict.py" file and change the "path" in line 24 which is the path of your project and the "model_name" in line 26 which is the model you want to predict. Then run "predict.py" repeatedly to generate prediction data for all pre-trained models. 
```
python predict.py
   --images_dir=/project_path/images
```
Chose the pre-trained model and run "ROC.py" and "confusion_matrix.py". You will get the ROC curves and confusion_matrix of the pre-trained model you selected and the ROC curves of each pre-trained models. ROC curves and confusion_matrix for all models can be obtained in the same manner.
```
python ROC.py 
   --path=/project_path 
   --model_name=model_name
```
```
python confusion_matrix.py 
   --path=/project_path 
   --model_name=model_name
```

## Visualization:
First, open directory of "CNN", and run "CNN/train.py" to train the CNN model.
```
python train.py 
   --path=/project_path/code/CNN
   --images_dir=/project_path/images
```

#### Hidden Layer Output Visualization:
--Modify the correct path and run "CNN/hidden_layer_output_visualization.py", you will get the hidden layer output of each channel and the feature map with maximum activation.
```
python hidden_layer_output_visualization.py
   --path=/project_path/code/CNN
   --images_dir=/project_path/images
```

#### Feature Visualization:
--Generate a random image and feed it to "CNN/feature_visualization.py", you will get the activation feature of the top 16 activation values.
```
python feature_visualization.py
   --path=/project_path/code/CNN
```

#### Grad-CAM:
--Modify the correct path and run "CNN/Grad-CAM.py", you will get all images of Grad-CAM. After adjusting the code, you will get the corresponding heatmap.
```
python Grad-CAM.py
   --path=/project_path/code/CNN
   --images_dir=/project_path/images
```

