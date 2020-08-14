# Deep learning framework for organelle fluorescent image classification

## How to set up?
1) Install:
   Python 3.6,
   Tensorflow-gpu 1.14.0,
   CUDA 10.0,
   cuDNN v7.4,
   Keras 2.2.4,
   pycharm

2) Clone this repository

3) Untar the downloaded file

## Operating environment
These models were trained on an Ubuntu 18.04 computer with 2 Intel Xeon CPUs, using a GTX TITAN X 12Gb GPU for training and testing, with 96Gb available in RAM memory.

## DataSet spec:
Dataset: DeepOrganelleDataset.zip. 
Before training, follow these data prep steps:
1) Divide the data set into training set, validation set and test set. 
2) Use "dataArgument.py" for data argument. What "dataArgument.py" does is divide each image into 4 parts, and rotate and flip them, which can be increase by 32 times. (TODO: different steps for data aug among training, val, test dataset.)

# Pre-trained model weights
Download the pre-training weights to the "pretrained" folder and untar them. The DenseNet model weights are under the folder "pretrained/densenet". If you want to change the folder name of each pre-trained model. Please open the code in "code/nets/pretrain_model.py", and edit the parameter "model_file_name".

The weights of VGG16, resnet_v2_50, resnet_V2_101, resnet_v2_152 and mobilenet_v2 can be downloaded here: https://github.com/tensorflow/models/tree/master/research/slim.

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
Run the "train_[model_name].py" file (such as "train_densenet_169.py") of the corresponding model, and then you should be able to train the model. 
In case you want to change the data set, please modify the "--images_dir" in the code. We have adopted the relative path, if your dataset is placed elsewhere, please modify "--images_dir".
```
python train_[model_name].py
   --images_dir=/your_dataset_directory
```

## Generate ROC curves and confusion matrix
Run the "predict_[model_name].py" file (such as "p_densenet_169.py") of the corresponding model, and then generate prediction data for each pre-trained models. 
We have adopted the relative path, if you want to change the dataset, please change "--images_dir".
```
python predict.py
   --images_dir=/your_dataset_directory
```
Chose the pre-trained model and run "ROC.py" and "confusion_matrix.py". You can change the model by modifying "--model_name".
The "--model_name" you can choose from "inception_v3", "vgg16", "resnet_v2_50", "resnet_v2_101", "resnet_v2_152", "mobilenet_v2", "densenet121", "densenet169" and "densenet201".
You will get the ROC curves and confusion_matrix of the pre-trained model you selected. 
You can get ROC curves and confusion matrix of all models by change "--model_name".
```
python ROC.py 
   --model_name=model_name
```
```
python confusion_matrix.py 
   --model_name=model_name
```

## Train and predict CNN
Run the "train_cnn.py" in the folder "code/CNN" to train the CNN model that we constructed. We have adopted the relative path, if your dataset is placed elsewhere, please modify "--images_dir".
```
python train_cnn.py 
   --images_dir=/your_dataset_directory
```
After training, the model will be saved in "code/CNN/save_model/CNN.h5df". Here, we save the model we trained. You can run "predict_cnn.py" in the folder "code/CNN" to get the results in the paper.
You can also retrain the CNN model, and the new training model weights will overwrite the file "code/CNN/save_model/CNN.h5df".
```
python predict_cnn.py 
   --images_dir=/your_dataset_directory
```

## Visualization:
#### Hidden Layer Output Visualization:
Run "CNN/hidden_layer_output_visualization.py", you will get the hidden layer output of each channel and the feature map with maximum activation. We have adopted the relative path, if your dataset is placed elsewhere, please modify "--images_dir".
```
python hidden_layer_output_visualization.py
   --images_dir=/your_dataset_directory
```

#### Feature Visualization:
Generate a random image and feed it to "CNN/feature_visualization.py", you will get the activation feature of the top 16 activation values.
```
python feature_visualization.py
```

#### Grad-CAM:
Run "CNN/Grad-CAM.py", you will get all images of Grad-CAM. After adjusting the code, you will get the corresponding heatmap. We have adopted the relative path, if your dataset is placed elsewhere, please modify "--images_dir".
```
python Grad-CAM.py
   --images_dir=/your_dataset_directory
```

## Code reference:
Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning.
DOI: http://dx.doi.org/10.17632/rscbjbr9sj.2#file-b267c884-6b3b-43ff-a992-57960f740d0f

## Pre-trained model in Tensorflow for Transfer Learning:
"TensorFlow-Slim image classification model library" N. Silberman and S. Guadarrama, 2016.
https://github.com/tensorflow/models/tree/master/research/slim

## DenseNet weights reference:
Keras. Chollet, Francois and others. 2015.
https://github.com/fchollet/deep-learning-models/releases/tag/v0.8

