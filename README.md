# RBE595 AI for Autonomous Vehicles Final Project


This repository holds the code that we used to train each of our models on the KITTI [dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php). We experimented with a number of architectures, but the most successful was the FCN based on the work by Luca Caltagirone et al. To see a demonstration of each of the models, look at inference.ipynb
---
## Contents
- CNN-Trainig.ipynb: This notebook trained the CNN model mentioned in the paper.
- FCN-Training.ipynb: This notebook trained the FCN model.
- inference.ipynb: Demonstrate inference on each of the three models: FCN, CNN, and LSTM.
- lable-upsample.ipynb: Process the label images to use as labels for the FCN model. Each label needs to be (384,1248) in size with 2-channels per pixel to classify as either road or background.
- LSTM-preprocessing.ipynb: Carries out data augmentation on the original dataset for training the LSTM and CNN models. Breaks each input image into several smaller ones by moving a sliding window across each of the images and saving the result.
- LSTM-training-notebook.ipynb: Trains the LSTM model.
- network.ipynb: Prototype model to try and understand the original task. The contents of this notebook are not covered in the paper.