
# Speaker Identification Project

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Model Architecture](#model-architecture)
- [Results](#results)
  
## Introduction

This project focuses on the identification of speakers from audio samples.

## Features

- **Preprocessing: Feature Extraction**: Extraction of MFCCs.
- **Model Training**: Convolutional Neural Networks based model for classification.
- **Evaluation**: Performance metrics such as accuracy and sparse cross entropy.

## Installation

### Prerequisites

- Python 3.10+
- Tensorflow 2.10

### Dependencies

All required Python packages are listed in the `requirements.txt` file. Install them using pip:

pip install -r requirements.txt


## Usage

All functions inside application after running Design.py
The input file should be called analisys.wav in root folder
Be aware that this project using 1 sec fragments for both training and prediction. Cut needed audio manually or comment penultimate line in Predict.py, all files will stay in segmented folder KEKWait


## Datasets

This project utilizes publicly available datasets such as:

- [VoxCeleb](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/)
- [Kaggle](https://www.kaggle.com/datasets/vjcalling/speaker-recognition-audio-dataset)

## Model Architecture

The project uses a Convolutional Neural Network (CNN) architecture. The model is built using the following layers:

- **Conv2D**: Convolutional layers for feature extraction.
- **MaxPooling2D**: Pooling layers for dimensionality reduction.
- **Flatten**: Flattening layer to prepare for dense layers.
- **Dense**: Fully connected layers for classification.


## Results

The model achieves an accuracy of 98% on the test dataset for my voice. Used dataset structure:
  1. My voice (3000 samples)
  2. 50 speakers voice (Also 3000 samples. 60 from each)
