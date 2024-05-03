# BirdCLEF 2024

![Image de présentation du projet BirdCLEF 2024](image/birds.jpg)

[Competition Link on Kaggle](https://www.kaggle.com/competitions/birdclef-2024)

## Overview
This project is designed for the BirdCLEF 2024 competition hosted on Kaggle. The goal of this challenge is to build a machine learning model capable of identifying bird species from audio recordings.

## Dataset
The dataset comprises various audio recordings of bird songs. Each recording is segmented into chunks of 5 seconds. These segments serve as the primary data from which features are extracted.

## Feature Extraction
We convert the 5-second audio chunks into Mel Spectrograms. Mel Spectrograms are a visual representation of the spectrum of frequencies in a sound as they vary with time. This transformation is crucial as it allows us to use image-based classifiers for our task.

## Model
The extracted Mel Spectrograms are used as inputs to a vision classifier. This approach leverages techniques commonly used in image recognition tasks to identify the bird species based on the audio-derived visual patterns.
