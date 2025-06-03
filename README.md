# BCI-final-project

## Introduction
This project aims to build **an emotion-based music recommendation system** that leverages both the DREAMER and DEAM datasets to suggest music that aligns with a user's current emotional state, as inferred from EEG signals and affective ratings.

We utilize the [DREAMER dataset](https://ieeexplore.ieee.org/document/7887697) to analyze users' emotional states. DREAMER is a widely-used dataset in **affective computing and brain-computer interface (BCI) research**. It provides EEG recordings collected while participants watched emotion-eliciting video clips, along with their self-assessed valence, arousal, and dominance scores.

For the music recommendation component, we adopt the [DEAM dataset](https://cvml.unige.ch/databases/DEAM/) (Database for Emotional Analysis of Music), which contains 2000 music excerpts annotated with dynamic valence and arousal values. DEAM is commonly used for music emotion recognition and regression tasks.

![image](https://github.com/user-attachments/assets/da34875c-49fb-4d90-ada3-10c5c51e7f94)


### Emotional Dimensions:

The system focuses on two core affective dimensions:

* **Valence**: Indicates the pleasantness of an emotion. A higher value reflects a more positive or happy state.
* **Arousal**: Indicates the intensity or activation level of an emotion. A higher value represents a more excited or energetic state.

### System Architecture Overview:

The implementation is divided into two main stages:

1. **Emotion Prediction Stage**
We use EEG signals from the DREAMER dataset to predict users' valence and arousal scores through machine learning models. Three different models were tested and compared. Although the prediction accuracy has room for improvement, the results still serve as a useful emotional reference.

2. **Music Recommendation Stage**
The predicted emotion values are matched with the DEAM music database through regression or distance-based methods (Euclidean distance) to retrieve the most emotionally-aligned music excerpts. This forms a bridge from EEG input to music output via emotional representation.



## Model Framework
![image](https://github.com/user-attachments/assets/ef718d33-9a9e-43de-a8b4-240811a94a79)


## Validation

## Usage

## References
