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

The architecture is composed of three major components: EEG preprocessing and feature extraction, emotion classification models, and music recommendation logic.

### EEG Preprocessing & Feature Extraction

The system uses the DREAMER EEG dataset as input. For each of the 18 emotion-inducing trials per subject, the following preprocessing steps are applied:

1. EEG data are loaded and configured using MNE, with channels aligned to the standard 10-20 system.

2. Signals are bandpass filtered (1–40 Hz), segmented into 4-second epochs, and cleaned using AutoReject to remove noisy segments.

3. For each clean epoch, Power Spectral Density (PSD) is computed, and average bandpower features are extracted across five standard frequency bands: Delta, Theta, Alpha, Beta, and Gamma.

4. Final features are saved as a NumPy .npz file with shape (N, 14, 5), where N is the number of valid trials. Each sample is labeled with arousal and valence scores (on a 5-class scale: 0–4).


### Emotion Classification Models

The system implements and compares three models for classifying both arousal and valence simultaneously:
* **Simple CNN**: 
  A lightweight convolutional neural network with one convolution block (Conv → BatchNorm → ReLU → MaxPooling). It accepts the PSD feature maps of shape 1×14×5 and outputs two sets of 5-class softmax scores (for arousal and valence respectively).
* **Advanced CNN**: 
  An extended architecture with multiple convolution layers, adaptive pooling, and dropout for better generalization. This model captures spatial-frequency dependencies across EEG channels and frequency bands more effectively, resulting in improved performance and training stability.
* **Random Forest Classifier**: 
  A traditional tree-based model used as a baseline. PSD features are flattened, and a MultiOutputClassifier wraps around a RandomForestClassifier to handle multi-label prediction. It provides interpretability and allows feature importance analysis.

### Music Recommendation (Post-processing with DEAM)

After predicting the affective state, the output valence-arousal vector is passed to a recommendation module. This module compares the predicted values with those of labeled songs in the DEAM music dataset (stored in two CSV files).
> Alternatively, the ground truth valence-arousal scores provided by the DREAMER dataset can also be used for recommendation, allowing evaluation of the system’s performance under ideal label conditions.

This step includes: 
1. Cleaning the dataset by removing NaNs or missing entries.
2. Computing the Euclidean distance between the predicted (or labeled) emotion vector and each song’s emotion label.
3. Returning the top_n most similar songs based on smallest distance.

## Validation

Our system was validated using the ground truth valence and arousal labels provided by the **DREAMER EEG dataset**.

### **Classification Models**

### **CNN**

### **Random Forest Classifier**

### **Result**

As observed from the training loss curves, validation accuracy, RMSE plots, and confusion matrices, the overall performance of our model on the emotion classification task was not satisfactory. Although the CNN models converged during training, validation accuracy remained low (20–35%), and RMSE values above 1 indicate a noticeable gap between predicted and actual labels.

Possible reasons include:

* Insufficient data size: The DREAMER dataset includes only 23 subjects and 18 trials per subject, which is limited for deep learning and prone to overfitting.
* High individual variability in EEG signals: EEG data is highly subject-dependent, making generalization difficult across individuals.
* Subjective emotion labeling: The perceived valence/arousal of each video may vary per subject, adding noise to ground truth labels.
* Class imbalance: Certain emotion classes (e.g., 3 or 4) dominate the dataset, biasing the model to predict mid-range values and neglect extremes.

## Music Recommendation

The music recommendation system is based on regression-style continuous emotion predictions. Since there is no ground truth for the "correct" song, conventional classification metrics (accuracy, F1) are not applicable. Therefore, this component is not quantitatively evaluated in this report but remains demonstrative of the BCI system's practical use.

## Usage

## References
