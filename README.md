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
![image](https://github.com/user-attachments/assets/0e086e79-abec-49d0-bca7-438f1321ea87)


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

### **Classification Models**

Our system was validated using the ground truth valence and arousal labels provided by the DREAMER EEG dataset. Since our emotion classification models were designed as a multi-output 5-class classifier, the evaluation was carried out using accuracy, Root Mean Square Error (RMSE), and confusion matrices.

### **Music Recommendation**

The music recommendation system is based on regression-style continuous emotion predictions. Since there is no ground truth for the "correct" song, conventional classification metrics (accuracy, F1) are not applicable. Therefore, this component is not quantitatively evaluated in this report but remains demonstrative of the BCI system's practical use.

## Usage
* Data preprocess:
  - Download Dreamer dataset from Kaggle [DREAMER dataset](https://www.kaggle.com/datasets/phhasian0710/dreamer)
  - run data preprocess code
  ```
  python data_preprocess.py
  ```
  - Folder structure:
  ```
  project\
  │
  ├── Dreamer\DREAMER.mat                        ← dataset
  ├── data_preprocess.py              ← data preprocess code
  ├── dreamer_psd_features.npz        ← training data
  ├── train_complex_cnn.py            ← complex CNN training
  ├── train_easy_cnn.py               ← easy CNN training
  └── train_RF.py                     ← RF training
  ```

* Training:
  - To run complex CNN:
  ```
  python train_complex_cnn.py
  ```
  - To run easy CNN:
  ```
  python train_easy_cnn.py
  ```
  - To run RF:
  ```
  python train_RF.py
  ```

* Classification
  - You can modified `valence`  `arousal` values in `classicication.m` to get the recommanded music


## Result

This study evaluates the performance of three different models for EEG-based emotion classification: a simple CNN, a Random Forest (RF) classifier, and an advanced CNN architecture. The results and analysis are summarized as follows:

### Simple CNN Model
![image](https://github.com/user-attachments/assets/dd2b9848-0cdc-45c4-9fae-30a94bb4e1aa)

* Training loss quickly converged to near zero within the first few hundred epochs, indicating that the model effectively fit the training data.
* Validation accuracy fluctuated between 20% and 35% for both arousal and valence, showing that the model struggled to generalize.
* This may be attributed to the limited amount of training data, which caused the model to reduce loss but fail to learn sufficiently representative features for accurate prediction.

### Random Forest Model
![image](https://github.com/user-attachments/assets/7631ee1b-69ca-489c-a45d-fba5e9986972)
![image](https://github.com/user-attachments/assets/df474d58-8747-4136-ae29-777b16f5d0dd)

* The Random Forest model was trained on flattened PSD features using a traditional machine learning approach.
* Regression results (Figure 1):
  - Arousal RMSE: 1.07
  - Valence RMSE: 1.13
  - Predictions were mostly centered around the average score (~3), suggesting the model had difficulty capturing extreme emotional values. The distribution was compressed toward the center.
* Confusion matrix analysis (Figure 2):
  - Arousal accuracy: 19.28%
  - Valence accuracy: 36.14%
  - While the model performed slightly better on valence than arousal, overall classification performance remained weak, indicating that the PSD features might lack strong emotional discriminative power.


### Advanced CNN Model
![image](https://github.com/user-attachments/assets/22147e06-36bb-4527-95a5-a32f5f769529)
* The deeper CNN architecture exhibited a smoother and more stable training loss curve, demonstrating improved training dynamics.
* However, validation accuracy remained in the 20–35% range, similar to the simple CNN model.
* Despite its increased complexity, the model's performance showed only marginal improvement. This may be due to insufficient data, limited signal clarity in EEG, and high inter-subject variability in brain responses.

### Possible causes
* **Insufficient data size**: The DREAMER dataset includes only 23 subjects and 18 trials per subject, which is limited for deep learning and prone to overfitting.
* **High individual variability in EEG signals**: EEG data is highly subject-dependent, making generalization difficult across individuals.
* **Subjective emotion labeling**: The perceived valence/arousal of each video may vary per subject, adding noise to ground truth labels.
* **Class imbalance**: Certain emotion classes (e.g., 3 or 4) dominate the dataset, biasing the model to predict mid-range values and neglect extremes.

### Summary
This study utilized the DREAMER dataset for EEG-based emotion classification. Compared to [studies using larger datasets](https://www.frontiersin.org/files/Articles/1289816/fpsyg-14-1289816-HTML/image_m/fpsyg-14-1289816-t001.jpg), our models achieved relatively lower classification accuracy (literature typically reports ~60–75%). However, many of those studies rely heavily on complex data augmentation, deep temporal models (e.g., LSTM, CNN-LSTM), or subject-independent generalization, often requiring much more training data and computation.

In contrast, our goal was to build a simplified and modular pipeline for fast experimentation and practical deployment. Notably, while large models often perform well in dependent settings (same-subject training/testing), many lack results in independent settings (cross-subject evaluation), indicating that emotion recognition in BCI remains an open and evolving challenge.

Despite modest classification performance, our system demonstrates a key contribution: **Emotion-driven music recommendation based on EEG**.

This part of the system can function independently of prediction models by directly utilizing the ground truth valence-arousal scores provided in the DREAMER dataset, offering a practical proof-of-concept for brain-signal-to-music systems.

To our knowledge, no previous study has directly applied EEG-derived emotional states to music recommendation. While some existing approaches rely on facial expressions or voice features to infer mood and suggest songs, such methods may not reflect internal emotional states reliably and lack generalizability.

By contrast, EEG offers a direct window into a user’s internal emotional landscape, enabling more authentic and personalized emotion-based recommendation. This highlights the system’s novelty and potential for future affective BCI applications.



## References
[DEAP dataset](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/)
[SEED dataset](https://bcmi.sjtu.edu.cn/home/seed/)
[Mini review: Challenges in EEG emotion recognition](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2023.1289816/full)
[Emotion-Music-Recommendation](https://github.com/aj-naik/Emotion-Music-Recommendation)
