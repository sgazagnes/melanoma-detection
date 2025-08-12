# Deep Learning App for Self Skin Check-up

A privacy-friendly app to assist individuals in **screening their skin moles** using deep learning, computer vision, and real-world dermatology datasets.

---

## Project Goal

Skin cancer, especially melanoma, can be deadly if not detected early. The goal of this project is to build a **lightweight, offline-capable mobile/web application** that lets users take a photo of their mole and get an immediate, **AI-powered risk score** — helping raise awareness and encouraging medical consultation when needed.

The core of the system is a **neural network trained to classify skin lesions as benign or malignant** using real dermatology datasets.

---

## Dataset

We currently use ~45,000 dermatoscopic images from the following sources:

1. **Kaggle: Melanoma Skin Cancer Dataset (~10k images)**
   - 10,000 images
   - Source: [Kaggle - hasnainjaved/melanoma-skin-cancer-dataset](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images)

2. **ISIC Challenges: 2016 & 2020**
   - High-quality dermoscopic images and pixel-level annotations
   - ISIC 2016: 900 images
   - ISIC 2020: 33,000+ images
All datasets have two classes: `benign`, `malignant`
The datasets were **merged**, deduplicated (using perceptual hashing), and curated into a single unified dataset. 

---

## Preprocessing Pipeline

The pipeline handles several key image preparation steps:

1. **Duplicate Detection & Removal**
   - Images with identical content (via hash matching) are removed to avoid bias.

2. **Vignette Border Detection & Cropping**
   - Many dermatoscopic images include circular black/white borders.
   - A smart cropping technique estimates the radius and removes the border **while preserving aspect ratio**.

3. **Hair Removal (optional)**
   - Filters or inpainting-based methods to remove hair artifacts were explored and evaluated.

4. **Data augmentation**
    - Classical augmentation techniques are used: flipping, rotation, contraste/brightness/saliency variations

---

## Model Architecture

We use a **MobileNetV3-Small** convolutional neural network, which is:

- Lightweight and fast — ideal for mobile or embedded use
- Pretrained on ImageNet, then fine-tuned on our dataset
- Final layer modified for **binary classification**

### Training Parameters

| Parameter       | Value               |
|----------------|---------------------|
| Input size     | 224x224 RGB         |
| Batch size     | 32                  |
| Optimizer      | Adam                |
| Learning rate  | 1e-4                |
| Loss function  | BCEWithLogitsLoss   |
| Epochs         | 10–30 (ongoing tuning) |
| Class balancing| Weighted sampling for training |

---


##  Results and Evaluation

We evaluated the final model on a dedicated test set (15% of the full dataset), separated from the training and validation data.

### Dataset Split

| Split     | % of data | Purpose                               |
| --------- | --------- | ------------------------------------- |
| **Train** | \~70%     | Model learning and optimization       |
| **Val.**  | \~15%     | Hyperparameter tuning, early stopping |
| **Test**  | \~15%     | Final, unbiased performance check     |

---

###  Confusion Matrix (on Test Set)

```
                Predicted
              | Benign | Malignant
    ----------|--------|----------
    Benign    |  4862  |   267     
    Malignant |    27  |   756     
```

---

###  Classification Report

| Class            | Precision | Recall   | F1-score | Support |
| ---------------- | --------- | -------- | -------- | ------- |
| Benign           | **0.99**  | 0.95     | 0.97     | 5129    |
| Malignant        | 0.74      | **0.97** | 0.84     | 783     |
| Accuracy         |           |          | **0.95** | 5912    |
| Macro Avg        | 0.87      | 0.96     | 0.90     |         |
| Weighted Avg     | 0.96      | 0.95     | 0.95     |         |


- **Accuracy**: The model correctly classified **95%** of test images.
- **Recall (malignant)**: **97%** — nearly all melanoma cases are detected.
- **Precision (malignant)**: **74%** — about 26% of positive predictions are false alarms.

---

###  Why we prefer more false positives than false negatives

Missing a real cancer case (false negative) is far more dangerous than mistakenly flagging a benign mole (false positive).

- Only 27 malignant images were missed, that's 3.4% of all cancer cases.
- Meanwhile, 267 benign moles were incorrectly flagged as malignant, 5.2% of all benign moles.


Note that the app is not meant to replace the consultation of a medical professional — the app will  not be a replacement for diagnosis.


These are **preliminary results** and will evolve as the dataset, preprocessing, and training improve.

---

## Work in Progress

- [x] Data merging and deduplication
- [x] Border removal system (with geometric estimation)
- [x] MobileNetV3 model training and tuning
- [ ] Model visualization: false positives/negatives, saliency maps
- [ ] Deployment via Gradio + Hugging Face Space: 
- [ ] Build mobile-friendly UI for real-time image input
- [ ] Implement embedded/edge-friendly model conversion (ONNX, TFLite)
- [ ] Publish detailed blog and documentation

---

