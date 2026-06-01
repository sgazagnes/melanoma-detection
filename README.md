# Deep Learning App for Self Skin Check-up

A privacy-friendly app to assist individuals in **screening their skin moles** using deep learning, computer vision, and real-world dermatology datasets.

---

> ⚠️ **Critical Disclaimer**
>
> This model was trained exclusively on **dermoscopic images** — high-magnification, uniformly lit, clinically captured photographs of skin lesions. These images have very different sharpness, contrast, color profile, and texture compared to photos taken with a standard webcam or smartphone camera.
>
> **Inference results on non-dermoscopic images (webcam, phone camera, etc.) should be treated with extreme caution and should NOT be used as a basis for any medical decision.** The model may produce unreliable or misleading predictions on such inputs. Always consult a qualified dermatologist for any skin concern.
>
> This project was intended for research and educational purposes only.

---

## Project Goal

Skin cancer, especially melanoma, can be deadly if not detected early. The goal of this project is to build a **lightweight, offline-capable application** that lets users scan a mole and get an immediate, **AI-powered risk score** — helping raise awareness and encouraging medical consultation when needed.

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

All datasets have two classes: `benign`, `malignant`.
The datasets were **merged**, deduplicated (using perceptual hashing), and curated into a single unified dataset of approx 40000 images.

---

## Preprocessing Pipeline

The pipeline handles several key image preparation steps:

1. **Duplicate Detection & Removal**
   - Images with identical content (via hash matching) are removed to avoid bias.

2. **Vignette Border Detection & Cropping**
   - Many dermatoscopic images include circular black/white borders. A smart cropping technique estimates the radius and removes the border while preserving aspect ratio.

3. **Hair Removal (optional)**
   - Filters or inpainting-based methods to remove hair artifacts were explored and evaluated.

4. **Data Augmentation**
   - Classical augmentation techniques: flipping, rotation, contrast/brightness/saturation variations.

---

## Model Architecture

We use a **MobileNetV3-Small** convolutional neural network, which is:

- Lightweight and fast — ideal for mobile or embedded use
- Pretrained on ImageNet, then fine-tuned on our dataset
- Final layer modified for **binary classification**

### Training Parameters

| Parameter      | Value               |
|----------------|---------------------|
| Input size     | 224x224 RGB         |
| Batch size     | 32                  |
| Optimizer      | AdamW               |
| Learning rate  | 1e-4                |
| Loss function  | BCEWithLogitsLoss (pos_weight balanced) |
| Epochs         | 10–30 (early stopping on PR-AUC) |
| Class balancing| Weighted random sampler |

---

## Results and Evaluation

We evaluated the final model on a dedicated test set (15% of the full dataset), separated from training and validation data.

### Dataset Split

| Split      | % of data | Purpose                               |
| ---------- | --------- | ------------------------------------- |
| **Train**  | ~70%      | Model learning and optimization       |
| **Val.**   | ~15%      | Hyperparameter tuning, early stopping |
| **Test**   | ~15%      | Final performance check               |

---

### Confusion Matrix (on Test Set)

```
                Predicted
              | Benign | Malignant
    ----------|--------|----------
    Benign    |  4862  |   267
    Malignant |    27  |   756
```

---

### Classification Report

| Class         | Precision | Recall   | F1-score | Support |
| ------------- | --------- | -------- | -------- | ------- |
| Benign        | **0.99**  | 0.95     | 0.97     | 5129    |
| Malignant     | 0.74      | **0.97** | 0.84     | 783     |
| Accuracy      |           |          | **0.95** | 5912    |
| Macro Avg     | 0.87      | 0.96     | 0.90     |         |
| Weighted Avg  | 0.96      | 0.95     | 0.95     |         |

- **Accuracy**: 95% of test images correctly classified.
- **Recall (malignant)**: 97% — nearly all melanoma cases detected.
- **Precision (malignant)**: 74% — ~26% of positive predictions are false alarms.

### Why We Prefer False Positives Over False Negatives

Missing a real cancer case (false negative) is far more dangerous than mistakenly flagging a benign mole (false positive).

- Only 27 malignant images were missed — 3.4% of all cancer cases.
- 267 benign moles were incorrectly flagged — 5.2% of all benign moles.

> These results hold **only on dermoscopic test images**. Performance on webcam or phone images is expected to be significantly lower due to the domain gap described in the disclaimer above.

---

## Real-Time Webcam Pipeline

A real-time inference pipeline (`camera_inference.py`) has been developed to test the model on a live camera stream. It includes:

1. **Auto camera discovery** — scans to find an available device.
2. **Mole detection** — blob detection with centroid filtering to locate a mole centered in the frame.
3. **Sharpness check** — Laplacian variance gate to reject blurry frames.
4. **Guided capture** — countdown + burst of 15 frames, selects the sharpest one.
5. **Preprocessing** — tight contour crop + Lanczos upscale + contrast enhancement.
6. **Inference + display** — side-by-side panel showing ROI, tight crop, CLAHE image, label and confidence. Press **N** for a new scan, **Q** to quit.

### Hardware Notes

The pipeline was developed with a **Logitech C270** (fixed focus) and an Android phone via **DroidCam**. Image quality from consumer cameras differs substantially from dermoscopic equipment.For more reliable results, a **USB dermatoscope** should be used. Predictions from standard webcams or phone cameras should be considered indicative only. Again, this project is meant for Deep Learning courses, and predictions should be taken with caution.

---

## Work in Progress

- [x] Data merging and deduplication
- [x] Border removal system (with geometric estimation)
- [x] MobileNetV3 model training and tuning
- [x] Real-time webcam inference pipeline with guided capture
- [x] CLAHE contrast enhancement to partially close the domain gap
- [ ] Fine-tuning on phone/webcam images to reduce domain gap
- [ ] Model visualization: saliency maps, false positive/negative analysis
- [x] Deployment via Gradio + Hugging Face Space: https://huggingface.co/spaces/sgazagnes/melanoma-detection
- [ ] Build mobile-friendly UI for real-time image input
- [ ] Implement edge-friendly model conversion (ONNX, TFLite)

---
