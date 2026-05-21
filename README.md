# ClarityXR — Explainable Pneumonia Detection with Uncertainty Quantification
ClarityXR is a chest X-ray classification model that goes beyond a single prediction. For every diagnosis it tells you how confident the model is and where it looked to decide. These are the two things a clinician needs before trusting a AI system.

🔗 Try the live demo:
https://huggingface.co/spaces/aytsrndl/ClarityXR



---

## Why This Project Exists
A standard classification model only outputs "positive/negative" when it makes a decision with no reliability layer. A model that is 88% accurate is still wrong 12% of the time and without uncertainty estimation there is no way to know when the model is more likely to fail.ClarityXR adresses this with two layers on top of the base classifier:
  - Monte Carlo Dropout: quantifies prediction uncertainty so unreliable cases can be escalated for manual review.
  - Grad-CAM: produces heatmap that let a clinican verify the model is reasoning by focusing on the correct portion of the image.

---

## Architecture

| Component | Detail |
|-----------|--------|
| Base model | DenseNet-121, pretrained on ImageNet |
| Dataset | RSNA Pneumonia Detection Challenge — 30,227 DICOM chest X-rays |
| Fine-tuning | denseblock4 + custom classifier head (discriminative learning rates: 1e-5 / 1e-4) |
| Class imbalance | WeightedRandomSampler + weighted BCE loss (2:1 normal:pneumonia) |
| Validation AUC | **0.8836** |

## Layer 1: Classification
DenseNet121 was chosen based on Rajpurkar et al. (2017), whose CheXNet model used the same architecture to reach radiologist level classification accuracy for pneumonia.The classifier head was replaced with Dropout + Linear layer, early layers were frozen and denseblcok4 was fine-tuned at a lower learning rate to preserve pretrained features while allowing the classifier to learn quickly.Data was split based on uniqe patientID to prevent data leakage across train and validation datasets.

## Layer 2: Uncertainty Classification (Monte Carlo Dropout)
Dropout is kept active at inference time and each image was forward passed through the model 30 times.The predictions were averaged and the value was the diagnosis and the standard deviation was the uncertainty.This was implemented based on the findings of Gal & Ghahramani (2016), who proved dropout at inference is the mathematical equivalent of approximate Bayes inference.

<img width="1395" height="747" alt="image" src="https://github.com/user-attachments/assets/dd4e8d54-c62e-4dab-8246-3948459c3fbf" />



