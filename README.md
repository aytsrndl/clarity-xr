# ClarityXR — Explainable Pneumonia Detection with Uncertainty Quantification
ClarityXR is a chest X-ray classification model that goes beyond a single prediction. For every diagnosis it tells you how confident the model is and where it looked to decide. These are the two things a clinician needs before trusting a AI system.

🔗 Try the live demo:
https://huggingface.co/spaces/aytsrndl/ClarityXR

<img width="2041" height="1050" alt="image" src="https://github.com/user-attachments/assets/b075ea94-94fb-4aa0-9edc-42b34742ea17" />




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
DenseNet121 was chosen based on Rajpurkar et al. (2017), whose CheXNet model used the same architecture to reach radiologist level classification accuracy for pneumonia.The classifier head was replaced with Dropout + Linear layer, early layers were frozen and denseblock4 was fine-tuned at a lower learning rate to preserve pretrained features while allowing the classifier to learn quickly.Data was split based on uniqe patientID to prevent data leakage across train and validation datasets.

## Layer 2: Uncertainty Classification (Monte Carlo Dropout)
Dropout is kept active at inference time and each image was forward passed through the model 30 times.The predictions were averaged and the value was the diagnosis and the standard deviation was the uncertainty.This was implemented based on the findings of Gal & Ghahramani (2016), who proved dropout at inference is the mathematical equivalent of approximate Bayes inference.

<img width="1395" height="747" alt="image" src="https://github.com/user-attachments/assets/dd4e8d54-c62e-4dab-8246-3948459c3fbf" />

### Layer 3: Explainability (Grad-CAM)

Grad-CAM generates a heatmap of the X-ray, showing which regions were most influential for a model's prediction. It uses PyTorch hooks to grab the feature maps and gradients from the last convolutional layer, weights each feature map by its contribution to the output, and sums them up into a single spatial heatmap. Based on Selvaraju et al. (2017). When the model makes the correct pneumonia prediction, the heatmap identifies the opacity of the lung, the same area a radiologist would examine. If the model gets a case wrong, the heatmap shows it was looking at the wrong anatomy—like the shoulder—and that’s why.

### Layer 4: Deployment (Gradio)

The last layer combines classification, uncertainty quantification and explainability into one interactive pipeline. A user uploads a chest X-ray and gets a diagnosis, a confidence score, a Grad-CAM overlay and a recommendation to trust the prediction or to flag it for radiologist review. Built with Gradio and live hosted on Hugging Face Spaces.


---

## Repository Structure

clarity-xr/

├── README.md          
├── clarityxr.ipynb    
├── app.py              
├── requirements.txt   
└── images/                          

Model weights are hosted on the [Hugging Face Space](https://huggingface.co/spaces/aytsrndl/ClarityXR) rather than in this repo due to file size.

---

## Limitations
ClarityXR is a research and learning project and is not a clinical tool. MC Dropout does not catch all errors - there are a few predictions that are confidently wrong, with low uncertainty despite being wrong. The AUC of 0.8836 outperforms published baselines, but further validation, external test sets, and prospective clinical evaluation are needed before it can be considered for production use.

Also, ClearnessXR doesn’t do out of distribution (OOD) detection . The model gives a prediction for any image it receives, regardless of whether the input is actually a chest X-ray. In real clinical deployment, this classifier would have to be combined with an OOD detection step that flags non-X-ray inputs before running pneumonia classification. Adding this is a natural next iteration of the project .

---

## References

- Gal & Ghahramani (2016) — *Dropout as a Bayesian Approximation* (arXiv:1506.02142)
- Selvaraju et al. (2017) — *Grad-CAM: Visual Explanations from Deep Networks* (arXiv:1610.02391)
- Rajpurkar et al. (2017) — *CheXNet: Radiologist-Level Pneumonia Detection* (arXiv:1711.05225)

---

## Author

**Aytunc Sarandal** — Virginia Tech, built as part of the Microsoft AI Summer Program 2026

