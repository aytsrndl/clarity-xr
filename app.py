"""
ClarityXR — Explainable Pneumonia Detection with Uncertainty Quantification

A clinical-grade chest X-ray classifier that combines DenseNet-121 prediction,
Monte Carlo Dropout uncertainty estimation, and Grad-CAM visual explainability.

Author: Aytunc Sarandal
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gradio as gr

# ─── Device Setup ───
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Model Setup ───
model = models.densenet121(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(in_features=1024, out_features=1)
)
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ─── Transforms ───
val_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ─── Hooks for Grad-CAM ───
feature_maps = []
gradients = []


def forward_hook(module, input, output):
    """Capture feature maps during the forward pass for Grad-CAM."""
    feature_maps.append(output.clone())


def backward_hook(module, grad_input, grad_output):
    """Capture gradients during the backward pass for Grad-CAM."""
    gradients.append(grad_output[0].clone())


# Remove any existing hooks and register new ones
for module in model.modules():
    module._forward_hooks.clear()
    module._backward_hooks.clear()

target_layer = model.features.denseblock4
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)


# ─── Helper Functions ───

def enable_dropout(model):
    """
    Selectively reactivate dropout layers during inference.

    After model.eval() disables all stochastic layers,
    this function re-enables only dropout while keeping
    BatchNorm frozen — required for MC Dropout inference.
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def preprocess_image(image):
    """
    Convert an uploaded image (numpy array) into a model-ready tensor.

    Args:
        image: Numpy array from Gradio upload

    Returns:
        Tensor of shape [1, 3, 224, 224] with ImageNet normalization
    """
    image = Image.fromarray(image)
    image = image.convert("RGB")
    image = val_transform(image)
    image = image.unsqueeze(0)
    return image


def mc_dropout_predict(model, image, n_iterations=30):
    """
    Run Monte Carlo Dropout inference on a single image.

    Performs N stochastic forward passes with dropout active
    while keeping BatchNorm frozen, returning the mean prediction
    and standard deviation as a measure of model uncertainty.
    Based on Gal & Ghahramani (2016), arXiv:1506.02142.

    Args:
        model: Trained DenseNet-121 with dropout in classifier
        image: Preprocessed image tensor of shape [1, 3, 224, 224]
        n_iterations: Number of stochastic forward passes (default: 30)

    Returns:
        mean_pred: Average predicted probability of pneumonia
        std_pred: Standard deviation across N passes (uncertainty)
    """
    model.eval()
    enable_dropout(model)

    predictions = []
    with torch.no_grad():
        for i in range(n_iterations):
            output = model(image.to(DEVICE)).squeeze(1)
            prob = torch.sigmoid(output).cpu().numpy()
            predictions.append(prob)

    predictions = np.array(predictions)
    mean_pred = predictions.mean()
    std_pred = predictions.std()

    return mean_pred, std_pred


def grad_cam(model, image):
    """
    Generate a Grad-CAM heatmap for a single image.

    Computes gradient-weighted class activation maps by capturing
    feature maps and gradients from the last convolutional layer
    via registered hooks. Based on Selvaraju et al. (2017), arXiv:1610.02391.

    Args:
        model: Trained DenseNet-121 with hooks registered on target layer
        image: Preprocessed image tensor of shape [1, 3, 224, 224]

    Returns:
        heatmap: Numpy array of shape (224, 224) with values normalized
                 to 0-1, ready for overlay on the original image
    """
    feature_maps.clear()
    gradients.clear()

    # Forward pass
    model.eval()
    output = model(image.to(DEVICE)).squeeze(1)

    # Backward pass
    model.zero_grad()
    output.backward()

    # Get feature maps and gradients
    fmaps = feature_maps[0]
    grads = gradients[0]

    # Average gradients to get weights
    weights = grads.mean(dim=[2, 3])
    weights = weights.unsqueeze(-1).unsqueeze(-1)

    # Weighted combination
    heatmap = (weights * fmaps).sum(dim=1)
    heatmap = torch.relu(heatmap)

    # Normalize to 0-1
    heatmap = heatmap.squeeze().cpu().detach().numpy()
    if heatmap.max() - heatmap.min() > 0:
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    else:
        heatmap = np.zeros_like(heatmap)

    # Resize to match image
    heatmap = np.uint8(255 * heatmap)
    heatmap = Image.fromarray(heatmap).resize((224, 224))
    heatmap = np.array(heatmap) / 255.0

    return heatmap


# ─── Main Predict Function ───

def predict(image):
    """
    Full ClarityXR pipeline: preprocess, predict, quantify uncertainty,
    and generate visual explanation for an uploaded chest X-ray.

    Args:
        image: Numpy array from Gradio image upload

    Returns:
        fig: Matplotlib figure with original, Grad-CAM, and overlay panels
        result_text: Diagnosis report with prediction, uncertainty, and recommendation
    """
    # Preprocess
    tensor = preprocess_image(image)

    # MC Dropout — uncertainty quantification
    mean_pred, std_pred = mc_dropout_predict(model, tensor)

    # Grad-CAM — visual explainability
    heatmap = grad_cam(model, tensor)

    # Create overlay visualization
    original_resized = Image.fromarray(image).convert("RGB").resize((224, 224))
    original_array = np.array(original_resized) / 255.0

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(original_array)
    axes[0].set_title("Original", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(original_array)
    axes[2].imshow(heatmap, cmap="jet", alpha=0.4)
    axes[2].set_title("Overlay", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()

    # Build diagnosis report
    diagnosis = "Pneumonia" if mean_pred > 0.5 else "Normal"
    confidence = "High" if std_pred < 0.05 else "Medium" if std_pred < 0.08 else "Low"
    action = "Trust model prediction" if std_pred < 0.05 else "Flag for radiologist review"

    result_text = f"Diagnosis: {diagnosis}\n"
    result_text += f"Probability: {float(mean_pred):.1%}\n"
    result_text += f"Uncertainty (std): {float(std_pred):.4f}\n"
    result_text += f"Confidence: {confidence}\n"
    result_text += f"Recommendation: {action}"

    return fig, result_text


# ─── Gradio Interface ───

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Upload Chest X-Ray"),
    outputs=[
        gr.Plot(label="Grad-CAM Analysis"),
        gr.Textbox(label="Diagnosis Report")
    ],
    title="ClarityXR — Explainable Pneumonia Detection",
    description=(
        "Upload a chest X-ray to receive a diagnosis with uncertainty "
        "quantification and visual explainability. Built with DenseNet-121, "
        "Monte Carlo Dropout, and Grad-CAM."
    ),
    examples=None,
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
