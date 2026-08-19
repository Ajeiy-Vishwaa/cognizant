# backend/app/gradcam.py

import io
import base64
import cv2
import numpy as np
import torch
from PIL import Image

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.hook = target_layer.register_forward_hook(self.forward_hook)

    def forward_hook(self, module, inputs, output):
        self.activations = output
        if output.requires_grad:
            output.retain_grad()

    def generate(self, image_tensor, predicted_class):
        with torch.enable_grad():
            self.model.zero_grad(set_to_none=True)
            output = self.model(image_tensor)
            logit = output[0, 0]
            target_score = logit if predicted_class == 1 else -logit
            target_score.backward()

        activations = self.activations.detach()
        gradients = self.activations.grad.detach()
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam, output.detach()

    def remove_hooks(self):
        self.hook.remove()


def predict_fraud_with_gradcam(image_bytes: bytes, model, transform, device):
    original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original_array = np.array(original_image)
    
    image_tensor = transform(original_image).unsqueeze(0).to(device)

    # ConvNeXt feature layer targeting
    target_layer = model.classifier[0] if hasattr(model, 'classifier') else model.features[-1][-1]
    gradcam = GradCAM(model, target_layer)

    with torch.no_grad():
        output = model(image_tensor)
        fraud_prob = torch.sigmoid(output[0, 0]).item()

    threshold = 0.30
    predicted_class = 1 if fraud_prob >= threshold else 0
    assessment = "Fraud Detected / Suspicious" if predicted_class == 1 else "Non-Fraud / Legitimate"

    heatmap, _ = gradcam.generate(image_tensor, predicted_class)
    gradcam.remove_hooks()

    heatmap_resized = cv2.resize(heatmap, (original_array.shape[1], original_array.shape[0]))
    heatmap_uint8 = np.uint8(heatmap_resized * 255)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(original_array, 0.60, heatmap_color, 0.40, 0)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    
    _, buffer = cv2.imencode(".jpg", overlay_bgr)
    gradcam_base64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "assessment": assessment,
        "fraud_risk_score": round(fraud_prob * 100, 2),
        "decision_threshold": threshold,
        "gradcam_image_base64": f"data:image/jpeg;base64,{gradcam_base64}"
    }