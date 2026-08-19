import io
import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", "convnext_final_development_best.pth")
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    # 1. Instantiate ConvNeXt Tiny architecture (9 blocks in stage 5)
    model = models.convnext_tiny(weights=None)
    
    # Adjust classifier head for binary output classification
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, 1)

    # 2. Check existence and load checkpoint
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    # 3. Extract dictionary containing state_dict
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # 4. Clean prefix mismatches and index mappings
    clean_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("model.", "") if key.startswith("model.") else key
        
        # Remap classifier indices if saved from wrapped module variants
        new_key = new_key.replace("classifier.1.", "classifier.0.")
        new_key = new_key.replace("classifier.3.", "classifier.2.")
        
        clean_state_dict[new_key] = value

    # 5. Load state dict
    model.load_state_dict(clean_state_dict)
    model.to(device)
    model.eval()
    return model

model = load_model()

# Image Preprocessing Pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_fraud(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        score = torch.sigmoid(output).item()

    threshold = 0.3
    is_fraud = score >= threshold
    assessment = "Fraud Detected / Suspicious" if is_fraud else "Non-Fraud / Legitimate"

    return {
        "assessment": assessment,
        "fraud_risk_score": round(score * 100, 2),
        "decision_threshold": threshold
    }