import os
import torch
from PIL import Image
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from app.model import load_model

# Standard ConvNeXt preprocessing pipeline
transform_image = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def evaluate():
    model = load_model()
    model.eval()

    test_dir = "test_data"  # Subfolders: test_data/0_no_damage and test_data/1_damage
    y_true, y_pred = [], []

    if not os.path.exists(test_dir):
        print(f"Directory '{test_dir}' not found. Please create it and add test images.")
        return

    with torch.no_grad():
        for class_name in os.listdir(test_dir):
            class_path = os.path.join(test_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            
            label = 1 if "damage" in class_name.lower() and "no" not in class_name.lower() else 0

            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                
                try:
                    image = Image.open(img_path).convert("RGB")
                    tensor = transform_image(image).unsqueeze(0)
                    
                    outputs = model(tensor)
                    probs = torch.sigmoid(outputs)
                    pred = 1 if probs.item() >= 0.5 else 0

                    y_true.append(label)
                    y_pred.append(pred)
                except Exception as e:
                    print(f"Error processing {img_name}: {e}")

    if y_true:
        print(f"\n--- Evaluation Results ---")
        print(f"Accuracy:  {accuracy_score(y_true, y_pred) * 100:.2f}%")
        print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
        print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
        print(f"F1-Score:  {f1_score(y_true, y_pred, zero_division=0):.4f}")
        print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    evaluate()