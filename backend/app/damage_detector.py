import cv2
from pathlib import Path
from typing import Union, Dict, Any, List
import numpy as np
from PIL import Image
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

class VehicleDamageDetector:
    def __init__(
        self,
        model_path: Union[str, Path] = "models/best.pt",
        confidence: float = 0.25,
        iou: float = 0.45,
    ):
        self.confidence = confidence
        self.iou = iou

        # Fetch model from Hugging Face Hub (handles caching automatically)
        print("[INFO] Fetching YOLO model from Hugging Face...")
        downloaded_path = hf_hub_download(
            repo_id="vineetsarpal/yolov11n-car-damage",
            filename="best.pt",
            force_download=False
        )
        self.model_path = Path(downloaded_path)

        print(f"Loading YOLO damage model: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        print("YOLO damage model loaded successfully.")

    def _prepare_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Failed to load image from {image}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            return image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

    def _calculate_box_severity(self, box_area: float, img_area: float, conf: float) -> Dict[str, Any]:
        area_ratio = box_area / img_area
        score = min(100, int((area_ratio * 70 + conf * 30) * 100))
        
        if score < 30:
            level = "Minor"
        elif score < 65:
            level = "Moderate"
        else:
            level = "Severe"

        return {"score": score, "level": level}

    def detect(self, image: Union[str, Path, Image.Image, np.ndarray]) -> List[Dict[str, Any]]:
        img_np = self._prepare_image(image)
        results = self.model.predict(
            source=img_np,
            conf=self.confidence,
            iou=self.iou,
            verbose=False
        )[0]

        img_height, img_width = img_np.shape[:2]
        img_area = img_height * img_width
        detections = []

        if results.boxes is not None:
            for box in results.boxes:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                class_name = self.model.names[cls_id]

                box_area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                severity = self._calculate_box_severity(box_area, img_area, conf)

                detections.append({
                    "class": class_name,
                    "confidence": round(conf, 4),
                    "box": [round(x, 2) for x in xyxy],
                    "severity": severity["level"],
                    "severity_score": severity["score"]
                })

        return detections

    def _calculate_damage_coverage(self, detections: List[Dict[str, Any]], img_shape: tuple) -> float:
        if not detections:
            return 0.0

        mask = np.zeros(img_shape[:2], dtype=np.uint8)
        for det in detections:
            x1, y1, x2, y2 = map(int, det["box"])
            mask[y1:y2, x1:x2] = 1

        total_damage_pixels = np.sum(mask)
        total_pixels = img_shape[0] * img_shape[1]
        return round((total_damage_pixels / total_pixels) * 100, 2)

    def predict(self, image: Union[str, Path, Image.Image, np.ndarray]) -> Dict[str, Any]:
        img_np = self._prepare_image(image)
        detections = self.detect(img_np)
        coverage = self._calculate_damage_coverage(detections, img_np.shape)

        if not detections:
            overall_score = 0
            overall_level = "None"
        else:
            max_score = max(d["severity_score"] for d in detections)
            overall_score = min(100, int(max_score * 0.7 + coverage * 0.3))
            
            if overall_score < 30:
                overall_level = "Minor"
            elif overall_score < 65:
                overall_level = "Moderate"
            else:
                overall_level = "Severe"

        return {
            "damage_count": len(detections),
            "damage_coverage_percentage": coverage,
            "overall_severity": overall_level,
            "overall_severity_score": overall_score,
            "detections": detections
        }

    def annotate(self, image: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
        img_np = self._prepare_image(image)
        # Convert RGB to BGR for OpenCV drawing operations
        annotated = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        detections = self.detect(img_np)

        for det in detections:
            x1, y1, x2, y2 = map(int, det["box"])
            label = f"{det['class']} ({det['severity']})"
            
            # Draw Bounding Box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Draw Label Background & Text
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - 20), (x1 + w, y1), (0, 0, 255), -1)
            cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return annotated