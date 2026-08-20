import io
import os
import base64
import uuid
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, EmailStr
from PIL import Image
import torch
from torchvision import transforms
from sqlalchemy.orm import Session

from app.model import model as convnext_model
from app.damage_detector import VehicleDamageDetector
from app.database import get_db, User, Claim
from app.utils import preprocess_image, compute_phash_from_bytes, check_duplicate_claim
from app.gradcam import GradCAM
from app.pdf_generator import generate_claim_report_pdf

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ARTIFACTS_DIR = Path("artifacts")
REPORTS_DIR = ARTIFACTS_DIR / "reports"
TEMP_DIR = ARTIFACTS_DIR / "temp"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

class AuthCredentials(BaseModel):
    email: EmailStr
    password: str

def get_current_user(credentials: AuthCredentials, db: Session = Depends(get_db)) -> User:
    user = db.query(User).filter(User.email == credentials.email).first()
    if not user:
        user = User(email=credentials.email, hashed_password=credentials.password)
        db.add(user)
        db.commit()
        db.refresh(user)
    elif user.hashed_password != credentials.password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Invalid email or password."
        )
    return user

@app.post("/api/v1/auth/signup")
async def signup(credentials: AuthCredentials, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.email == credentials.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="User with this email already exists."
        )
    
    new_user = User(email=credentials.email, hashed_password=credentials.password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return {
        "message": "Account created successfully",
        "user_id": new_user.id,
        "email": new_user.email
    }

@app.post("/api/v1/auth/login")
async def login(credentials: AuthCredentials, db: Session = Depends(get_db)):
    user = get_current_user(credentials, db)
    return {
        "message": "Login successful",
        "user_id": user.id,
        "token": f"fake-jwt-token-{user.email}",
        "email": user.email
    }

# Reuse the model loaded by app.model instead of allocating a second copy.
for param in convnext_model.parameters():
    param.requires_grad = True

convnext_model.eval()

FRAUD_THRESHOLD = 0.30
detector = VehicleDamageDetector(model_path="models/best.pt", confidence=0.25, iou=0.45)


def image_to_base64(img_obj, fallback_pil=None) -> str:
    """Safely converts PIL Image or NumPy Array (BGR/RGB) to a valid JPEG Data URI string."""
    try:
        target_img = img_obj if img_obj is not None else fallback_pil
        if target_img is None:
            return ""

        # Case 1: PIL Image
        if isinstance(target_img, Image.Image):
            buf = io.BytesIO()
            target_img.convert("RGB").save(buf, format="JPEG")
            b64_data = base64.b64encode(buf.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{b64_data}"

        # Case 2: OpenCV / NumPy Array
        if isinstance(target_img, np.ndarray):
            if target_img.dtype != np.uint8:
                target_img = target_img.astype(np.uint8)
            
            # Encode via OpenCV
            success, buffer = cv2.imencode(".jpg", target_img)
            if success:
                b64_data = base64.b64encode(buffer.tobytes()).decode("utf-8")
                return f"data:image/jpeg;base64,{b64_data}"

        # Fallback to converting original PIL if present
        if fallback_pil is not None:
            buf = io.BytesIO()
            fallback_pil.convert("RGB").save(buf, format="JPEG")
            b64_data = base64.b64encode(buf.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{b64_data}"

    except Exception as e:
        print(f"[Image Conversion Error]: {e}")

    return ""


def generate_gradcam_overlay(model, tensor_img, original_pil_img, is_fraud_class=1):
    """Generates Grad-CAM overlay heatmap targeting the fraud class."""
    try:
        if hasattr(model, 'model') and hasattr(model.model, 'features'):
            target_layer = model.model.features[-1][-1]
        elif hasattr(model, 'features'):
            target_layer = model.features[-1][-1]
        else:
            target_layer = model.classifier[0]

        gradcam = GradCAM(model, target_layer)
        heatmap, _ = gradcam.generate(tensor_img, is_fraud_class)
        gradcam.remove_hooks()

        original_array = np.array(original_pil_img)
        heatmap_resized = cv2.resize(heatmap, (original_array.shape[1], original_array.shape[0]))
        heatmap_uint8 = np.uint8(heatmap_resized * 255)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        overlay = cv2.addWeighted(original_array, 0.60, heatmap_color, 0.40, 0)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

        success, buffer = cv2.imencode(".jpg", overlay_bgr)
        if not success:
            return None, None, None

        raw_bytes = buffer.tobytes()
        b64_str = base64.b64encode(raw_bytes).decode("utf-8")

        return raw_bytes, f"data:image/jpeg;base64,{b64_str}", overlay_bgr
    except Exception as err:
        print(f"[Grad-CAM Error] Generation failed: {err}")
        return None, None, None


@app.post("/api/v1/predict")
async def predict(
    email: str = Form(default="default@example.com"), 
    file: UploadFile = File(...), 
    db: Session = Depends(get_db)
):
    try:
        user = db.query(User).filter(User.email == email).first()
        if not user:
            user = User(email=email, hashed_password="defaultpassword")
            db.add(user)
            db.commit()
            db.refresh(user)

        contents = await file.read()
        phash_str = compute_phash_from_bytes(contents)
        is_duplicate, duplicate_detail = check_duplicate_claim(db, user.id, phash_str)
        if is_duplicate:
            raise HTTPException(status_code=409, detail=duplicate_detail)

        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        tensor = preprocess_image(contents)

        # ConvNeXt model evaluation
        with torch.no_grad():
            output = convnext_model(tensor)
            prob = torch.sigmoid(output).item()
            
        fraud_score = prob * 100
        is_fraud = prob >= FRAUD_THRESHOLD

        # Grad-CAM heatmap generation
        _, gradcam_b64, _ = generate_gradcam_overlay(convnext_model, tensor, pil_img, is_fraud_class=1)

        # YOLO Damage Assessment (Runs for both Fraud & Non-Fraud)
        damage_data = detector.predict(pil_img)
        annotated_img = detector.annotate(pil_img)

        # Robust Conversion to Base64 JPEG
        annotated_b64 = image_to_base64(annotated_img, fallback_pil=pil_img)

        # Save record to Database
        new_claim = Claim(
            user_id=user.id,
            phash=phash_str,
            image_name=file.filename,
            fraud_score=round(fraud_score, 2)
        )
        db.add(new_claim)
        db.commit()

        # Returning multiple field aliases to ensure frontend state binding works
        return {
            "assessment": "FLAGGED FOR FRAUD" if is_fraud else "Non-Fraud",
            "fraud_risk_score": round(fraud_score, 2),
            "is_fraud": is_fraud,
            "gradcam_image": gradcam_b64,
            "damage_analysis": damage_data,
            "annotated_image": annotated_b64,
            "yolo_image": annotated_b64,
            "damage_localization_image": annotated_b64,
            "annotated_image_url": annotated_b64,
            "damage_image": annotated_b64
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/analyze-and-report")
async def analyze_and_report(
    email: str = Form(default="default@example.com"), 
    file: UploadFile = File(...), 
    db: Session = Depends(get_db)
):
    try:
        user = db.query(User).filter(User.email == email).first()
        if not user:
            user = User(email=email, hashed_password="defaultpassword")
            db.add(user)
            db.commit()
            db.refresh(user)

        customer_id = f"CUST-{user.id:04d}" if hasattr(user, 'id') else "CUST-0001"
        contents = await file.read()
        unique_id = uuid.uuid4().hex[:6].upper()
        processing_id = f"PRC-{unique_id}"
        execution_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

        orig_img_path = TEMP_DIR / f"{processing_id}_orig.jpg"
        with open(orig_img_path, "wb") as f:
            f.write(contents)

        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        tensor = preprocess_image(contents)

        # Model Inference
        with torch.no_grad():
            output = convnext_model(tensor)
            prob = torch.sigmoid(output).item()

        fraud_score = prob * 100
        is_fraud = prob >= FRAUD_THRESHOLD
        fraud_label = "FLAGGED FOR FRAUD" if is_fraud else "Non-Fraud"
        
        model_confidence = (prob if is_fraud else (1.0 - prob)) * 100

        # Grad-CAM Explainability
        gradcam_path = TEMP_DIR / f"{processing_id}_gradcam.jpg"
        _, _, gradcam_bgr = generate_gradcam_overlay(convnext_model, tensor, pil_img, is_fraud_class=1)
        if gradcam_bgr is not None:
            cv2.imwrite(str(gradcam_path), gradcam_bgr)

        # YOLO Damage Detection (Runs for both Fraud & Non-Fraud)
        damage_data = detector.predict(pil_img)
        annotated_img = detector.annotate(pil_img)
        annotated_path = TEMP_DIR / f"{processing_id}_yolo.jpg"
        
        if annotated_img is not None:
            if isinstance(annotated_img, np.ndarray):
                cv2.imwrite(str(annotated_path), annotated_img)
            elif isinstance(annotated_img, Image.Image):
                annotated_img.save(str(annotated_path))
        else:
            pil_img.save(str(annotated_path))

        # Build PDF Report
        pdf_output_path = REPORTS_DIR / f"claim_report_{processing_id}.pdf"
        generate_claim_report_pdf(
            output_path=pdf_output_path,
            customer_id=customer_id,
            customer_email=user.email,
            execution_time=execution_time,
            original_image_path=str(orig_img_path),
            annotated_image_path=str(annotated_path),
            gradcam_path=str(gradcam_path),
            fraud_label=fraud_label,
            fraud_probability=fraud_score,
            model_confidence=model_confidence,
            damage_result=damage_data,
        )

        with open(pdf_output_path, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()

        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=claim_report_{processing_id}.pdf"}
        )
    except Exception as e:
        print(f"[API Error] /analyze-and-report failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))