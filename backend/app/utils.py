import io
import torch
from PIL import Image, ImageFile
from torchvision import transforms
import imagehash
from sqlalchemy.orm import Session
from app.database import Claim

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Validation/Test Transformation pipeline matching model training
eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = eval_transform(image)
    return tensor.unsqueeze(0)  # Add batch dimension

def compute_phash_from_bytes(image_bytes: bytes) -> str:
    """Generates a perceptual hash (pHash) string from raw image bytes."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    hash_obj = imagehash.phash(image)
    return str(hash_obj)

def check_duplicate_claim(db: Session, user_id: int, new_phash_str: str, threshold: int = 5) -> tuple[bool, str]:
    """
    Compares new pHash against past claim hashes for a given user.
    Handles PNG to JPEG conversions, compression, and scaling edits.
    
    Returns (is_duplicate: bool, detail_message: str)
    """
    # Query all previous claims for this exact user
    previous_claims = db.query(Claim).filter(Claim.user_id == user_id).all()
    new_hash = imagehash.hex_to_hash(new_phash_str)

    for claim in previous_claims:
        existing_hash = imagehash.hex_to_hash(claim.phash)
        distance = new_hash - existing_hash  # Computes Hamming Distance

        # Distance <= threshold (e.g. 5) flags visually identical or edited duplicates
        if distance <= threshold:
            formatted_date = claim.created_at.strftime("%Y-%m-%d %H:%M:%S")
            return True, f"Duplicate image detected! Matches claim #{claim.id} submitted on {formatted_date} (Hamming Distance: {distance})."

    return False, ""