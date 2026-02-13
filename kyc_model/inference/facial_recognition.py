import base64
from functools import lru_cache
from io import BytesIO
from typing import Dict

import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image


def _decode_base64_image(image_base64: str) -> Image.Image:
    image_bytes = base64.b64decode(image_base64)
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=1)
def _get_mtcnn() -> MTCNN:
    return MTCNN(image_size=160, margin=20, keep_all=False, device=_get_device())


@lru_cache(maxsize=1)
def _get_facenet() -> InceptionResnetV1:
    return InceptionResnetV1(pretrained="vggface2").to(_get_device()).eval()


def _get_embedding(image_base64: str) -> torch.Tensor:
    image = _decode_base64_image(image_base64)
    mtcnn = _get_mtcnn()
    face_tensor = mtcnn(image)
    if face_tensor is None:
        raise ValueError("No face detected in the provided image")
    face_tensor = face_tensor.to(_get_device())
    facenet = _get_facenet()
    with torch.no_grad():
        embedding = facenet(face_tensor.unsqueeze(0))
    return embedding.squeeze(0)


def compare_faces_from_base64(
    document_image_base64: str,
    selfie_image_base64: str,
) -> float:
    embedding_document = _get_embedding(document_image_base64)
    embedding_selfie = _get_embedding(selfie_image_base64)
    distance = torch.nn.functional.cosine_similarity(
        embedding_document.unsqueeze(0), embedding_selfie.unsqueeze(0)
    ).item()
    return distance



def calculate_score_from_facial_recognition(document_image_base64: str, selfie_image_base64: str) -> float:
    """Return a 0-10 confidence score by comparing document and selfie embeddings."""
    similarity = compare_faces_from_base64(document_image_base64, selfie_image_base64)
    distance = 1.0 - similarity
    distance = max(0.0, min(distance, 2.0))
    score = (1.0 - (distance / 2.0)) * 10.0
    return max(0.0, min(score, 10.0))