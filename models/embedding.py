import torch
import insightface
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from PIL import Image


class FaceEmbedder:
    def __init__(self, model_name='buffalo_l', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device_str = device
        self.device = torch.device(device)
        self.ctx_id = 0 if device == 'cuda' else -1

        # Initialize InsightFace
        self.app = FaceAnalysis(name=model_name,
                                providers=['CUDAExecutionProvider' if device == 'cuda' else 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=self.ctx_id, det_size=(640, 640))

        print(f"Face embedder initialized on {device}")

    def get_embedding(self, img):
        """Extract embedding from a PIL or numpy image"""
        if isinstance(img, Image.Image):
            img = np.array(img)

        if img.shape[2] == 4:  # If RGBA, convert to RGB
            img = img[:, :, :3]

        # BGR to RGB if needed
        if img.shape[2] == 3 and isinstance(img, np.ndarray):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Get face embedding
        faces = self.app.get(img)

        if len(faces) == 0:
            return None

        # Use the largest face by default
        face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        embedding = face.embedding

        # Convert to tensor
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32, device=self.device)

        return embedding_tensor, face.bbox

    def get_aligned_face(self, img, size=128):
        """Extract aligned face from image"""
        if isinstance(img, Image.Image):
            img = np.array(img)

        if img.shape[2] == 4:
            img = img[:, :, :3]

        # BGR to RGB if needed
        if img.shape[2] == 3 and isinstance(img, np.ndarray):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Get face with landmarks
        faces = self.app.get(img)

        if len(faces) == 0:
            return None

        # Use the largest face by default
        face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))

        # Get aligned face using landmarks
        aligned = self.app.get_input(img, face)

        # Resize to target size
        aligned = cv2.resize(aligned, (size, size))

        # Convert back to RGB
        aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)

        return aligned, face.embedding