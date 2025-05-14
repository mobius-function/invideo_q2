import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import random
import json
from tqdm import tqdm


class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None, embedding_dir=None, load_on_memory=False, img_size=128):
        """
        Args:
            root_dir (str): Directory with all face images
            transform (callable, optional): Transform to be applied on images
            embedding_dir (str, optional): Directory to save/load pre-computed embeddings
            load_on_memory (bool): Whether to load all images in memory
            img_size (int): Size of images to resize to
        """
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        # Get list of image files
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Load or compute embeddings
        self.embeddings = {}
        self.cached_images = {}

        if embedding_dir and os.path.exists(embedding_dir):
            # Load pre-computed embeddings
            for f in os.listdir(embedding_dir):
                if f.endswith('.npy'):
                    img_name = f.replace('.npy', '')
                    if img_name + '.jpg' in self.image_files or img_name + '.png' in self.image_files:
                        self.embeddings[img_name] = np.load(os.path.join(embedding_dir, f))

            # Only keep images that have embeddings
            self.image_files = [f for f in self.image_files
                                if os.path.splitext(f)[0] in self.embeddings]

        # Load images to memory if requested
        if load_on_memory:
            for img_file in tqdm(self.image_files, desc="Loading images"):
                img_path = os.path.join(self.root_dir, img_file)
                self.cached_images[img_file] = Image.open(img_path).convert('RGB')

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]

        # Get image from cache or load from disk
        if img_name in self.cached_images:
            image = self.cached_images[img_name]
        else:
            img_path = os.path.join(self.root_dir, img_name)
            image = Image.open(img_path).convert('RGB')

        # Apply transform
        if self.transform:
            image = self.transform(image)

        # Get embedding
        embedding_key = os.path.splitext(img_name)[0]
        if embedding_key in self.embeddings:
            embedding = torch.tensor(self.embeddings[embedding_key], dtype=torch.float32)
        else:
            # Return zero embedding if not found
            embedding = torch.zeros(512, dtype=torch.float32)

        return embedding, image


def create_dataloaders(data_dir, embedding_dir=None, batch_size=32, img_size=128, num_workers=4):
    """Create train and validation dataloaders"""

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Create dataset
    dataset = FaceDataset(data_dir, transform, embedding_dir, img_size=img_size)

    # Split into train and validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader