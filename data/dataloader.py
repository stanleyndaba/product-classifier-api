"""
Data loading pipeline for product classification dataset.
Handles loading, preprocessing, and augmentation of product images.
"""

import os
import yaml
from typing import Tuple, Optional, List, Dict, Any
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np


class ProductDataset(Dataset):
    """Custom dataset for product classification."""
    
    def __init__(
        self,
        data_dir: str,
        transform: Optional[transforms.Compose] = None,
        class_to_idx: Optional[Dict[str, int]] = None
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.class_to_idx = class_to_idx or self._build_class_to_idx()
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.samples = self._load_samples()
    
    def _build_class_to_idx(self) -> Dict[str, int]:
        """Build class to index mapping from directory structure."""
        classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        return {cls_name: idx for idx, cls_name in enumerate(classes)}
    
    def _load_samples(self) -> List[Tuple[Path, int]]:
        """Load all image samples with their labels."""
        samples = []
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                continue
                
            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in valid_extensions:
                    samples.append((img_path, class_idx))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        try:
            # Load image using PIL
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            if self.transform:
                image = self.transform(Image.new('RGB', (224, 224), (0, 0, 0)))
            else:
                image = torch.zeros(3, 224, 224)
            return image, label


class DataLoaderFactory:
    """Factory class for creating data loaders with different configurations."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.train_transforms = self._get_train_transforms()
        self.val_transforms = self._get_val_transforms()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config.get('training', {})
    
    def _get_train_transforms(self) -> transforms.Compose:
        """Get training data transforms with augmentation."""
        aug_config = self.config.get('augmentation', {})
        
        transform_list = [
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomRotation(aug_config.get('rotation', 15)),
            transforms.RandomHorizontalFlip(p=aug_config.get('horizontal_flip', 0.5)),
            transforms.ColorJitter(
                brightness=aug_config.get('brightness', 0.2),
                contrast=aug_config.get('contrast', 0.2)
            ),
            transforms.ToTensor(),
        ]
        
        # Add normalization if specified
        normalize_config = aug_config.get('normalize', {})
        if normalize_config:
            transform_list.append(
                transforms.Normalize(
                    mean=normalize_config.get('mean', [0.485, 0.456, 0.406]),
                    std=normalize_config.get('std', [0.229, 0.224, 0.225])
                )
            )
        
        return transforms.Compose(transform_list)
    
    def _get_val_transforms(self) -> transforms.Compose:
        """Get validation/test data transforms without augmentation."""
        aug_config = self.config.get('augmentation', {})
        
        transform_list = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
        
        # Add normalization if specified
        normalize_config = aug_config.get('normalize', {})
        if normalize_config:
            transform_list.append(
                transforms.Normalize(
                    mean=normalize_config.get('mean', [0.485, 0.456, 0.406]),
                    std=normalize_config.get('std', [0.229, 0.224, 0.225])
                )
            )
        
        return transforms.Compose(transform_list)
    
    def create_dataloaders(
        self,
        train_dir: str,
        val_dir: Optional[str] = None,
        test_dir: Optional[str] = None,
        val_split: float = 0.2
    ) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        """
        Create train, validation, and test data loaders.
        
        Args:
            train_dir: Path to training data directory
            val_dir: Path to validation data directory (optional)
            test_dir: Path to test data directory (optional)
            val_split: Validation split ratio if val_dir is not provided
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        data_config = self.config.get('data', {})
        batch_size = data_config.get('batch_size', 32)
        num_workers = data_config.get('num_workers', 4)
        
        # Create training dataset
        train_dataset = ProductDataset(train_dir, transform=self.train_transforms)
        
        # Create validation dataset
        if val_dir and os.path.exists(val_dir):
            val_dataset = ProductDataset(
                val_dir, 
                transform=self.val_transforms,
                class_to_idx=train_dataset.class_to_idx
            )
        else:
            # Split training dataset for validation
            train_size = int((1 - val_split) * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
            
            # Update validation dataset transform
            val_dataset.dataset.transform = self.val_transforms
        
        # Create test dataset
        test_loader = None
        if test_dir and os.path.exists(test_dir):
            test_dataset = ProductDataset(
                test_dir,
                transform=self.val_transforms,
                class_to_idx=train_dataset.class_to_idx
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available()
            )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=data_config.get('shuffle', True),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        return train_loader, val_loader, test_loader
    
    def create_inference_dataloader(
        self,
        images: List[np.ndarray],
        batch_size: int = 1
    ) -> DataLoader:
        """
        Create a data loader for inference from a list of numpy arrays.
        
        Args:
            images: List of image arrays (H, W, C)
            batch_size: Batch size for inference
            
        Returns:
            DataLoader for inference
        """
        class InferenceDataset(Dataset):
            def __init__(self, images: List[np.ndarray], transform: transforms.Compose):
                self.images = images
                self.transform = transform
            
            def __len__(self):
                return len(self.images)
            
            def __getitem__(self, idx):
                img_array = self.images[idx]
                # Convert numpy array to PIL Image
                if img_array.dtype != np.uint8:
                    img_array = (img_array * 255).astype(np.uint8)
                image = Image.fromarray(img_array)
                return self.transform(image)
        
        dataset = InferenceDataset(images, self.val_transforms)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0  # No multiprocessing for inference
        )


def create_sample_data_structure(base_dir: str = "data/raw"):
    """Create sample directory structure for demonstration purposes."""
    categories = [
        "electronics", "clothing", "furniture", "books", "toys",
        "sports", "beauty", "home_garden", "automotive", "health",
        "food", "jewelry"
    ]
    
    base_path = Path(base_dir)
    
    for split in ["train", "val", "test"]:
        split_dir = base_path / split
        split_dir.mkdir(parents=True, exist_ok=True)
        
        for category in categories:
            category_dir = split_dir / category
            category_dir.mkdir(exist_ok=True)
            
            # Create a placeholder README
            readme_path = category_dir / "README.md"
            readme_path.write_text(f"# {category.title()} Images\n\nPlace {category} product images in this directory.\n")


if __name__ == "__main__":
    # Create sample data structure
    create_sample_data_structure()
    print("Sample data structure created!")
    
    # Example usage
    try:
        factory = DataLoaderFactory()
        train_loader, val_loader, test_loader = factory.create_dataloaders(
            train_dir="data/raw/train",
            val_dir="data/raw/val",
            test_dir="data/raw/test"
        )
        print(f"Created data loaders:")
        print(f"  Train: {len(train_loader)} batches")
        print(f"  Validation: {len(val_loader)} batches")
        if test_loader:
            print(f"  Test: {len(test_loader)} batches")
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        print("Make sure to add image data to the data/raw directories")