import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

import optimisations  # from part 2

# Configure logging
logging.basicConfig(level=logging.INFO)


def get_dataloader(batch_size=32):
    # Get project root directory to ensure data is downloaded to correct location
    import os
    try:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    except NameError:
        # fallback for Jupyter notebooks
        project_root = os.getcwd()
    
    data_root = os.path.join(project_root, "data")
    
    # ImageNet standard transforms - these match the preprocessing used during training
    transform = transforms.Compose([
        transforms.Resize(256),  # Resize to 256x256
        transforms.CenterCrop(224),  # Center crop to 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    # Check if we have the downloaded ImageNet validation data first
    imagenet_val_path = os.path.join(data_root, "imagenet", "val")
    
    if os.path.exists(imagenet_val_path) and os.listdir(imagenet_val_path):
        # Use the downloaded ImageNet validation data with ImageFolder
        try:
            testset = torchvision.datasets.ImageFolder(
                root=imagenet_val_path, transform=transform
            )
            logging.info(f"✅ Using downloaded ImageNet validation data with {len(testset)} samples and {len(testset.classes)} classes")
            return DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
        except Exception as e:
            logging.error(f"Failed to load ImageNet validation folder: {e}")
    
    # Try the official torchvision ImageNet dataset (requires devkit)
    try:
        testset = torchvision.datasets.ImageNet(
            root=data_root, split='val', transform=transform
        )
        logging.info(f"Using official ImageNet validation dataset with {len(testset)} samples")
    except (RuntimeError, FileNotFoundError) as e:
        logging.warning(f"Official ImageNet dataset not found: {e}")
        logging.info("Falling back to alternative ImageNet data sources")
        
        # Fallback options
        try:
            # Try using FakeData as a better fallback than pure synthetic
            logging.info("Using FakeData as ImageNet substitute for testing")
            testset = torchvision.datasets.FakeData(
                size=1000,  # 1000 samples
                image_size=(3, 224, 224),
                num_classes=1000,  # ImageNet has 1000 classes
                transform=transform
            )
            logging.info(f"Using FakeData dataset with {len(testset)} samples")
        except Exception as fallback_error:
            logging.error(f"FakeData fallback failed: {fallback_error}")
            # Final fallback: Create synthetic dataset
            logging.warning("Creating synthetic ImageNet-like dataset for testing")
            testset = create_synthetic_imagenet_dataset(data_root, transform)
    
    return DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)


def create_synthetic_imagenet_dataset(data_root, transform):
    """
    Create a synthetic ImageNet-like dataset for testing when real ImageNet is not available.
    This creates random images with 1000 classes to match ImageNet structure.
    """
    import numpy as np
    from PIL import Image
    import os
    
    class SyntheticImageNetDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples=1000, num_classes=1000, transform=None):
            self.num_samples = num_samples
            self.num_classes = num_classes
            self.transform = transform
            
        def __len__(self):
            return self.num_samples
            
        def __getitem__(self, idx):
            # Create a random RGB image (224x224x3)
            np.random.seed(idx)  # For reproducibility
            image_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            image = Image.fromarray(image_array, 'RGB')
            
            # Random class label
            label = np.random.randint(0, self.num_classes)
            
            if self.transform:
                image = self.transform(image)
                
            return image, label
    
    logging.info("Created synthetic ImageNet dataset with 1000 samples")
    return SyntheticImageNetDataset(num_samples=1000, transform=transform)


def accuracy(output, target, topk=(1,)):
    """Compute top-k accuracy"""
    maxk = max(topk)
    batch_size = target.size(0)
    
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def evaluate_model(model, dataloader, device, use_amp=False):
    """Evaluate model on entire dataset"""
    model.eval()
    model.to(device)

    top1_acc, top5_acc, total = 0.0, 0.0, 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)

            if use_amp and device == "cuda":
                with torch.amp.autocast("cuda"):
                    outputs = model(images)
            else:
                outputs = model(images)

            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            top1_acc += acc1.item() * images.size(0)
            top5_acc += acc5.item() * images.size(0)
            total += images.size(0)

    return top1_acc / total, top5_acc / total


def evaluate_model_subset(model, device, use_amp=False, num_batches=10, batch_size=32):
    """
    Evaluate model on a subset of ImageNet validation data for benchmarking purposes.
    This is more efficient than evaluating on the entire dataset.
    
    Expected accuracy ranges:
    - With real ImageNet validation data: DenseNet121 should achieve ~74% Top-1, ~92% Top-5
    - With FakeData/synthetic data: Random accuracy ~0.1% Top-1, ~0.5% Top-5
    
    Args:
        model: The model to evaluate (should be ImageNet pretrained)
        device: Device to run evaluation on
        use_amp: Whether to use automatic mixed precision
        num_batches: Number of batches to evaluate (default: 10)
        batch_size: Batch size for evaluation
    
    Returns:
        tuple: (top1_accuracy, top5_accuracy)
    """
    try:
        # Get dataloader with specified batch size
        dataloader = get_dataloader(batch_size=batch_size)
        
        model.eval()
        model.to(device)

        top1_acc, top5_acc, total = 0.0, 0.0, 0
        batch_count = 0

        with torch.no_grad():
            for images, labels in dataloader:
                if batch_count >= num_batches:
                    break
                    
                images, labels = images.to(device), labels.to(device)

                if use_amp and device == "cuda":
                    with torch.amp.autocast("cuda"):
                        outputs = model(images)
                else:
                    outputs = model(images)

                acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
                top1_acc += acc1.item() * images.size(0)
                top5_acc += acc5.item() * images.size(0)
                total += images.size(0)
                batch_count += 1

        if total == 0:
            logging.warning("No samples evaluated for accuracy calculation")
            return 0.0, 0.0
            
        return top1_acc / total, top5_acc / total
        
    except Exception as e:
        logging.error(f"Error during accuracy evaluation: {e}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        return 0.0, 0.0


def main():
    """Test accuracy evaluation with ImageNet validation data"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Models to test
    sample_input = torch.randn(1, 3, 224, 224).to(device)
    models = {
        "baseline": optimisations.get_model("baseline", device, sample_input),
        "amp": optimisations.get_model("amp", device, sample_input),
    }

    # Test with a subset for quick evaluation
    results = {}
    for name, model in models.items():
        print(f"\n[INFO] Evaluating {name} model on ImageNet validation subset...")
        use_amp = True if name == "amp" else False
        acc1, acc5 = evaluate_model_subset(
            model, device, use_amp=use_amp, num_batches=5, batch_size=16
        )
        results[name] = {"Top-1": acc1, "Top-5": acc5}
        print(f"{name} -> Top-1: {acc1:.2f}%, Top-5: {acc5:.2f}%")

    print("\n✅ Final Results:")
    for name, metrics in results.items():
        print(f"{name}: {metrics}")


if __name__ == "__main__":
    main()
