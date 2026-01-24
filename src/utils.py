import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from kornia.losses import dice_loss, focal_loss, hausdorff as hausdorff_distance_loss
from torch import nn
from torch.utils.data import Dataset
from torchgeo.datasets import IntersectionDataset, RasterDataset, VectorDataset
from tqdm import tqdm


class RAMPImageDataset(RasterDataset):
    filename_glob = "*.tif"
    is_image = True
    all_bands = ("R", "G", "B")
    rgb_bands = ("R", "G", "B")


class RAMPMaskDataset(VectorDataset):
    filename_glob = "*.geojson"
    
    def __init__(self, paths, crs=None, res=0.3, **kwargs):
        super().__init__(
            paths=paths,
            crs=crs,
            res=res,
            task="instance_segmentation",
            **kwargs
        )


def get_ramp_dataset(root: Path, regions: list[str], res: float = 0.3):
    image_paths = []
    label_paths = []
    
    for region in regions:
        region_path = root / region
        img_path = region_path / "source"
        lbl_path = region_path / "labels"
        
        if img_path.exists() and lbl_path.exists():
            image_paths.append(img_path)
            label_paths.append(lbl_path)
    
    if not image_paths:
        raise ValueError(f"No valid regions found in {root}")
    
    images = RAMPImageDataset(paths=image_paths)
    masks = RAMPMaskDataset(paths=label_paths, crs=images.crs, res=res)
    
    return IntersectionDataset(images, masks)


def get_all_ramp_regions(root: Path) -> list[str]:
    regions = [d.name for d in root.iterdir() if d.is_dir() and d.name.startswith("ramp_")]
    if not regions:
        raise ValueError(f"No RAMP regions found in {root}")
    return sorted(regions)


def split_regions(regions: list[str], val_ratio: float = 0.2, seed: int = 42):
    """Split regions into train and validation sets"""
    rng = random.Random(seed)
    shuffled = regions.copy()
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - val_ratio))
    return shuffled[:split_idx], shuffled[split_idx:]


def compute_num_queries(dataset: Dataset, num_samples: int = 100, percentile: float = 95.0) -> int:
    """Compute optimal num_queries by sampling dataset
    
    Args:
        dataset: TorchGeo dataset
        num_samples: Number of samples to check
        percentile: Percentile to use (handles outliers)
    
    Returns:
        Recommended num_queries value
    """
    print(f"Computing optimal num_queries from {num_samples} samples...")
    
    instance_counts = []
    num_samples = min(num_samples, len(dataset))
    
    indices = random.sample(range(len(dataset)), num_samples)
    
    for idx in tqdm(indices, desc="Sampling dataset"):
        try:
            sample = dataset[idx]
            mask = sample["mask"]
            
            if mask.ndim == 2:
                num_instances = 1 if mask.sum() > 0 else 0
            else:
                num_instances = sum(1 for i in range(mask.shape[0]) if mask[i].sum() > 10)
            
            instance_counts.append(num_instances)
        except Exception:
            continue
    
    if not instance_counts:
        print("Warning: Could not compute num_queries, using default 100")
        return 100
    
    max_instances = int(np.percentile(instance_counts, percentile))
    recommended = max(max_instances + 20, 50)
    
    print(f"Instance statistics:")
    print(f"  Mean: {np.mean(instance_counts):.1f}")
    print(f"  Median: {np.median(instance_counts):.1f}")
    print(f"  Max: {np.max(instance_counts)}")
    print(f"  {percentile}th percentile: {max_instances}")
    print(f"  Recommended num_queries: {recommended}")
    
    return recommended


def collate_fn_mask2former(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate function for Mask2Former"""
    images = torch.stack([s["image"].float() / 255.0 for s in batch])
    
    targets = []
    for sample in batch:
        mask = sample["mask"]
        
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        
        instances = []
        class_labels = []
        
        for i in range(mask.shape[0]):
            instance_mask = mask[i]
            if instance_mask.sum() > 10:
                instances.append(instance_mask)
                class_labels.append(1)
        
        if len(instances) > 0:
            masks_tensor = torch.stack(instances).bool()
            labels_tensor = torch.tensor(class_labels, dtype=torch.long)
        else:
            masks_tensor = torch.zeros((1, *mask.shape[-2:]), dtype=torch.bool)
            labels_tensor = torch.zeros(1, dtype=torch.long)
        
        targets.append({
            "masks": masks_tensor,
            "class_labels": labels_tensor
        })
    
    return {"pixel_values": images, "mask_labels": targets}


class CombinedLoss(nn.Module):
    """Combined loss: CE + Dice + Focal + Hausdorff"""
    
    def __init__(self, ce_weight=2.0, dice_weight=5.0, focal_weight=5.0, hausdorff_weight=0.1):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.hausdorff_weight = hausdorff_weight
    
    def forward(self, pred_masks, pred_logits, gt_masks, gt_labels):
        B, Q = pred_logits.shape[:2]
        N = gt_masks.shape[1]
        
        loss = 0.0
        
        if pred_logits is not None and self.ce_weight > 0:
            pred_classes = pred_logits.flatten(0, 1)
            target_classes = gt_labels.flatten()
            
            if target_classes.max() < pred_classes.shape[-1]:
                ce_loss = F.cross_entropy(pred_classes, target_classes, reduction="mean")
                loss += self.ce_weight * ce_loss
        
        pred_masks_flat = pred_masks.flatten(0, 1)[:N*B]
        gt_masks_flat = gt_masks.flatten(0, 1)
        
        if pred_masks_flat.shape[0] > 0 and gt_masks_flat.shape[0] > 0:
            pred_sigmoid = torch.sigmoid(pred_masks_flat)
            gt_float = gt_masks_flat.float()
            
            if self.dice_weight > 0:
                dice = dice_loss(pred_sigmoid, gt_float)
                loss += self.dice_weight * dice
            
            if self.focal_weight > 0:
                focal = focal_loss(pred_masks_flat, gt_masks_flat.long(), alpha=0.25, gamma=2.0, reduction="mean")
                loss += self.focal_weight * focal
            
            if self.hausdorff_weight > 0:
                try:
                    hd = hausdorff_distance_loss(pred_sigmoid, gt_float)
                    loss += self.hausdorff_weight * hd
                except Exception:
                    pass
        
        return loss


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.should_stop


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(model, optimizer, epoch, loss, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, path)


def load_checkpoint(model, optimizer, path: Path, device="cuda"):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("epoch", 0), checkpoint.get("loss", float("inf"))