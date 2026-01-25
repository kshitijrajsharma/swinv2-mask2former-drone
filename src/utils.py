import random
from pathlib import Path
from typing import Any
import json 

import numpy as np
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from kornia.losses import HausdorffERLoss
from torch import nn
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
    from rasterio.crs import CRS
    
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
    
    target_crs = CRS.from_epsg(3857)
    
    images = RAMPImageDataset(paths=image_paths, crs=target_crs, res=res)
    masks = RAMPMaskDataset(paths=label_paths, crs=target_crs, res=res)
    
    return IntersectionDataset(images, masks), label_paths


def get_all_ramp_regions(root: Path) -> list[str]:
    regions = [d.name for d in root.iterdir() if d.is_dir() and d.name.startswith("ramp_")]
    if not regions:
        raise ValueError(f"No RAMP regions found in {root}")
    return sorted(regions)


def split_regions(regions: list[str], val_ratio: float = 0.2, seed: int = 42):
    rng = random.Random(seed)
    shuffled = regions.copy()
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - val_ratio))
    return shuffled[:split_idx], shuffled[split_idx:]


def compute_num_queries_from_geojson(
    label_paths: list[Path], 
    percentile: float = 95.0,
    buffer_ratio: float = 0.2,
    min_queries: int = 50
) -> int:
    feature_counts = []
    
    for label_path in label_paths:
        geojson_files = list(Path(label_path).glob("*.geojson"))
        
        for geojson_file in tqdm(geojson_files, desc=f"Analyzing {label_path.name}", leave=False):
            try:
                with open(geojson_file, 'r') as f:
                    data = json.load(f)
                    if "features" in data:
                        feature_counts.append(len(data["features"]))
            except Exception:
                continue
    
    if not feature_counts:
        print(f"Warning: No features found, using min_queries={min_queries}")
        return min_queries
    
    max_features = int(np.percentile(feature_counts, percentile))
    recommended = max(int(max_features * (1 + buffer_ratio)), min_queries)
    
    print(f"Feature stats: mean={np.mean(feature_counts):.1f}, "
          f"median={np.median(feature_counts):.1f}, "
          f"p{percentile}={max_features}, "
          f"recommended_queries={recommended}")
    
    return recommended


def collate_fn_mask2former(batch: list[dict[str, Any]]) -> dict[str, Any]:
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
    def __init__(self, ce_weight=2.0, dice_weight=5.0, focal_weight=5.0, bce_weight=1.0, hausdorff_weight=0.1):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.bce_weight = bce_weight
        self.hausdorff_weight = hausdorff_weight
        
        self.dice_loss = smp.losses.DiceLoss(mode='binary', from_logits=True)
        self.focal_loss = smp.losses.FocalLoss(mode='binary', alpha=0.25, gamma=2.0)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.hausdorff_loss = HausdorffERLoss()
    
    def forward(self, pred_masks, pred_logits, gt_masks, gt_labels):
        B, Q, H, W = pred_masks.shape
        num_classes = pred_logits.shape[-1]
        
        if gt_masks.shape[-2:] != (H, W):
            gt_masks = F.interpolate(
                gt_masks.float(), 
                size=(H, W), 
                mode='nearest'
            ).bool()
        
        loss = 0.0
        
        if pred_logits is not None and self.ce_weight > 0:
            pred_classes = pred_logits.view(B * Q, num_classes)
            target_classes = gt_labels.view(B * Q)
            ce_loss = F.cross_entropy(pred_classes, target_classes, reduction="mean")
            loss += self.ce_weight * ce_loss
        
        pred_masks_flat = pred_masks.view(B * Q, 1, H, W)
        gt_masks_flat = gt_masks.view(B * Q, 1, H, W).float()
        
        if self.dice_weight > 0:
            dice = self.dice_loss(pred_masks_flat, gt_masks_flat)
            loss += self.dice_weight * dice
        
        if self.focal_weight > 0:
            focal = self.focal_loss(pred_masks_flat, gt_masks_flat)
            loss += self.focal_weight * focal
        
        if self.bce_weight > 0:
            bce = self.bce_loss(pred_masks_flat, gt_masks_flat)
            loss += self.bce_weight * bce
        
        if self.hausdorff_weight > 0:
            try:
                pred_sigmoid = torch.sigmoid(pred_masks_flat)
                hd = self.hausdorff_loss(pred_sigmoid, gt_masks_flat)
                loss += self.hausdorff_weight * hd
            except Exception:
                pass
        
        return loss


class EarlyStopping:
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