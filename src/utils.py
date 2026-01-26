import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import ndimage
from torchgeo.datasets import RasterDataset, VectorDataset
from transformers import Mask2FormerImageProcessor


class RAMPImageDataset(RasterDataset):
    filename_glob = "*.tif"
    is_image = True
    all_bands = ("R", "G", "B")
    rgb_bands = ("R", "G", "B")


class RAMPMaskDataset(VectorDataset):
    filename_glob = "*.geojson"

    def __init__(self, paths, crs=None, **kwargs):
        super().__init__(
            paths=paths, crs=crs, task="semantic_segmentation", **kwargs
        )  # torchgeo has bug on instance_seg , it drops polygon from geojson if its in edge but the mask remains , this is where the bug is : `torchgeo/datasets/geo.py` (lines ~1070-1100) #TODO : raise this to torchgeo

        # super().__init__(
        #     paths=paths, crs=crs, task="instance_segmentation", **kwargs
        # )  # torchgeo has bug on instance_seg , it drops polygon from geojson if its in edge but the mask remains , this is where the bug is : `torchgeo/datasets/geo.py` (lines ~1070-1100) #TODO : raise this to torchgeo


def get_ramp_dataset(root: Path, regions: list[str]):

    image_paths, label_paths = [], []
    print(f"Finding image,label path for {regions}...")
    for region in regions:
        region_path = root / region
        img_path, lbl_path = region_path / "source", region_path / "labels"
        if img_path.exists() and lbl_path.exists():
            image_paths.append(img_path)
            label_paths.append(lbl_path)

    if not image_paths:
        raise ValueError(f"No valid regions found in {root}")

    print("Loading images ...")
    images = RAMPImageDataset(paths=image_paths)
    print(
        f"Loaded {len(images)} image tiles. using crs : {images.crs} with res {images.res}"
    )
    print("Loading labels ...")
    masks = RAMPMaskDataset(paths=label_paths)

    print(
        f"Loaded {len(masks)} mask tiles. using crs : {masks.crs} with res {masks.res}"
    )
    return images & masks


def get_all_ramp_regions(root: Path) -> list[str]:
    regions = [
        d.name for d in root.iterdir() if d.is_dir() and d.name.startswith("ramp_")
    ]
    if not regions:
        raise ValueError(f"No RAMP regions found in {root}")
    return sorted(regions)


def split_regions(regions: list[str], val_ratio: float = 0.2, seed: int = 42):
    rng = random.Random(seed)
    shuffled = regions.copy()
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - val_ratio))
    return shuffled[:split_idx], shuffled[split_idx:]


def get_image_processor(
    pretrained_model: str, size: int = 256
) -> Mask2FormerImageProcessor:
    return Mask2FormerImageProcessor.from_pretrained(
        pretrained_model,
        num_labels=2,
        # do_reduce_labels=True,
        ignore_index=255,
        size=size,
        do_normalize=True,
    )


def make_collate_fn(image_processor: Mask2FormerImageProcessor):
    def collate_fn(
        batch: list[dict[str, Any]],
    ) -> dict[str, Any]:  # source : https://debuggercafe.com/fine-tuning-mask2former/
        images = [sample["image"].float() for sample in batch]
        inputs = image_processor(images=images, return_tensors="pt")

        mask_labels = []
        class_labels = []

        for sample in batch:
            mask = sample["mask"]
            # semantic
            if mask.ndim == 2:
                mask_np = mask.numpy() if isinstance(mask, torch.Tensor) else mask

                labeled_mask, num_instances = ndimage.label(mask_np > 0)

                if num_instances > 0:
                    instance_masks = []
                    for instance_id in range(1, num_instances + 1):
                        instance_mask = (labeled_mask == instance_id).astype(np.float32)
                        instance_masks.append(torch.from_numpy(instance_mask))

                    mask_labels.append(torch.stack(instance_masks))
                    class_labels.append(
                        torch.tensor([1] * num_instances, dtype=torch.long)
                    )
                else:
                    H, W = mask.shape
                    mask_labels.append(torch.zeros((0, H, W), dtype=torch.float32))
                    class_labels.append(torch.tensor([], dtype=torch.long))

            # instance_seg option
            elif mask.ndim == 3:
                instance_masks = []
                instance_classes = []

                for i in range(mask.shape[0]):
                    instance_mask = mask[i]

                    if instance_mask.sum() > 0:
                        instance_masks.append(instance_mask.float())
                        instance_classes.append(1)

                if instance_masks:
                    mask_labels.append(torch.stack(instance_masks))
                    class_labels.append(
                        torch.tensor(instance_classes, dtype=torch.long)
                    )
                else:
                    # no buildings
                    H, W = mask.shape[-2:]
                    mask_labels.append(torch.zeros((0, H, W), dtype=torch.float32))
                    class_labels.append(torch.tensor([], dtype=torch.long))

        inputs["mask_labels"] = mask_labels
        inputs["class_labels"] = class_labels
        return inputs

    return collate_fn


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
