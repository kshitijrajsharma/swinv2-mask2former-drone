import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchgeo.samplers import RandomGeoSampler, Units
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryJaccardIndex,
    BinaryPrecision,
    BinaryRecall,
)
from torchmetrics.detection import MeanAveragePrecision
from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation

from src.config import Config
from src.utils import (
    get_all_ramp_regions,
    get_augmentation,
    get_image_processor,
    get_ramp_dataset,
    make_collate_fn,
    set_seed,
    split_regions,
)


class Mask2FormerModule(pl.LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self.image_processor = get_image_processor(cfg.pretrained_model)

        base_config = Mask2FormerConfig.from_pretrained(cfg.pretrained_model)
        base_config.num_labels = 2
        base_config.ignore_index = 255
        base_config.id2label = {0: "background", 1: "building"}
        base_config.label2id = {"background": 0, "building": 1}
        base_config.class_weight = cfg.class_weight
        base_config.dice_weight = cfg.dice_weight
        base_config.mask_weight = cfg.mask_weight

        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            cfg.pretrained_model, config=base_config, ignore_mismatched_sizes=True
        )

        metrics = MetricCollection(
            {
                "acc": BinaryAccuracy(),
                "p": BinaryPrecision(),
                "r": BinaryRecall(),
                "f1": BinaryF1Score(),
                "iou": BinaryJaccardIndex(),
            }
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")
        
        self.train_map = MeanAveragePrecision(class_metrics=True)
        self.val_map = MeanAveragePrecision(class_metrics=True)
        self.test_map = MeanAveragePrecision(class_metrics=True)

    def forward(self, pixel_values, mask_labels=None, class_labels=None):
        return self.model(
            pixel_values=pixel_values,
            mask_labels=mask_labels,
            class_labels=class_labels,
        )

    def _get_boundary_weights(self, masks):
        """Extract boundary pixels and create weight map emphasizing edges."""
        kernel_size = self.cfg.boundary_kernel_size
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)
        # Edge detection: max_pool - avg_pool
        boundaries = F.max_pool2d(masks.float(), kernel_size, stride=1, padding=kernel_size//2)
        boundaries = boundaries - F.avg_pool2d(masks.float(), kernel_size, stride=1, padding=kernel_size//2)
        boundaries = (boundaries.abs() > 0.1).float()
        
        # Boundary pixels get weight multiplier, interior pixels get 1x
        weights = torch.ones_like(masks.float())
        weights = weights + boundaries * self.cfg.boundary_weight_multiplier
        
        return weights.squeeze(1) if weights.shape[1] == 1 else weights

    def _compute_boundary_dice_loss(self, pred_masks, target_masks):
        """Compute boundary-weighted Dice loss for sharper edges."""
        boundary_weights = self._get_boundary_weights(target_masks)
        
        pred_flat = pred_masks.flatten(1)
        target_flat = target_masks.float().flatten(1)
        weight_flat = boundary_weights.flatten(1)
        
        # Standard weighted Dice formula
        intersection = (pred_flat * target_flat * weight_flat).sum(1)
        union = (pred_flat * weight_flat).sum(1) + (target_flat * weight_flat).sum(1)
        
        dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
        return 1.0 - dice.mean()

    def _compute_compactness_loss(self, pred_masks):
        """
        Penalize irregular shapes using compactness (isoperimetric quotient).
        Compact shapes (circles, squares) have low perimeter^2/area ratio.
        Irregular shapes have high ratio. Encourages regular, cartographic geometries.
        """
        losses = []
        for mask in pred_masks:
            if mask.sum() < 10:  # tiny masks probably noise
                continue
            
            # perimeter computation
            mask_float = mask.float().unsqueeze(0).unsqueeze(0)
            edges = F.max_pool2d(mask_float, 3, stride=1, padding=1) - mask_float
            perimeter = edges.abs().sum() + 1e-7
            
            area = mask.sum() + 1e-7
            
            # compactness: 4π * area / perimeter^2 (circle = 1, irregular < 1)
            # we encourage higher compactness
            compactness = (4 * 3.14159 * area) / (perimeter.clamp(min=1.0) ** 2)
            losses.append(1.0 - compactness.clamp(0, 1))
        
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=pred_masks.device)


    def _get_pred_masks(self, outputs, target_size):
        results = self.image_processor.post_process_instance_segmentation(
            outputs, target_sizes=[target_size] * outputs.masks_queries_logits.shape[0]
        )
        batch_binary_masks = []
        batch_instance_data = []
        
        for r in results:
            binary_mask = torch.zeros(target_size, device=self.device, dtype=torch.long)
            seg_map = r["segmentation"].to(self.device)
            
            instance_masks = []
            instance_boxes = []
            instance_scores = []
            instance_labels = []
            
            for info in r["segments_info"]:
                if info["label_id"] == 1:
                    instance_mask = (seg_map == info["id"]).long()
                    
                    ys, xs = torch.where(instance_mask)
                    if len(ys) == 0:
                        continue
                    
                    binary_mask = binary_mask | instance_mask
                    instance_masks.append(instance_mask.cpu().bool())
                    instance_boxes.append(self._compute_bbox(instance_mask))
                    
                    instance_scores.append(info["score"])
                    instance_labels.append(1)
            
            batch_binary_masks.append(binary_mask)
            
            batch_instance_data.append({
                "masks": torch.stack(instance_masks) if instance_masks else torch.zeros((0, *target_size), dtype=torch.bool),
                "boxes": torch.tensor(instance_boxes, dtype=torch.float32),
                "scores": torch.tensor(instance_scores, dtype=torch.float32),
                "labels": torch.tensor(instance_labels, dtype=torch.long),
            })
        
        return torch.stack(batch_binary_masks), batch_instance_data

    def training_step(self, batch, batch_idx):
        mask_labels = [m.to(self.device) for m in batch["mask_labels"]]
        class_labels = [c.to(self.device) for c in batch["class_labels"]]

        outputs = self(batch["pixel_values"], mask_labels, class_labels)
        base_loss = outputs.loss
        
        target = torch.stack([(m.sum(0) > 0).long() for m in mask_labels])
        binary_preds, _ = self._get_pred_masks(outputs, target.shape[-2:])
        boundary_loss = self._compute_boundary_dice_loss(binary_preds, target)

        compactness_loss = self._compute_compactness_loss(binary_preds)
        
        total_loss = (
            base_loss 
            + self.cfg.boundary_loss_weight * boundary_loss
            + self.cfg.compactness_loss_weight * compactness_loss
        )
        
        self.log("train_loss", total_loss, prog_bar=True, sync_dist=True)
        self.log("train_base_loss", base_loss, prog_bar=False, sync_dist=True)
        self.log("train_boundary_loss", boundary_loss, prog_bar=False, sync_dist=True)
        self.log("train_compactness_loss", compactness_loss, prog_bar=False, sync_dist=True)



        with torch.no_grad():
            binary_preds_metrics, instance_preds = self._get_pred_masks(outputs, target.shape[-2:])
            
            if binary_preds_metrics.sum() > 0 or target.sum() > 0:
                self.train_metrics.update(binary_preds_metrics.flatten(), target.flatten())
            
            target_instances = self._prepare_target_instances(mask_labels, class_labels)
            if any(len(p['masks']) > 0 for p in instance_preds) or any(len(t['masks']) > 0 for t in target_instances):
                self.train_map.update(instance_preds, target_instances)

        return total_loss

    def on_train_epoch_end(self):
        self._log_epoch_metrics("train", self.train_metrics, self.train_map)

    def _eval_step(self, batch, metrics, map_metric):
        mask_labels = [m.to(self.device) for m in batch["mask_labels"]]
        class_labels = [c.to(self.device) for c in batch["class_labels"]]
        outputs = self(batch["pixel_values"], mask_labels, class_labels)
        
        target = torch.stack([(m.sum(0) > 0).long() for m in mask_labels])
        binary_preds, instance_preds = self._get_pred_masks(outputs, target.shape[-2:])
        
        if binary_preds.sum() > 0 or target.sum() > 0:
            metrics.update(binary_preds.flatten(), target.flatten())
        
        target_instances = self._prepare_target_instances(mask_labels, class_labels)
        if any(len(p['masks']) > 0 for p in instance_preds) or any(len(t['masks']) > 0 for t in target_instances):
            map_metric.update(instance_preds, target_instances)
        
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        loss = self._eval_step(batch, self.val_metrics, self.val_map)
        self.log("val_loss", loss, prog_bar=False, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        self._log_epoch_metrics("val", self.val_metrics, self.val_map, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss = self._eval_step(batch, self.test_metrics, self.test_map)
        self.log("test_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def on_test_epoch_end(self):
        self._log_epoch_metrics("test", self.test_metrics, self.test_map, prog_bar=True)
    
    def _log_epoch_metrics(self, prefix, metrics, map_metric, prog_bar=False):
        """Helper to log metrics and mAP results at epoch end."""
        self.log_dict(metrics.compute(), prog_bar=prog_bar, sync_dist=True)
        metrics.reset()
        
        map_results = map_metric.compute()
        self.log(f"{prefix}_map", map_results["map"], prog_bar=prog_bar, sync_dist=True)
        self.log(f"{prefix}_map_50", map_results["map_50"], prog_bar=prog_bar, sync_dist=True)
        map_metric.reset()

    def _prepare_target_instances(self, mask_labels, class_labels):
        target_instances = []
        for masks, labels in zip(mask_labels, class_labels):
            instance_masks = []
            instance_boxes = []
            instance_labels = []
            
            for mask, label in zip(masks, labels):
                if mask.sum() > 0:
                    instance_masks.append(mask.cpu().bool())
                    instance_boxes.append(self._compute_bbox(mask))
                    instance_labels.append(label.item())
            
            h, w = masks.shape[-2:]
            target_instances.append({
                "masks": torch.stack(instance_masks) if instance_masks else torch.zeros((0, h, w), dtype=torch.bool),
                "boxes": torch.tensor(instance_boxes, dtype=torch.float32) if instance_boxes else torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.tensor(instance_labels, dtype=torch.long) if instance_labels else torch.zeros((0,), dtype=torch.long),
            })
        
        return target_instances
    
    def _compute_bbox(self, mask):
        """Compute bounding box from mask."""
        ys, xs = torch.where(mask)
        x1, y1 = xs.min().item(), ys.min().item()
        x2, y2 = xs.max().item(), ys.max().item()
        return [x1, y1, x2, y2]
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=self.cfg.scheduler_factor, patience=self.cfg.scheduler_patience
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_map_50",
                "frequency": 1,
            },
        }

    def visualize_batch(self, batch, num_samples=4, save_path=None):
        """Visualize image, ground truth, and prediction with instance segmentation."""
        import matplotlib.pyplot as plt

        self.eval()
        with torch.no_grad():
            mask_labels = [m.to(self.device) for m in batch["mask_labels"]]
            class_labels = [c.to(self.device) for c in batch["class_labels"]]
            outputs = self(
                batch["pixel_values"].to(self.device), mask_labels, class_labels
            )

            target = torch.stack([(m.sum(0) > 0).long() for m in mask_labels])
            binary_preds, instance_preds = self._get_pred_masks(outputs, target.shape[-2:])

        images = batch["pixel_values"].cpu()
        target = target.cpu()
        binary_preds = binary_preds.cpu()

        num_samples = min(num_samples, len(images))
        _, axes = plt.subplots(num_samples, 4, figsize=(12, 3 * num_samples))
        if num_samples == 1:
            axes = axes[None, :]

        for i in range(num_samples):
            img = images[i].permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)

            axes[i, 0].imshow(img)
            axes[i, 0].set_title("Image")
            axes[i, 1].imshow(target[i], cmap="gray", vmin=0, vmax=1)
            axes[i, 1].set_title("Ground Truth")
            axes[i, 2].imshow(binary_preds[i], cmap="gray", vmin=0, vmax=1)
            axes[i, 2].set_title("Binary Prediction")
            
            instance_viz = self._create_instance_viz(instance_preds[i]["masks"], img.shape[:2])
            axes[i, 3].imshow(instance_viz)
            axes[i, 3].set_title(f"Instances ({len(instance_preds[i]['masks'])} bldgs)")

            for ax in axes[i]:
                ax.axis("off")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close()
    
    def _create_instance_viz(self, instance_masks, target_size):
        """Create colored visualization of instance masks."""
        from matplotlib import cm
        
        viz = np.zeros((*target_size, 3), dtype=np.float32)
        num_instances = len(instance_masks)
        
        if num_instances == 0:
            return viz
        
        colors = cm.get_cmap('tab20')(np.linspace(0, 1, max(20, num_instances)))
        
        for idx, mask in enumerate(instance_masks):
            mask_np = mask.cpu().numpy()
            color = colors[idx % len(colors)][:3]
            for c in range(3):
                viz[:, :, c] = np.where(mask_np, color[c], viz[:, :, c])
        
        return viz


class OAMDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.image_processor = get_image_processor(
            cfg.pretrained_model, cfg.image_size
        )
        self.collate_fn = make_collate_fn(self.image_processor)

    def setup(self, stage=None):
        if self.cfg.train_regions and self.cfg.val_regions:
            train_regions, val_regions = self.cfg.train_regions, self.cfg.val_regions
        else:
            all_regions = get_all_ramp_regions(self.cfg.data_root)
            train_regions, val_regions = split_regions(
                all_regions, self.cfg.val_split, self.cfg.seed
            )

        self.train_dataset = get_ramp_dataset(self.cfg.data_root, train_regions)
        if self.cfg.enable_data_augmentation:
            print("Applying data augmentation to training dataset")
            self.train_dataset.transforms = get_augmentation()
        print(f"Train dataset length: {len(self.train_dataset)}")
        
        self.val_dataset = get_ramp_dataset(self.cfg.data_root, val_regions)
        print(f"Val dataset length: {len(self.val_dataset)}")
        
        self.test_dataset = get_ramp_dataset(self.cfg.data_root, self.cfg.test_regions)
        print(f"Test dataset length: {len(self.test_dataset)}")

    def _create_dataloader(self, dataset, split_name):
        """Helper to create dataloader with sampler."""
        sampler = RandomGeoSampler(
            dataset,
            size=self.cfg.sampler_size,
            units=Units.PIXELS,
        )
        print(f"{split_name}_sampler length", len(sampler))
        return DataLoader(
            dataset,
            sampler=sampler,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            drop_last=True,
        )

    def train_dataloader(self):
        return self._create_dataloader(self.train_dataset, "train")

    def val_dataloader(self):
        return self._create_dataloader(self.val_dataset, "val")

    def test_dataloader(self):
        return self._create_dataloader(self.test_dataset, "test")


def main():
    cfg = Config()
    set_seed(cfg.seed)

    model = Mask2FormerModule(cfg)
    datamodule = OAMDataModule(cfg)

    callbacks = [
        EarlyStopping(
            monitor="val_map_50",
            patience=cfg.early_stopping_patience,
            mode="max",
        ),
        ModelCheckpoint(
            dirpath=cfg.output_dir,
            filename="best",
            monitor="val_map_50",
            mode="max",
            save_top_k=1,
        ),
    ]

    logger = (
        WandbLogger(project=cfg.wandb_project, name=cfg.wandb_run_name)
        if cfg.use_wandb
        else None
    )

    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        accelerator="auto",
        devices="auto",
        callbacks=callbacks,
        logger=logger,
        precision="16-mixed",
        default_root_dir=cfg.output_dir,
    )

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
