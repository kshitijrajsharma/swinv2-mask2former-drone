import numpy as np
import pytorch_lightning as pl
import torch
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
                    binary_mask = binary_mask | instance_mask
                    
                    instance_masks.append(instance_mask.cpu().bool())
                    
                    ys, xs = torch.where(instance_mask)
                    if len(ys) > 0:
                        x1, y1 = xs.min().item(), ys.min().item()
                        x2, y2 = xs.max().item(), ys.max().item()
                        instance_boxes.append([x1, y1, x2, y2])
                    else:
                        instance_boxes.append([0, 0, 1, 1])
                    
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
        self.log("train_loss", outputs.loss, prog_bar=True, sync_dist=True)

        with torch.no_grad():
            target = torch.stack([(m.sum(0) > 0).long() for m in mask_labels])
            binary_preds, instance_preds = self._get_pred_masks(outputs, target.shape[-2:])
            self.train_metrics.update(binary_preds.flatten(), target.flatten())
            
            target_instances = self._prepare_target_instances(mask_labels, class_labels)
            self.train_map.update(instance_preds, target_instances)

        return outputs.loss

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute(), sync_dist=True)
        self.train_metrics.reset()
        
        map_results = self.train_map.compute()
        self.log("train_map", map_results["map"], sync_dist=True)
        self.log("train_map_50", map_results["map_50"], sync_dist=True)
        self.train_map.reset()

    def validation_step(self, batch, batch_idx):
        mask_labels = [m.to(self.device) for m in batch["mask_labels"]]
        class_labels = [c.to(self.device) for c in batch["class_labels"]]
        outputs = self(batch["pixel_values"], mask_labels, class_labels)
        self.log("val_loss", outputs.loss, prog_bar=False, sync_dist=True)

        target = torch.stack([(m.sum(0) > 0).long() for m in mask_labels])
        binary_preds, instance_preds = self._get_pred_masks(outputs, target.shape[-2:])
        self.val_metrics.update(binary_preds.flatten(), target.flatten())
        
        target_instances = self._prepare_target_instances(mask_labels, class_labels)
        self.val_map.update(instance_preds, target_instances)

        return outputs.loss

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), prog_bar=True, sync_dist=True)
        self.val_metrics.reset()
        
        map_results = self.val_map.compute()
        self.log("val_map", map_results["map"], prog_bar=True, sync_dist=True)
        self.log("val_map_50", map_results["map_50"], prog_bar=True, sync_dist=True)
        self.val_map.reset()

    def test_step(self, batch, batch_idx):
        mask_labels = [m.to(self.device) for m in batch["mask_labels"]]
        class_labels = [c.to(self.device) for c in batch["class_labels"]]
        outputs = self(batch["pixel_values"], mask_labels, class_labels)
        self.log("test_loss", outputs.loss, prog_bar=True, sync_dist=True)

        target = torch.stack([(m.sum(0) > 0).long() for m in mask_labels])
        binary_preds, instance_preds = self._get_pred_masks(outputs, target.shape[-2:])
        self.test_metrics.update(binary_preds.flatten(), target.flatten())
        
        target_instances = self._prepare_target_instances(mask_labels, class_labels)
        self.test_map.update(instance_preds, target_instances)

        return outputs.loss

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute(), prog_bar=True, sync_dist=True)
        self.test_metrics.reset()
        
        map_results = self.test_map.compute()
        self.log("test_map", map_results["map"], prog_bar=True, sync_dist=True)
        self.log("test_map_50", map_results["map_50"], prog_bar=True, sync_dist=True)
        self.test_map.reset()

    def _prepare_target_instances(self, mask_labels, class_labels):
        target_instances = []
        for masks, labels in zip(mask_labels, class_labels):
            instance_masks = []
            instance_boxes = []
            instance_labels = []
            
            for mask, label in zip(masks, labels):
                if mask.sum() > 0:
                    instance_masks.append(mask.cpu().bool())
                    
                    ys, xs = torch.where(mask)
                    x1, y1 = xs.min().item(), ys.min().item()
                    x2, y2 = xs.max().item(), ys.max().item()
                    instance_boxes.append([x1, y1, x2, y2])
                    instance_labels.append(label.item())
            
            target_instances.append({
                "masks": torch.stack(instance_masks) if instance_masks else torch.zeros((0, *mask_labels[0].shape[-2:]), dtype=torch.bool),
                "boxes": torch.tensor(instance_boxes, dtype=torch.float32),
                "labels": torch.tensor(instance_labels, dtype=torch.long),
            })
        
        return target_instances
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cfg.epochs
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "strict": True,
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
        self.image_size = 256
        self.image_processor = get_image_processor(
            cfg.pretrained_model, self.image_size
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
        print(f"Train dataset length: {len(self.train_dataset)}")
        self.val_dataset = get_ramp_dataset(self.cfg.data_root, val_regions)
        print(f"Val dataset length: {len(self.val_dataset)}")
        self.test_dataset = get_ramp_dataset(self.cfg.data_root, self.cfg.test_regions)
        print(f"Test dataset length: {len(self.test_dataset)}")

    def train_dataloader(self):
        sampler = RandomGeoSampler(
            self.train_dataset,
            size=256,
            units=Units.PIXELS,
        )
        print("train_sampler length", len(sampler))
        return DataLoader(
            self.train_dataset,
            sampler=sampler,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        sampler = RandomGeoSampler(
            self.val_dataset,
            size=256,
            units=Units.PIXELS,
        )
        print("val_sampler length", len(sampler))

        return DataLoader(
            self.val_dataset,
            sampler=sampler,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            drop_last=True,
        )

    def test_dataloader(self):
        sampler = RandomGeoSampler(
            self.test_dataset,
            size=256,
            units=Units.PIXELS,
        )
        print("test_sampler length", len(sampler))

        return DataLoader(
            self.test_dataset,
            sampler=sampler,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            drop_last=True,
        )


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
