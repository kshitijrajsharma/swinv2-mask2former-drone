import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchgeo.samplers import RandomBatchGeoSampler
from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation

from config import Config
from utils import (
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

    def forward(self, pixel_values, mask_labels=None, class_labels=None):
        return self.model(
            pixel_values=pixel_values,
            mask_labels=mask_labels,
            class_labels=class_labels,
        )

    def training_step(self, batch, batch_idx):
        mask_labels = [m.to(self.device) for m in batch["mask_labels"]]
        class_labels = [c.to(self.device) for c in batch["class_labels"]]
        outputs = self(batch["pixel_values"], mask_labels, class_labels)
        self.log("train_loss", outputs.loss, prog_bar=True, sync_dist=True)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        mask_labels = [m.to(self.device) for m in batch["mask_labels"]]
        class_labels = [c.to(self.device) for c in batch["class_labels"]]
        outputs = self(batch["pixel_values"], mask_labels, class_labels)
        self.log("val_loss", outputs.loss, prog_bar=True, sync_dist=True)
        return outputs.loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cfg.epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class RAMPDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.image_processor = get_image_processor(cfg.pretrained_model)
        self.collate_fn = make_collate_fn(self.image_processor)

    def setup(self, stage=None):
        if self.cfg.train_regions and self.cfg.val_regions:
            train_regions, val_regions = self.cfg.train_regions, self.cfg.val_regions
        else:
            all_regions = get_all_ramp_regions(self.cfg.data_root)
            train_regions, val_regions = split_regions(
                all_regions, self.cfg.val_split, self.cfg.seed
            )

        self.train_dataset = get_ramp_dataset(
            self.cfg.data_root, train_regions, self.cfg.res
        )
        self.val_dataset = get_ramp_dataset(
            self.cfg.data_root, val_regions, self.cfg.res
        )

    def train_dataloader(self):
        sampler = RandomBatchGeoSampler(
            self.train_dataset,
            size=256,
            batch_size=self.cfg.batch_size,
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            num_workers=self.cfg.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        sampler = RandomBatchGeoSampler(
            self.val_dataset, size=256, batch_size=self.cfg.batch_size
        )
        return DataLoader(
            self.val_dataset,
            batch_sampler=sampler,
            num_workers=self.cfg.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )


def main():
    cfg = Config()
    set_seed(cfg.seed)

    model = Mask2FormerModule(cfg)
    datamodule = RAMPDataModule(cfg)

    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=cfg.early_stopping_patience, mode="min"
        ),
        ModelCheckpoint(
            dirpath=cfg.output_dir,
            filename="best",
            monitor="val_loss",
            mode="min",
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
