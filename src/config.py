from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    data_root: Path = Path("data/banepa")
    output_dir: Path = Path("outputs")
    seed: int = 64

    train_regions: list[str] = field(default_factory=lambda: ["train"])
    val_regions: list[str] = field(default_factory=lambda: ["val"])
    test_regions: list[str] = field(default_factory=lambda: ["test"])

    val_split: float = 0.2
    pretrained_model: str = (
        "facebook/mask2former-swin-base-IN21k-coco-instance"  # https://huggingface.co/facebook/mask2former-swin-base-IN21k-coco-instance
    )

    epochs: int = 50
    batch_size: int = 16

    # hyper params
    dice_weight: float = 5.0
    mask_weight: float = 5.0
    class_weight: float = 5.0

    boundary_loss_weight: float = 5.0 
    compactness_loss_weight: float = 5.0

    learning_rate: float = 0.00001
    weight_decay: float = 0.0001
    early_stopping_patience: int = 15

    num_workers: int = 31
    use_wandb: bool = True
    wandb_project: str = "building-seg-mask2former"
    wandb_run_name: str = "default_run"

    enable_data_augmentation: bool = False

    image_size: int = 256
    sampler_size: int = 256
    
    boundary_kernel_size: int = 3
    boundary_weight_multiplier: float = 10.0
    
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
