from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    data_root: Path = Path("/home/krschap/data")
    output_dir: Path = Path("outputs")
    res: float = 0.3
    seed: int = 42

    train_regions: list[str] = field(
        default_factory=lambda: ['ramp_dhaka_bangladesh', 'ramp_barishal_bangladesh','ramp_sylhet_bangladesh']
    )
    val_regions: list[str] = field(
        default_factory=lambda: ['ramp_coxs_bazar_bangladesh']
    )
    
    val_split: float = 0.2
    
    num_queries: int = 0
    pretrained_model: str = "facebook/mask2former-swin-base-coco-instance"
    
    stage1_epochs: int = 100
    stage1_batch_size: int = 4
    stage1_lr: float = 1e-4
    stage1_weight_decay: float = 0.01
    stage1_max_batches_per_epoch: int = 50
    
    stage2_epochs: int = 2
    stage2_batch_size: int = 2
    stage1_checkpoint: str = "stage1_best.pt"
    
    use_optuna: bool = True
    n_trials: int = 20
    optuna_study_name: str = "stage2_hparam_search"
    
    stage2_lr: float = 1e-5
    stage2_weight_decay: float = 0.01
    stage2_lora_rank: int = 16
    
    num_workers: int = 4
    gradient_accumulation_steps: int = 1
    early_stopping_patience: int = 10
    checkpoint_freq: int = 5
    
    loss_ce_weight: float = 2.0
    loss_dice_weight: float = 5.0
    loss_focal_weight: float = 5.0
    loss_bce_weight: float = 1.0
    loss_hausdorff_weight: float = 0.1