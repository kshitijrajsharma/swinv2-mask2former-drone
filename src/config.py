from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # Data
    data_root: Path = Path("/home/krschap/data")
    output_dir: Path = Path("outputs")
    res: float = 0.3
    seed: int = 42
    
    # Manual region split (leave empty for automatic split)
    train_regions: list[str] = field(default_factory=list)
    val_regions: list[str] = field(default_factory=list)
    val_split: float = 0.2
    
    # Model (set num_queries=0 for automatic computation from dataset)
    num_queries: int = 0
    pretrained_model: str = "facebook/mask2former-swin-base-coco-instance"
    
    # Stage 1
    stage1_epochs: int = 50
    stage1_batch_size: int = 4
    stage1_lr: float = 1e-4
    stage1_weight_decay: float = 0.01
    stage1_num_samples: int = 100000
    
    # Stage 2
    stage2_epochs: int = 30
    stage2_batch_size: int = 2
    stage1_checkpoint: str = "stage1_best.pt"
    
    # Stage 2 - Optuna
    use_optuna: bool = True
    n_trials: int = 20
    optuna_study_name: str = "stage2_hparam_search"
    
    # Stage 2 - Default hyperparams (used if use_optuna=False)
    stage2_lr: float = 1e-5
    stage2_weight_decay: float = 0.01
    stage2_lora_rank: int = 16
    stage2_loss_hausdorff_weight: float = 0.1
    
    # Training
    num_workers: int = 4
    gradient_accumulation_steps: int = 1
    early_stopping_patience: int = 10
    checkpoint_freq: int = 5
    
    # Loss weights
    loss_ce_weight: float = 2.0
    loss_dice_weight: float = 5.0
    loss_focal_weight: float = 5.0
    loss_hausdorff_weight: float = 0.1