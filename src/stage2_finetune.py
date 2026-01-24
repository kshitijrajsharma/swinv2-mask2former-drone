import optuna
import torch
from optuna.pruners import MedianPruner
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Mask2FormerForUniversalSegmentation
from torchgeo.samplers import RandomBatchGeoSampler
from config import Config
from utils import (
    CombinedLoss,
    EarlyStopping,
    collate_fn_mask2former,
    compute_num_queries,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        images = batch["pixel_values"].to(device)
        targets = batch["mask_labels"]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        outputs = model(pixel_values=images, output_hidden_states=True)
        
        pred_masks = outputs.masks_queries_logits
        pred_logits = outputs.class_queries_logits
        
        max_instances = max(t["masks"].shape[0] for t in targets)
        gt_masks = []
        gt_labels = []
        
        for t in targets:
            masks = t["masks"]
            labels = t["class_labels"]
            if masks.shape[0] < max_instances:
                pad_size = max_instances - masks.shape[0]
                masks = torch.cat([masks, torch.zeros(pad_size, *masks.shape[1:], device=masks.device, dtype=masks.dtype)])
                labels = torch.cat([labels, torch.zeros(pad_size, device=labels.device, dtype=labels.dtype)])
            gt_masks.append(masks)
            gt_labels.append(labels)
        
        gt_masks = torch.stack(gt_masks)
        gt_labels = torch.stack(gt_labels)
        
        loss = criterion(pred_masks, pred_logits, gt_masks, gt_labels)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return total_loss / len(dataloader)


@torch.no_grad()
def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Validation")
    for batch in pbar:
        images = batch["pixel_values"].to(device)
        targets = batch["mask_labels"]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        outputs = model(pixel_values=images, output_hidden_states=True)
        
        pred_masks = outputs.masks_queries_logits
        pred_logits = outputs.class_queries_logits
        
        max_instances = max(t["masks"].shape[0] for t in targets)
        gt_masks = []
        gt_labels = []
        
        for t in targets:
            masks = t["masks"]
            labels = t["class_labels"]
            if masks.shape[0] < max_instances:
                pad_size = max_instances - masks.shape[0]
                masks = torch.cat([masks, torch.zeros(pad_size, *masks.shape[1:], device=masks.device, dtype=masks.dtype)])
                labels = torch.cat([labels, torch.zeros(pad_size, device=labels.device, dtype=labels.dtype)])
            gt_masks.append(masks)
            gt_labels.append(labels)
        
        gt_masks = torch.stack(gt_masks)
        gt_labels = torch.stack(gt_labels)
        
        loss = criterion(pred_masks, pred_logits, gt_masks, gt_labels)
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return total_loss / len(dataloader)


def objective(trial, cfg, train_loader, val_loader, device):
    """Optuna objective function"""
    
    lr = trial.suggest_float("lr", 1e-6, 1e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True)
    lora_rank = trial.suggest_categorical("lora_rank", [8, 16, 32])
    hausdorff_weight = trial.suggest_float("hausdorff_weight", 0.0, 0.5)
    
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        cfg.pretrained_model,
        id2label={0: "background", 1: "building"},
        num_labels=2,
        num_queries=cfg.num_queries,
        ignore_mismatched_sizes=True,
    )
    
    checkpoint_path = cfg.output_dir / cfg.stage1_checkpoint
    if checkpoint_path.exists():
        load_checkpoint(model, None, checkpoint_path, device)
    
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        target_modules=["query", "key", "value"],
        lora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model = model.to(device)
    
    for name, param in model.named_parameters():
        if "model.pixel_level_module" in name:
            param.requires_grad = False
    
    criterion = CombinedLoss(
        ce_weight=cfg.loss_ce_weight,
        dice_weight=cfg.loss_dice_weight,
        focal_weight=cfg.loss_focal_weight,
        hausdorff_weight=hausdorff_weight,
    )
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.stage2_epochs,
    )
    
    early_stopping = EarlyStopping(patience=5)
    
    for epoch in range(cfg.stage2_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()
        
        trial.report(val_loss, epoch)
        
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        if early_stopping(val_loss):
            break
    
    return val_loss


def train_with_best_params(cfg, train_loader, val_loader, device, best_params):
    """Train with best hyperparameters from Optuna"""
    
    print("\nTraining with best hyperparameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        cfg.pretrained_model,
        id2label={0: "background", 1: "building"},
        num_labels=2,
        num_queries=cfg.num_queries,
        ignore_mismatched_sizes=True,
    )
    
    checkpoint_path = cfg.output_dir / cfg.stage1_checkpoint
    if checkpoint_path.exists():
        load_checkpoint(model, None, checkpoint_path, device)
    
    lora_config = LoraConfig(
        r=best_params["lora_rank"],
        lora_alpha=best_params["lora_rank"] * 2,
        target_modules=["query", "key", "value"],
        lora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model = model.to(device)
    
    for name, param in model.named_parameters():
        if "model.pixel_level_module" in name:
            param.requires_grad = False
    
    criterion = CombinedLoss(
        ce_weight=cfg.loss_ce_weight,
        dice_weight=cfg.loss_dice_weight,
        focal_weight=cfg.loss_focal_weight,
        hausdorff_weight=best_params["hausdorff_weight"],
    )
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.stage2_epochs,
    )
    
    early_stopping = EarlyStopping(patience=cfg.early_stopping_patience)
    best_val_loss = float("inf")
    
    for epoch in range(cfg.stage2_epochs):
        print(f"\nEpoch {epoch + 1}/{cfg.stage2_epochs}")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = cfg.output_dir / "stage2_best.pt"
            save_checkpoint(model, optimizer, epoch, val_loss, best_path)
            print(f"Saved best checkpoint: {best_path}")
        
        if (epoch + 1) % cfg.checkpoint_freq == 0:
            ckpt_path = cfg.output_dir / f"stage2_epoch_{epoch + 1}.pt"
            save_checkpoint(model, optimizer, epoch, val_loss, ckpt_path)
        
        if early_stopping(val_loss):
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break
    
    return best_val_loss


def train_without_optuna(cfg, train_loader, val_loader, device):
    """Train with fixed hyperparameters"""
    
    print("\nTraining with fixed hyperparameters")
    
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        cfg.pretrained_model,
        id2label={0: "background", 1: "building"},
        num_labels=2,
        num_queries=cfg.num_queries,
        ignore_mismatched_sizes=True,
    )
    
    checkpoint_path = cfg.output_dir / cfg.stage1_checkpoint
    if checkpoint_path.exists():
        load_checkpoint(model, None, checkpoint_path, device)
        print(f"Loaded checkpoint: {checkpoint_path}")
    
    lora_config = LoraConfig(
        r=cfg.stage2_lora_rank,
        lora_alpha=cfg.stage2_lora_rank * 2,
        target_modules=["query", "key", "value"],
        lora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model = model.to(device)
    
    for name, param in model.named_parameters():
        if "model.pixel_level_module" in name:
            param.requires_grad = False
    
    criterion = CombinedLoss(
        ce_weight=cfg.loss_ce_weight,
        dice_weight=cfg.loss_dice_weight,
        focal_weight=cfg.loss_focal_weight,
        hausdorff_weight=cfg.stage2_loss_hausdorff_weight,
    )
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.stage2_lr,
        weight_decay=cfg.stage2_weight_decay,
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.stage2_epochs,
    )
    
    early_stopping = EarlyStopping(patience=cfg.early_stopping_patience)
    best_val_loss = float("inf")
    
    for epoch in range(cfg.stage2_epochs):
        print(f"\nEpoch {epoch + 1}/{cfg.stage2_epochs}")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = cfg.output_dir / "stage2_best.pt"
            save_checkpoint(model, optimizer, epoch, val_loss, best_path)
            print(f"Saved best checkpoint: {best_path}")
        
        if (epoch + 1) % cfg.checkpoint_freq == 0:
            ckpt_path = cfg.output_dir / f"stage2_epoch_{epoch + 1}.pt"
            save_checkpoint(model, optimizer, epoch, val_loss, ckpt_path)
        
        if early_stopping(val_loss):
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break
    
    return best_val_loss


def run_stage2(train_dataset, val_dataset, cfg=None):
    """Run Stage 2 fine-tuning with any TorchGeo dataset
    
    Args:
        train_dataset: TorchGeo dataset for training
        val_dataset: TorchGeo dataset for validation
        cfg: Config object (creates default if None)
    """
    if cfg is None:
        cfg = Config()
    
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    if cfg.num_queries == 0:
        cfg.num_queries = compute_num_queries(train_dataset, num_samples=100)
    else:
        print(f"Using configured num_queries: {cfg.num_queries}")
    
    train_sampler = RandomBatchGeoSampler(
        train_dataset,
        size=256,
        batch_size=cfg.stage2_batch_size,
        length=len(train_dataset) // cfg.stage2_batch_size,
    )
    
    val_sampler = RandomBatchGeoSampler(
        val_dataset,
        size=256,
        batch_size=cfg.stage2_batch_size,
        length=len(val_dataset) // cfg.stage2_batch_size,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn_mask2former,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn_mask2former,
        pin_memory=True,
    )
    
    if cfg.use_optuna:
        print(f"\nStage 2: Hyperparameter search with Optuna ({cfg.n_trials} trials)")
        
        study = optuna.create_study(
            study_name=cfg.optuna_study_name,
            direction="minimize",
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        )
        
        study.optimize(
            lambda trial: objective(trial, cfg, train_loader, val_loader, device),
            n_trials=cfg.n_trials,
            show_progress_bar=True,
        )
        
        print(f"\nBest trial: {study.best_trial.number}")
        print(f"Best validation loss: {study.best_value:.4f}")
        print("Best hyperparameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        
        best_val_loss = train_with_best_params(cfg, train_loader, val_loader, device, study.best_params)
        print(f"\nStage 2 complete. Best val loss: {best_val_loss:.4f}")
    else:
        best_val_loss = train_without_optuna(cfg, train_loader, val_loader, device)
        print(f"\nStage 2 complete. Best val loss: {best_val_loss:.4f}")
    
    return best_val_loss


if __name__ == "__main__":
    from utils import get_all_ramp_regions, get_ramp_dataset, split_regions
    
    cfg = Config()
    
    print("Example: Using RAMP dataset for Stage 2")
    all_regions = get_all_ramp_regions(cfg.data_root)
    target_regions = all_regions[:5]
    
    train_regions, val_regions = split_regions(target_regions, 0.2, cfg.seed)
    print(f"Train regions: {train_regions}")
    print(f"Val regions: {val_regions}")
    
    train_dataset = get_ramp_dataset(cfg.data_root, train_regions, cfg.res)
    val_dataset = get_ramp_dataset(cfg.data_root, val_regions, cfg.res)
    
    run_stage2(train_dataset, val_dataset, cfg)