import torch
from torch.utils.data import DataLoader
from torchgeo.samplers import RandomBatchGeoSampler
from tqdm import tqdm
from transformers import Mask2FormerForUniversalSegmentation

from config import Config
from utils import (
    CombinedLoss,
    EarlyStopping,
    collate_fn_mask2former,
    compute_num_queries,
    get_all_ramp_regions,
    get_ramp_dataset,
    save_checkpoint,
    set_seed,
    split_regions,
)


def train_epoch(model, dataloader, criterion, optimizer, device, grad_accum_steps=1):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc="Training")
    for idx, batch in enumerate(pbar):
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
        loss = loss / grad_accum_steps
        loss.backward()
        
        if (idx + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * grad_accum_steps
        pbar.set_postfix({"loss": f"{loss.item() * grad_accum_steps:.4f}"})
    
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


def main():
    cfg = Config()
    set_seed(cfg.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading RAMP dataset...")
    all_regions = get_all_ramp_regions(cfg.data_root)
    
    if cfg.train_regions and cfg.val_regions:
        train_regions = cfg.train_regions
        val_regions = cfg.val_regions
        print(f"Using manual region split:")
        print(f"  Train: {train_regions}")
        print(f"  Val: {val_regions}")
    else:
        train_regions, val_regions = split_regions(all_regions, cfg.val_split, cfg.seed)
        print(f"Using automatic region split (val_split={cfg.val_split}):")
        print(f"  Train regions: {train_regions}")
        print(f"  Val regions: {val_regions}")
    
    train_dataset = get_ramp_dataset(cfg.data_root, train_regions, cfg.res)
    val_dataset = get_ramp_dataset(cfg.data_root, val_regions, cfg.res)
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    if cfg.num_queries == 0:
        cfg.num_queries = compute_num_queries(train_dataset, num_samples=100)
    else:
        print(f"Using configured num_queries: {cfg.num_queries}")
    
    train_sampler = RandomBatchGeoSampler(
        train_dataset,
        size=256,
        batch_size=cfg.stage1_batch_size,
        length=cfg.stage1_num_samples // cfg.stage1_batch_size,
    )
    
    val_sampler = RandomBatchGeoSampler(
        val_dataset,
        size=256,
        batch_size=cfg.stage1_batch_size,
        length=len(val_dataset) // cfg.stage1_batch_size,
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
    
    print("Initializing Mask2Former...")
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        cfg.pretrained_model,
        id2label={0: "background", 1: "building"},
        num_labels=2,
        num_queries=cfg.num_queries,
        ignore_mismatched_sizes=True,
    )
    model = model.to(device)
    
    criterion = CombinedLoss(
        ce_weight=cfg.loss_ce_weight,
        dice_weight=cfg.loss_dice_weight,
        focal_weight=cfg.loss_focal_weight,
        hausdorff_weight=cfg.loss_hausdorff_weight,
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.stage1_lr,
        weight_decay=cfg.stage1_weight_decay,
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.stage1_epochs,
    )
    
    early_stopping = EarlyStopping(patience=cfg.early_stopping_patience)
    
    print("\nStage 1: Foundation Training")
    print(f"Loss weights - CE: {cfg.loss_ce_weight}, Dice: {cfg.loss_dice_weight}, Focal: {cfg.loss_focal_weight}, Hausdorff: {cfg.loss_hausdorff_weight}")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float("inf")
    
    for epoch in range(cfg.stage1_epochs):
        print(f"\nEpoch {epoch + 1}/{cfg.stage1_epochs}")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, cfg.gradient_accumulation_steps)
        val_loss = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = cfg.output_dir / "stage1_best.pt"
            save_checkpoint(model, optimizer, epoch, val_loss, best_path)
            print(f"Saved best checkpoint: {best_path}")
        
        if (epoch + 1) % cfg.checkpoint_freq == 0:
            ckpt_path = cfg.output_dir / f"stage1_epoch_{epoch + 1}.pt"
            save_checkpoint(model, optimizer, epoch, val_loss, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")
        
        if early_stopping(val_loss):
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break
    
    print(f"\nStage 1 complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()