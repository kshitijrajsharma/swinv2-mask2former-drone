import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchgeo.samplers import RandomGeoSampler
from tqdm import tqdm
from transformers import Mask2FormerForUniversalSegmentation

from config import Config
from utils import (
    CombinedLoss,
    EarlyStopping,
    collate_fn_mask2former,
    compute_num_queries_from_geojson,
    get_all_ramp_regions,
    get_metrics,
    get_ramp_dataset,
    save_checkpoint,
    set_seed,
    split_regions,
)


def pad_targets_to_queries(targets, num_queries, device):
    gt_masks = []
    gt_labels = []
    
    for t in targets:
        masks = t["masks"]
        labels = t["class_labels"]
        num_instances = masks.shape[0]
        
        if num_instances < num_queries:
            pad_size = num_queries - num_instances
            masks = torch.cat([masks, torch.zeros(pad_size, *masks.shape[1:], device=device, dtype=masks.dtype)])
            labels = torch.cat([labels, torch.zeros(pad_size, device=device, dtype=labels.dtype)])
        elif num_instances > num_queries:
            masks = masks[:num_queries]
            labels = labels[:num_queries]
        
        gt_masks.append(masks)
        gt_labels.append(labels)
    
    return torch.stack(gt_masks), torch.stack(gt_labels)


def train_epoch(model, dataloader, criterion, optimizer, device, metrics, grad_accum_steps=1):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    num_queries = model.config.num_queries
    metrics.reset()
    
    pbar = tqdm(dataloader, desc="Training")
    for idx, batch in enumerate(pbar):
        images = batch["pixel_values"].to(device)
        targets = batch["mask_labels"]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        outputs = model(pixel_values=images, output_hidden_states=True)
        
        pred_masks = outputs.masks_queries_logits
        pred_logits = outputs.class_queries_logits
        
        gt_masks, gt_labels = pad_targets_to_queries(targets, num_queries, device)
        
        loss = criterion(pred_masks, pred_logits, gt_masks, gt_labels)
        loss = loss / grad_accum_steps
        loss.backward()
        
        if (idx + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * grad_accum_steps
        
        with torch.no_grad():
            B, Q, H, W = pred_masks.shape
            if gt_masks.shape[-2:] != (H, W):
                gt_masks_resized = F.interpolate(gt_masks.float(), size=(H, W), mode='nearest').bool()
            else:
                gt_masks_resized = gt_masks
            
            building_mask = (gt_labels == 1)
            if building_mask.sum() > 0:
                pred_probs = torch.sigmoid(pred_masks[building_mask])
                gt_building = gt_masks_resized[building_mask]
                metrics.update(pred_probs.flatten(), gt_building.flatten().int())
        
        lr = optimizer.param_groups[0]['lr']
        mem = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        pbar.set_postfix({'loss': f'{loss.item() * grad_accum_steps:.4f}', 'lr': f'{lr:.2e}', 'mem': f'{mem:.1f}G'})
    
    epoch_metrics = metrics.compute()
    return total_loss / len(dataloader), epoch_metrics


@torch.no_grad()
def validate_epoch(model, dataloader, criterion, device, metrics):
    model.eval()
    total_loss = 0.0
    num_queries = model.config.num_queries
    metrics.reset()
    
    pbar = tqdm(dataloader, desc="Validation")
    for idx, batch in enumerate(pbar):
        images = batch["pixel_values"].to(device)
        targets = batch["mask_labels"]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        outputs = model(pixel_values=images, output_hidden_states=True)
        
        pred_masks = outputs.masks_queries_logits
        pred_logits = outputs.class_queries_logits
        
        gt_masks, gt_labels = pad_targets_to_queries(targets, num_queries, device)
        
        loss = criterion(pred_masks, pred_logits, gt_masks, gt_labels)
        total_loss += loss.item()
        
        B, Q, H, W = pred_masks.shape
        if gt_masks.shape[-2:] != (H, W):
            gt_masks_resized = F.interpolate(gt_masks.float(), size=(H, W), mode='nearest').bool()
        else:
            gt_masks_resized = gt_masks
        
        building_mask = (gt_labels == 1)
        if building_mask.sum() > 0:
            pred_probs = torch.sigmoid(pred_masks[building_mask])
            gt_building = gt_masks_resized[building_mask]
            metrics.update(pred_probs.flatten(), gt_building.flatten().int())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_metrics = metrics.compute()
    return total_loss / len(dataloader), epoch_metrics


def create_dataloader(dataset, batch_size, num_samples, num_workers, is_train=True):
    sampler = RandomGeoSampler(dataset, size=256, length=num_samples)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn_mask2former,
        pin_memory=True,
        drop_last=is_train,
    )
    
    return loader


def main():
    cfg = Config()
    set_seed(cfg.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    print("\nLoading RAMP dataset...")
    all_regions = get_all_ramp_regions(cfg.data_root)
    
    if cfg.train_regions and cfg.val_regions:
        train_regions = cfg.train_regions
        val_regions = cfg.val_regions
        print(f"Manual split - Train: {train_regions}, Val: {val_regions}")
    else:
        train_regions, val_regions = split_regions(all_regions, cfg.val_split, cfg.seed)
        print(f"Auto split (val={cfg.val_split}) - Train: {train_regions}, Val: {val_regions}")
    
    train_dataset, train_label_paths = get_ramp_dataset(cfg.data_root, train_regions, cfg.res)
    val_dataset, val_label_paths = get_ramp_dataset(cfg.data_root, val_regions, cfg.res)
    
    print(f"\nDataset info:")
    print(f"  Train bounds: {train_dataset.bounds}")
    print(f"  Val bounds: {val_dataset.bounds}")
    print(f"  Train CRS: {train_dataset.crs}")
    
    if cfg.num_queries == 0:
        print("\nComputing num_queries from GeoJSON...")
        cfg.num_queries = compute_num_queries_from_geojson(train_label_paths)
    else:
        print(f"Using configured num_queries: {cfg.num_queries}")
    
    train_samples = cfg.stage1_batch_size * cfg.stage1_max_batches_per_epoch
    val_samples = cfg.stage1_batch_size * max(10, cfg.stage1_max_batches_per_epoch // 5)
    
    print(f"\nCreating dataloaders: train_samples={train_samples}, val_samples={val_samples}")
    
    train_loader = create_dataloader(
        train_dataset, 
        cfg.stage1_batch_size, 
        train_samples,
        cfg.num_workers,
        is_train=True
    )
    
    val_loader = create_dataloader(
        val_dataset,
        cfg.stage1_batch_size,
        val_samples,
        cfg.num_workers,
        is_train=False
    )
    
    print("\nInitializing Mask2Former...")
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        cfg.pretrained_model,
        id2label={0: "background", 1: "building"},
        num_labels=2,
        num_queries=cfg.num_queries,
        ignore_mismatched_sizes=True,
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {total_params:,} total, {trainable_params:,} trainable")
    
    criterion = CombinedLoss(
        ce_weight=cfg.loss_ce_weight,
        dice_weight=cfg.loss_dice_weight,
        focal_weight=cfg.loss_focal_weight,
        bce_weight=cfg.loss_bce_weight,
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
    
    train_metrics = get_metrics(device)
    val_metrics = get_metrics(device)
    
    print(f"\nStage 1 Training: BS={cfg.stage1_batch_size}, Accum={cfg.gradient_accumulation_steps}, EffectiveBS={cfg.stage1_batch_size * cfg.gradient_accumulation_steps}")
    print(f"Loss: CE={cfg.loss_ce_weight}, Dice={cfg.loss_dice_weight}, Focal={cfg.loss_focal_weight}, BCE={cfg.loss_bce_weight}, Hausdorff={cfg.loss_hausdorff_weight}")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float("inf")
    
    for epoch in range(cfg.stage1_epochs):
        print(f"\nEpoch {epoch + 1}/{cfg.stage1_epochs}")
        
        train_loss, train_m = train_epoch(model, train_loader, criterion, optimizer, device, train_metrics, cfg.gradient_accumulation_steps)
        val_loss, val_m = validate_epoch(model, val_loader, criterion, device, val_metrics)
        scheduler.step()
        
        print(f"Train: Loss={train_loss:.4f} Acc={train_m['accuracy']:.4f} P={train_m['precision']:.4f} R={train_m['recall']:.4f} F1={train_m['f1']:.4f}")
        print(f"Val:   Loss={val_loss:.4f} Acc={val_m['accuracy']:.4f} P={val_m['precision']:.4f} R={val_m['recall']:.4f} F1={val_m['f1']:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = cfg.output_dir / "stage1_best.pt"
            save_checkpoint(model, optimizer, epoch, val_loss, best_path)
            print(f"Saved best: {best_path}")
        
        if (epoch + 1) % cfg.checkpoint_freq == 0:
            ckpt_path = cfg.output_dir / f"stage1_epoch_{epoch + 1}.pt"
            save_checkpoint(model, optimizer, epoch, val_loss, ckpt_path)
            print(f"Checkpoint: {ckpt_path}")
        
        if early_stopping(val_loss):
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    print(f"\nStage 1 complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()