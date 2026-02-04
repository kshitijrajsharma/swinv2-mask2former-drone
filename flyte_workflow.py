import os
from dataclasses import dataclass
from pathlib import Path

from flytekit import ImageSpec, Resources, task, workflow
from flytekit.types.directory import FlyteDirectory

image = ImageSpec(
    name="mask2former-training",
    base_image="ghcr.io/${{ github.repository }}:latest",
    registry="ghcr.io",
)


@dataclass
class DatasetConfig:
    drone_image_id: str = "62d86c65d8499800053796c4"
    zoom_level: int = 19
    chip_size_px: int = 512
    train_bbox: list[float] = None
    val_bbox: list[float] = None
    test_bbox: list[float] = None

    def __post_init__(self):
        if self.train_bbox is None:
            self.train_bbox = [85.51176609880189, 27.625518932561256, 85.52513148143508, 27.63551883131749]
        if self.val_bbox is None:
            self.val_bbox = [85.51883176039746, 27.63560, 85.52308324197179, 27.63833629629815]
        if self.test_bbox is None:
            self.test_bbox = [85.53039880381334, 27.62456651360527, 85.53606027956683, 27.629042810653335]


@task(
    requests=Resources(cpu="4", mem="8Gi"),
    cache=True,
    cache_version="1.0",
    container_image=image,
)
def download_data(config: DatasetConfig, work_dir: str = "/tmp/data/banepa") -> FlyteDirectory:
    from torchgeo.datasets import OpenAerialMap, OpenStreetMap
    
    os.makedirs(work_dir, exist_ok=True)
    
    datasets = [
        ("train", config.train_bbox),
        ("val", config.val_bbox),
        ("test", config.test_bbox),
    ]
    
    osm_classes = [{"name": "building", "selector": [{"building": "*"}]}]
    
    for folder, bbox in datasets:
        print(f"Downloading {folder} data...")
        
        oam_path = os.path.join(work_dir, folder, "source")
        os.makedirs(oam_path, exist_ok=True)
        
        oam_dataset = OpenAerialMap(
            paths=oam_path,
            bbox=bbox,
            zoom=config.zoom_level,
            download=True,
            image_id=config.drone_image_id,
            tile_size=config.chip_size_px,
        )
        
        osm_path = os.path.join(work_dir, folder, "labels")
        os.makedirs(osm_path, exist_ok=True)
        
        osm_dataset = OpenStreetMap(
            paths=osm_path,
            bbox=bbox,
            classes=osm_classes,
            download=True,
        )
        
        print(f"{folder} data downloaded")
    
    return FlyteDirectory(path=work_dir)


@task(
    requests=Resources(cpu="8", mem="16Gi", gpu="1"),
    cache=False,
    container_image=image,
)
def train_model(
    data_dir: FlyteDirectory,
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 1e-5,
    use_wandb: bool = False,
    wandb_project: str = "building-seg-mask2former",
    wandb_run_name: str = "flyte_run",
) -> FlyteDirectory:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import WandbLogger
    
    from src.config import Config
    from src.stage1 import Mask2FormerModule, OAMDataModule
    from src.utils import set_seed
    
    cfg = Config()
    cfg.data_root = Path(data_dir.path)
    cfg.output_dir = Path("/tmp/outputs")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    
    cfg.train_regions = ["train"]
    cfg.val_regions = ["val"]
    cfg.test_regions = ["test"]
    
    cfg.epochs = epochs
    cfg.batch_size = batch_size
    cfg.learning_rate = learning_rate
    cfg.use_wandb = use_wandb
    cfg.wandb_project = wandb_project
    cfg.wandb_run_name = wandb_run_name
    cfg.verbose = True
    
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
    
    logger = WandbLogger(project=cfg.wandb_project, name=cfg.wandb_run_name) if cfg.use_wandb else None
    
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
    
    return FlyteDirectory(path=str(cfg.output_dir))


@workflow
def mask2former_training_workflow(
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 1e-5,
    use_wandb: bool = False,
    wandb_project: str = "building-seg-mask2former",
    wandb_run_name: str = "flyte_compact_test",
) -> FlyteDirectory:
    config = DatasetConfig()
    data_dir = download_data(config=config)
    output_dir = train_model(
        data_dir=data_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
    )
    return output_dir


if __name__ == "__main__":
    result = mask2former_training_workflow(epochs=5, batch_size=8, use_wandb=False)
    print(f"Training completed. Output: {result}")
