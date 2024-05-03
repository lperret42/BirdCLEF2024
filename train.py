import os
import json
from typing import Dict
from datetime import datetime
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
import timm
from adan import Adan
from torch.optim.lr_scheduler import ExponentialLR
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from src.dataset import BirdDataset, collate_fn
from src.wave2spec import MelSpectrogram
from src.dl_wrapper import DataLoaderWrapper
from src.lightning_model import BirdModel
from src.loss import FocalLoss
from src.utils import set_seed, add_weight_column


class CFG:
    """
    Configuration class to store hyperparameters and dataset information.
    """
    rootpath: str = "/data/bird-datasets/birdclef-2024"
    trainpath: str = os.path.join(rootpath, "train_metadata_with_duration.csv")
    audio_dir: str = os.path.join(rootpath, "train_audio")
    extended_only_for_train: bool = False
    cache_audio_dir: str = '/data/bird-datasets/train_audio_as_tensor'
    seed: int = 42
    nfolds: int = 5
    batch_size: int = 32
    num_workers: int = 30
    val_batch_size: int = 64
    epochs: int = 60
    patience: int = 10
    lr: float = 5e-4
    gamma: float = 0.9
    mixup_prob: float = 0.7
    mixup_max_num: int = 2
    mixup_agg: str = 'mean'
    drop_rate: float = 0.1
    sr: int = 32000
    train_duration: int = 5
    val_duration: int = 5
    train_split: str = 'random'
    val_split: str = 'first'
    spec_args: Dict[str, int] = {
        'f_min': 0,
        'f_max': 16000,
        'n_fft': 2048,
        'win_length': 626,
        'hop_length': 626,
        'n_mels': 256,
    }
    train_use_weight: bool = True
    model_name: str = "repvit_m2.dist_in1k"
    output_dir: str = f'{model_name}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'


# Set seed for reproducibility
set_seed(CFG.seed)

# Data loading and preparation
df = pd.read_csv(CFG.trainpath)
df = df.drop_duplicates('filename').reset_index(drop=True)

if CFG.train_use_weight:
    df = add_weight_column(df)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df['row_id'] = df['filename']
# Label encoding
labels = sorted(df['primary_label'].unique())
label_to_num = {label: i for i, label in enumerate(labels)}
df['target'] = df['primary_label'].map(label_to_num)

# Stratified K-Fold Cross-Validation setup
skf = StratifiedKFold(n_splits=CFG.nfolds, shuffle=True, random_state=CFG.seed)
df["fold"] = -1
for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['target'])):
    df.loc[val_idx, "fold"] = fold

# Audio processing setup
wave2spec_obj = MelSpectrogram(device=device, **CFG.spec_args)

# Model and training setup
model = timm.create_model(CFG.model_name, in_chans=1, drop_rate=CFG.drop_rate, pretrained=True, num_classes=len(labels))
optimizer = Adan(model.parameters(), lr=CFG.lr)
lr_scheduler = ExponentialLR(optimizer, gamma=CFG.gamma)
loss_fn = FocalLoss()

# Directory and configuration file setup
os.makedirs(CFG.output_dir, exist_ok=False)
config_dict = {k: v for k, v in CFG.__dict__.items() if not k.startswith('__')}
with open(os.path.join(CFG.output_dir, 'config.json'), 'w') as f:
    json.dump(config_dict, f, indent=4)

# Cross-validation training loop
for fold in sorted(df['fold'].unique()):
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    val_df = df[df["fold"] == fold].reset_index(drop=True)

    train_ds = BirdDataset(train_df, CFG.audio_dir, num_classes=len(labels), sr=CFG.sr, duration=CFG.train_duration, split=CFG.train_split, mode='train', mixup_prob=CFG.mixup_prob, mixup_max_num=CFG.mixup_max_num, cache_audio_dir=CFG.cache_audio_dir)
    val_ds = BirdDataset(val_df, CFG.audio_dir, num_classes=len(labels), sr=CFG.sr, duration=CFG.val_duration, split=CFG.val_split, mode='valid', cache_audio_dir=CFG.cache_audio_dir)

    train_dl = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=CFG.val_batch_size, shuffle=False, num_workers=CFG.num_workers, collate_fn=collate_fn)
    train_dl = DataLoaderWrapper(train_dl, wave2spec_obj)
    val_dl = DataLoaderWrapper(val_dl, wave2spec_obj)

    # Initialize the PyTorch Lightning model
    lightning_model = BirdModel(model, optimizer, lr_scheduler, loss_fn)

    # Setup checkpoints and early stopping
    ckpt_path = os.path.join(CFG.output_dir, f'model_{fold}.ckpt')
    monitor = 'val_roc_auc'
    checkpoint_callback = ModelCheckpoint(monitor=monitor, dirpath=CFG.output_dir, filename=f'model_{fold}', save_top_k=1, mode='max', save_weights_only=True)
    early_stop_callback = EarlyStopping(monitor=monitor, patience=CFG.patience, verbose=True, mode='max')

    # Initialize trainer and start training
    trainer = Trainer(max_epochs=CFG.epochs, callbacks=[checkpoint_callback, early_stop_callback], precision='bf16-mixed', deterministic=True, num_sanity_val_steps=0)
    trainer.fit(lightning_model, train_dl, val_dl)
