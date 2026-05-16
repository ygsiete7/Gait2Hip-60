#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn

from sklearn.model_selection import GroupKFold


# =========================================================
# 0) Reproducibility
# =========================================================
def set_seed(seed: int = 2025) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


GLOBAL_SEED = 2025
set_seed(GLOBAL_SEED)


# =========================================================
# 1) Config
# =========================================================
# Default GitHub repository paths.
# Expected layout:
# Gait2Hip-60/
# ├── data/
# │   └── Gait2Hip_JM60.npz
# └── joint_moment_prediction/
#     ├── Transformer_JM60.py
#     ├── trained_models/
#     └── results/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)
NPZ_PATH = os.path.join(REPO_ROOT, "data", "Gait2Hip_JM60.npz")
SAVE_DIR = os.path.join(BASE_DIR, "trained_models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

TEST_RATIO = 0.2
SPLIT_SEED = 42

N_SPLITS = 5
BATCH_SIZE = 64
MAX_EPOCHS = 300
LR = 1e-4
WD = 1e-5
GRAD_CLIP = 0.5
PATIENCE = 30
MIN_DELTA = 1e-8

# Transformer config
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 2
DIM_FF = 256
DROPOUT = 0.1


# =========================================================
# 2) Load data
# =========================================================
data = np.load(NPZ_PATH, allow_pickle=True)

X = data["X"].astype(np.float32)
Y_raw = data["Y"].astype(np.float32)      # Raw target, unit: Nm
subject_id = data["subject_id"].astype(str)
weight_kg = data["weight_kg"].astype(np.float32)
speed_label = data["speed_label"].astype(str)

in_cols = data["in_cols"].tolist()
out_cols = data["out_cols"].tolist()

N, T, D_in = X.shape
M = Y_raw.shape[2]

assert len(in_cols) == D_in
assert len(out_cols) == M

# Convert raw joint moment targets to body-weight-normalized training targets.
# Raw dataset unit: Nm; model training target unit: Nm/kg.
Y = (Y_raw / weight_kg[:, None, None]).astype(np.float32)

meta = pd.DataFrame({
    "subject_id": subject_id,
    "speed_label": speed_label,
    "weight_kg": weight_kg,
})

print("X shape:", X.shape)
print("Raw Y shape (Nm):", Y_raw.shape)
print("Training target Y shape (Nm/kg):", Y.shape)
print("Subjects:", len(np.unique(subject_id)))


# =========================================================
# 3) Subject-holdout split: POOL vs TEST
# =========================================================
def split_pool_and_test_by_subject(
    subject_ids: np.ndarray,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    unique_subj = np.unique(subject_ids)
    n_subj = len(unique_subj)
    n_test = max(1, int(round(n_subj * test_ratio)))

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_subj)

    test_subjects = unique_subj[:n_test]
    pool_subjects = unique_subj[n_test:]

    test_mask = np.isin(subject_ids, test_subjects)
    pool_mask = np.isin(subject_ids, pool_subjects)

    test_idx = np.where(test_mask)[0]
    pool_idx = np.where(pool_mask)[0]

    print(f"[Split] total subjects={n_subj} | pool={len(pool_subjects)} | test={len(test_subjects)}")
    return pool_idx, test_idx, pool_subjects, test_subjects


pool_idx, test_idx, pool_subjects, test_subjects = split_pool_and_test_by_subject(
    subject_id, test_ratio=TEST_RATIO, seed=SPLIT_SEED
)

X_pool, Y_pool = X[pool_idx], Y[pool_idx]
X_test, Y_test = X[test_idx], Y[test_idx]

meta_pool = meta.iloc[pool_idx].reset_index(drop=True)
meta_test = meta.iloc[test_idx].reset_index(drop=True)

print("POOL trials:", len(meta_pool), "POOL subjects:", len(np.unique(meta_pool["subject_id"])))
print("TEST trials:", len(meta_test), "TEST subjects:", len(np.unique(meta_test["subject_id"])))


# =========================================================
# 4) Dataset / normalization
# =========================================================
class SeqDataset(Dataset):
    def __init__(self, X_: np.ndarray, Y_: Optional[np.ndarray] = None):
        self.X = torch.from_numpy(X_).float()
        self.Y = None if Y_ is None else torch.from_numpy(Y_).float()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        if self.Y is None:
            return self.X[idx]
        return self.X[idx], self.Y[idx]


def compute_x_norm_params(X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu_x = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    sd_x = X_train.reshape(-1, X_train.shape[-1]).std(axis=0) + 1e-8
    return mu_x.astype(np.float32), sd_x.astype(np.float32)


def apply_x_norm(X_: np.ndarray, mu_x: np.ndarray, sd_x: np.ndarray) -> np.ndarray:
    return ((X_ - mu_x) / sd_x).astype(np.float32)


def make_loaders(
    Xtr_n: np.ndarray,
    Ytr: np.ndarray,
    Xva_n: np.ndarray,
    Yva: np.ndarray,
    batch_size: int = 64,
) -> Tuple[DataLoader, DataLoader]:
    dl_tr = DataLoader(SeqDataset(Xtr_n, Ytr), batch_size=batch_size, shuffle=True)
    dl_va = DataLoader(SeqDataset(Xva_n, Yva), batch_size=batch_size, shuffle=False)
    return dl_tr, dl_va


# =========================================================
# 5) Device & Transformer model
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

MAX_LEN = max(500, T + 5)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        self.pos_embed = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        return x + self.pos_embed(pos)


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 500,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        x = self.output_proj(x)
        return x


# =========================================================
# 6) Metrics
# =========================================================
def regression_metrics(Y_true: np.ndarray, Y_pred: np.ndarray, eps: float = 1e-12) -> Dict[str, float]:
    diff = Y_pred - Y_true

    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))

    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((Y_true - Y_true.mean()) ** 2) + eps)
    r2 = float(1.0 - ss_res / ss_tot)

    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def subject_level_perf_arrays(
    Y_true_all: np.ndarray,
    Y_pred_all: np.ndarray,
    subj_ids: np.ndarray,
    subjects_keep: np.ndarray,
    speed_labels: Optional[np.ndarray] = None,
    target_speed: Optional[str] = None,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rmse_arr, mae_arr, r2_arr = [], [], []

    subj_ids = np.asarray(subj_ids).astype(str)
    subjects_keep = np.asarray(subjects_keep).astype(str)
    if speed_labels is not None:
        speed_labels = np.asarray(speed_labels).astype(str)

    for sid in subjects_keep:
        if target_speed is None:
            mask = (subj_ids == sid)
        else:
            mask = (subj_ids == sid) & (speed_labels == str(target_speed))

        if mask.sum() == 0:
            rmse_arr.append(np.nan)
            mae_arr.append(np.nan)
            r2_arr.append(np.nan)
            continue

        Yt = Y_true_all[mask].reshape(-1, Y_true_all.shape[-1])
        Yp = Y_pred_all[mask].reshape(-1, Y_true_all.shape[-1])
        diff = Yp - Yt

        rmse_arr.append(float(np.sqrt(np.mean(diff ** 2))))
        mae_arr.append(float(np.mean(np.abs(diff))))

        ss_res = float(np.sum(diff ** 2))
        ss_tot = float(np.sum((Yt - Yt.mean()) ** 2) + eps)
        r2_arr.append(float(1.0 - ss_res / ss_tot))

    return (
        np.array(rmse_arr, dtype=np.float64),
        np.array(mae_arr, dtype=np.float64),
        np.array(r2_arr, dtype=np.float64),
    )


def mean_ignore_nan(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(x)
    return float(np.mean(x[ok])) if ok.any() else np.nan


def print_metrics(prefix: str, metrics: Dict[str, float]) -> None:
    print(f"{prefix} | RMSE={metrics['RMSE']:.4f} | MAE={metrics['MAE']:.4f} | R2={metrics['R2']:.4f}")


# =========================================================
# 7) Train / predict (MSE only)
# =========================================================
def train_with_early_stopping(
    Xtr_n: np.ndarray,
    Ytr: np.ndarray,
    Xva_n: np.ndarray,
    Yva: np.ndarray,
    max_epochs: int = 300,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    grad_clip: float = 0.5,
    patience: int = 30,
    min_delta: float = 1e-8,
    seed: int = 2025,
    batch_size: int = 64,
) -> Dict[str, object]:
    set_seed(seed)

    dl_tr, dl_va = make_loaders(Xtr_n, Ytr, Xva_n, Yva, batch_size=batch_size)

    model = TransformerModel(
        input_size=Xtr_n.shape[-1],
        output_size=Ytr.shape[-1],
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FF,
        dropout=DROPOUT,
        max_len=MAX_LEN,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_epoch = -1
    best_state = None
    bad_count = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            yhat = model(xb)
            loss = criterion(yhat, yb)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb, yb = xb.to(device), yb.to(device)
                yhat = model(xb)
                val_loss = criterion(yhat, yb)
                val_loss_sum += float(val_loss.item())

        val_loss_mean = val_loss_sum / max(1, len(dl_va))

        if val_loss_mean < best_val - min_delta:
            best_val = val_loss_mean
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_count = 0
        else:
            bad_count += 1
            if bad_count >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    preds = []
    with torch.no_grad():
        dl_va2 = DataLoader(SeqDataset(Xva_n, Yva), batch_size=batch_size, shuffle=False)
        for xb, _ in dl_va2:
            xb = xb.to(device)
            preds.append(model(xb).detach().cpu().numpy())

    Y_pred_val = np.concatenate(preds, axis=0)

    return {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "best_state": best_state,
        "Y_pred_val": Y_pred_val,
    }


def train_fixed_epochs(
    Xtr_n: np.ndarray,
    Ytr: np.ndarray,
    epochs: int,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    grad_clip: float = 0.5,
    seed: int = 2025,
    batch_size: int = 64,
) -> nn.Module:
    set_seed(seed)

    dl_tr = DataLoader(SeqDataset(Xtr_n, Ytr), batch_size=batch_size, shuffle=True)

    model = TransformerModel(
        input_size=Xtr_n.shape[-1],
        output_size=Ytr.shape[-1],
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FF,
        dropout=DROPOUT,
        max_len=MAX_LEN,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    for _ in range(int(epochs)):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            yhat = model(xb)
            loss = criterion(yhat, yb)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

    return model


@torch.no_grad()
def predict_numpy(model: nn.Module, Xn: np.ndarray, batch_size: int = 64) -> np.ndarray:
    model.eval()
    dl = DataLoader(SeqDataset(Xn, None), batch_size=batch_size, shuffle=False)

    preds = []
    for xb in dl:
        xb = xb.to(device)
        yhat = model(xb).detach().cpu().numpy()
        preds.append(yhat)

    return np.concatenate(preds, axis=0)


# =========================================================
# 8) 5-fold benchmark on POOL
# =========================================================
groups_pool = meta_pool["subject_id"].to_numpy()
gkf = GroupKFold(n_splits=N_SPLITS)

cv_rows: List[Dict[str, object]] = []
best_epochs: List[int] = []

print("\n" + "=" * 90)
print("5-fold GroupKFold benchmark on POOL (Transformer baseline, MSE only, moment prediction)")
print("=" * 90)

for fold_id, (tr_idx, va_idx) in enumerate(gkf.split(X_pool, Y_pool, groups=groups_pool), start=1):
    print("\n" + "-" * 90)
    print(f"[Fold {fold_id}/{N_SPLITS}] train_trials={len(tr_idx)} val_trials={len(va_idx)}")
    print("-" * 90)

    X_tr, Y_tr = X_pool[tr_idx], Y_pool[tr_idx]
    X_va, Y_va = X_pool[va_idx], Y_pool[va_idx]

    meta_va = meta_pool.iloc[va_idx].reset_index(drop=True)
    subj_va = meta_va["subject_id"].to_numpy()
    subjects_keep = np.unique(subj_va)

    mu_x, sd_x = compute_x_norm_params(X_tr)
    X_tr_n = apply_x_norm(X_tr, mu_x, sd_x)
    X_va_n = apply_x_norm(X_va, mu_x, sd_x)

    rec = train_with_early_stopping(
        X_tr_n, Y_tr, X_va_n, Y_va,
        max_epochs=MAX_EPOCHS,
        lr=LR,
        weight_decay=WD,
        grad_clip=GRAD_CLIP,
        patience=PATIENCE,
        min_delta=MIN_DELTA,
        seed=GLOBAL_SEED,
        batch_size=BATCH_SIZE,
    )

    Y_va_pred = rec["Y_pred_val"]
    best_epoch = int(rec["best_epoch"])
    best_epochs.append(best_epoch)

    overall = regression_metrics(Y_va, Y_va_pred)

    rmse_arr, mae_arr, r2_arr = subject_level_perf_arrays(Y_va, Y_va_pred, subj_va, subjects_keep)
    subj_mean_rmse = mean_ignore_nan(rmse_arr)
    subj_mean_mae = mean_ignore_nan(mae_arr)
    subj_mean_r2 = mean_ignore_nan(r2_arr)

    row = {
        "fold": fold_id,
        "n_train_trials": int(len(tr_idx)),
        "n_val_trials": int(len(va_idx)),
        "n_val_subjects": int(len(subjects_keep)),
        "best_epoch": best_epoch,
        "best_val_loss": float(rec["best_val_loss"]),
        "val_RMSE": overall["RMSE"],
        "val_MAE": overall["MAE"],
        "val_R2": overall["R2"],
        "val_subject_mean_RMSE": subj_mean_rmse,
        "val_subject_mean_MAE": subj_mean_mae,
        "val_subject_mean_R2": subj_mean_r2,
    }
    cv_rows.append(row)

    print(
        f"[Fold {fold_id}] "
        f"best_epoch={best_epoch} | "
        f"RMSE={overall['RMSE']:.4f} | "
        f"MAE={overall['MAE']:.4f} | "
        f"R2={overall['R2']:.4f} | "
        f"subj_mean_RMSE={subj_mean_rmse:.4f} | "
        f"subj_mean_MAE={subj_mean_mae:.4f} | "
        f"subj_mean_R2={subj_mean_r2:.4f}"
    )

df_cv = pd.DataFrame(cv_rows)
cv_csv_path = os.path.join(RESULTS_DIR, "Transformer_JM60_cv_5fold_baseline_metrics.csv")
df_cv.to_csv(cv_csv_path, index=False, float_format="%.4f")

print("\nSaved 5-fold CV metrics to:", cv_csv_path)
print("\n5-fold summary:")
print(df_cv[
    [
        "best_epoch",
        "val_RMSE",
        "val_MAE",
        "val_R2",
        "val_subject_mean_RMSE",
        "val_subject_mean_MAE",
        "val_subject_mean_R2",
    ]
].mean(numeric_only=True))

final_epoch = int(np.median(best_epochs))
final_epoch = max(1, final_epoch)

print(f"\nChosen final_epoch = median(best_epochs) = {final_epoch}")


# =========================================================
# 9) Train final model on full POOL
# =========================================================
mu_pool, sd_pool = compute_x_norm_params(X_pool)
X_pool_n = apply_x_norm(X_pool, mu_pool, sd_pool)
X_test_n = apply_x_norm(X_test, mu_pool, sd_pool)

print("\n" + "=" * 90)
print("Training final baseline model on FULL POOL")
print("=" * 90)

final_model = train_fixed_epochs(
    X_pool_n, Y_pool,
    epochs=final_epoch,
    lr=LR,
    weight_decay=WD,
    grad_clip=GRAD_CLIP,
    seed=GLOBAL_SEED,
    batch_size=BATCH_SIZE,
)


# =========================================================
# 10) Evaluate on independent TEST
# =========================================================
Y_test_pred = predict_numpy(final_model, X_test_n, batch_size=BATCH_SIZE)

test_overall = regression_metrics(Y_test, Y_test_pred)
print_metrics("TEST overall", test_overall)

subj_test = meta_test["subject_id"].to_numpy()
subjects_test_keep = np.unique(subj_test)
speed_labels_test = meta_test["speed_label"].to_numpy()

rmse_test_subj, mae_test_subj, r2_test_subj = subject_level_perf_arrays(
    Y_test, Y_test_pred, subj_test, subjects_test_keep
)

test_subject_mean = {
    "RMSE": mean_ignore_nan(rmse_test_subj),
    "MAE": mean_ignore_nan(mae_test_subj),
    "R2": mean_ignore_nan(r2_test_subj),
}
print_metrics("TEST subject-mean", test_subject_mean)

speed_rows = []
for spd in np.unique(speed_labels_test):
    rmse_arr_spd, mae_arr_spd, r2_arr_spd = subject_level_perf_arrays(
        Y_test,
        Y_test_pred,
        subj_test,
        subjects_test_keep,
        speed_labels=speed_labels_test,
        target_speed=spd,
    )

    metrics_spd = {
        "RMSE": mean_ignore_nan(rmse_arr_spd),
        "MAE": mean_ignore_nan(mae_arr_spd),
        "R2": mean_ignore_nan(r2_arr_spd),
    }

    n_trials_spd = int((speed_labels_test == spd).sum())
    n_subjects_spd = int(np.sum([
        np.any((subj_test == sid) & (speed_labels_test == spd))
        for sid in subjects_test_keep
    ]))

    speed_rows.append({
        "speed": spd,
        "n_trials": n_trials_spd,
        "n_subjects": n_subjects_spd,
        "subject_mean_RMSE": metrics_spd["RMSE"],
        "subject_mean_MAE": metrics_spd["MAE"],
        "subject_mean_R2": metrics_spd["R2"],
    })

    print(
        f"TEST {spd} subject-mean | "
        f"RMSE={metrics_spd['RMSE']:.4f} | "
        f"MAE={metrics_spd['MAE']:.4f} | "
        f"R2={metrics_spd['R2']:.4f}"
    )

overall_row = {
    "speed": "overall",
    "n_trials": len(meta_test),
    "n_subjects": len(subjects_test_keep),
    "subject_mean_RMSE": test_subject_mean["RMSE"],
    "subject_mean_MAE": test_subject_mean["MAE"],
    "subject_mean_R2": test_subject_mean["R2"],
}

all_rows = [overall_row] + speed_rows
df_all = pd.DataFrame(all_rows)

speed_csv_path = os.path.join(RESULTS_DIR, "Transformer_JM60_test_metrics_by_speed_mean.csv")
df_all.to_csv(speed_csv_path, index=False, float_format="%.4f")


# =========================================================
# 11) Save final checkpoint
# =========================================================
ckpt_path = os.path.join(SAVE_DIR, "Transformer_JM60.pt")

torch.save({
    "model_type": "transformer",
    "target_unit": "Nm/kg",
    "raw_target_unit": "Nm",
    "alpha": 0.0,
    "r": 0,
    "synergy_loss_mode": None,
    "state_dict": final_model.state_dict(),
    "mu_x": mu_pool,
    "sd_x": sd_pool,
    "in_cols": in_cols,
    "out_cols": out_cols,
    "hparams": {
        "d_model": D_MODEL,
        "nhead": NHEAD,
        "num_layers": NUM_LAYERS,
        "dim_feedforward": DIM_FF,
        "dropout": DROPOUT,
        "d_in": D_in,
        "d_out": M,
        "T": T,
        "max_len": MAX_LEN,
    },
    "protocol": "Subject-holdout TEST + 5-fold GroupKFold benchmark on POOL; baseline Transformer; MSE only; moment prediction in Nm/kg; report subject-mean metrics; raw target unit=Nm; model target unit=Nm/kg",
    "global_seed": GLOBAL_SEED,
    "test_subjects": test_subjects.tolist(),
    "cv_metrics_csv": cv_csv_path,
    "test_metrics_overall": test_overall,
    "test_metrics_subject_mean": test_subject_mean,
    "test_metrics_by_speed_subject_mean_csv": speed_csv_path,
}, ckpt_path)

print("\nSaved final checkpoint to:", ckpt_path)
print("Saved subject-mean by-speed CSV to:", speed_csv_path)
print("Done.")