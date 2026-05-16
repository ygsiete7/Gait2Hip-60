#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run prediction with trained Gait2Hip-60 JM60 models and save results as CSV.

This script loads a trained LSTM, Mamba, or Transformer checkpoint and predicts
sequence-level outputs from the NPZ input array X.

Important unit convention:
- The original NPZ target unit is Nm.
- The model was trained on body-weight-normalized targets, so the default
  prediction unit is Nm/kg.
- Use --save_raw_unit to additionally save predictions restored to Nm.

The script saves CSV files only. It does not generate NPZ prediction files.
"""

from pathlib import Path
from typing import Dict, List

import argparse

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


# =========================================================
# 1) Default paths
# =========================================================
# Expected repository layout:
# Gait2Hip-60/
# ├── data/
# │   └── Gait2Hip_JM60.npz
# └── joint_moment_prediction/
#     ├── predict.py
#     └── trained_models/
#         ├── LSTM_JM60.pt
#         ├── Mamba_JM60.pt
#         └── Transformer_JM60.pt

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent

DEFAULT_NPZ_PATH = REPO_ROOT / "data" / "Gait2Hip_JM60.npz"
DEFAULT_MODEL_PATH = THIS_DIR / "trained_models" / "Transformer_JM60.pt"
DEFAULT_OUTPUT_DIR = THIS_DIR / "predictions"

TASK_NAME = "right hip joint moment prediction"
DATASET_ID = "JM60"
RAW_UNIT = "Nm"
MODEL_UNIT = "Nm/kg"


# =========================================================
# 2) Utilities
# =========================================================
def decode_columns(columns) -> List[str]:
    """Convert NPZ/checkpoint column names to a standard Python string list."""
    return [col.decode("utf-8") if isinstance(col, bytes) else str(col) for col in columns]


def unit_suffix(unit: str) -> str:
    """Convert a unit string into a safe column-name suffix."""
    return unit.replace("/", "_per_").replace(" ", "")


def load_checkpoint(model_path: Path, device: torch.device) -> Dict:
    """Load a PyTorch checkpoint with compatibility across PyTorch versions."""
    try:
        return torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(model_path, map_location=device)


def infer_model_type(model_path: Path, checkpoint: Dict) -> str:
    """Infer model type from checkpoint metadata, filename, or state_dict keys."""
    if "model_type" in checkpoint:
        model_type = str(checkpoint["model_type"]).lower()
        if model_type in {"lstm", "mamba", "transformer"}:
            return model_type

    name = model_path.name.lower()
    if "lstm" in name:
        return "lstm"
    if "mamba" in name:
        return "mamba"
    if "transformer" in name:
        return "transformer"

    state_dict = checkpoint.get("state_dict", checkpoint)
    keys = list(state_dict.keys())
    if any(k.startswith("lstm.") for k in keys):
        return "lstm"
    if any(k.startswith("blocks.") for k in keys):
        return "mamba"
    if any(k.startswith("encoder.") or k.startswith("pos_encoder.") for k in keys):
        return "transformer"

    raise ValueError(
        "Unable to infer model type. Please make sure the checkpoint filename contains "
        "LSTM, Mamba, or Transformer."
    )


def apply_x_norm(X: np.ndarray, mu_x: np.ndarray, sd_x: np.ndarray) -> np.ndarray:
    """Apply the input normalization parameters stored in the checkpoint."""
    return ((X - mu_x) / sd_x).astype(np.float32)


def regression_metrics(Y_true: np.ndarray, Y_pred: np.ndarray, eps: float = 1e-12) -> Dict[str, float]:
    """Compute overall regression metrics after flattening all trials and time steps."""
    diff = Y_pred - Y_true
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((Y_true - Y_true.mean()) ** 2) + eps)
    r2 = float(1.0 - ss_res / ss_tot)
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


# =========================================================
# 3) Dataset
# =========================================================
class SeqDataset(Dataset):
    """Simple sequence dataset for inference."""

    def __init__(self, X: np.ndarray):
        self.X = torch.from_numpy(X).float()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.X[idx]


# =========================================================
# 4) Model definitions
# =========================================================
class LSTMModel(nn.Module):
    """LSTM baseline used for Gait2Hip-60 sequence prediction."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out)


class PositionalEncoding(nn.Module):
    """Learnable positional embedding used by the Transformer baseline."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        self.pos_embed = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        return x + self.pos_embed(pos)


class TransformerModel(nn.Module):
    """Transformer encoder baseline used for Gait2Hip-60 sequence prediction."""

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
        return self.output_proj(x)


class StackedMambaModel(nn.Module):
    """Stacked Mamba baseline used for Gait2Hip-60 sequence prediction."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        d_model: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except ImportError as exc:
            raise ImportError(
                "The selected checkpoint is a Mamba model, but mamba_ssm is not installed. "
                "Please install mamba_ssm before running Mamba prediction."
            ) from exc

        self.input_proj = nn.Linear(input_size, d_model)
        self.blocks = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(layers)
        ])
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = x + self.drop(block(x))
        x = self.norm(x)
        return self.output_proj(x)


def build_model(model_type: str, hparams: Dict) -> nn.Module:
    """Rebuild the model architecture from checkpoint hyperparameters."""
    d_in = int(hparams["d_in"])
    d_out = int(hparams["d_out"])

    if model_type == "lstm":
        return LSTMModel(
            input_size=d_in,
            output_size=d_out,
            hidden_size=int(hparams.get("hidden", 128)),
            num_layers=int(hparams.get("layers", 2)),
            dropout=float(hparams.get("dropout", 0.1)),
        )

    if model_type == "transformer":
        return TransformerModel(
            input_size=d_in,
            output_size=d_out,
            d_model=int(hparams.get("d_model", 128)),
            nhead=int(hparams.get("nhead", 4)),
            num_layers=int(hparams.get("num_layers", 2)),
            dim_feedforward=int(hparams.get("dim_feedforward", 256)),
            dropout=float(hparams.get("dropout", 0.1)),
            max_len=int(hparams.get("max_len", max(500, int(hparams.get("T", 0)) + 5))),
        )

    if model_type == "mamba":
        return StackedMambaModel(
            input_size=d_in,
            output_size=d_out,
            d_model=int(hparams.get("d_model", 128)),
            d_state=int(hparams.get("d_state", 16)),
            d_conv=int(hparams.get("d_conv", 4)),
            expand=int(hparams.get("expand", 2)),
            layers=int(hparams.get("layers", 2)),
            dropout=float(hparams.get("dropout", 0.1)),
        )

    raise ValueError(f"Unsupported model type: {model_type}")


# =========================================================
# 5) Prediction and CSV saving
# =========================================================
@torch.no_grad()
def predict_numpy(model: nn.Module, X_norm: np.ndarray, device: torch.device, batch_size: int = 64) -> np.ndarray:
    """Run batch prediction and return a NumPy array with shape (N, T, D_out)."""
    model.eval()
    loader = DataLoader(SeqDataset(X_norm), batch_size=batch_size, shuffle=False)

    preds = []
    for xb in loader:
        xb = xb.to(device)
        preds.append(model(xb).detach().cpu().numpy())

    return np.concatenate(preds, axis=0)


def predictions_to_dataframe(
    Y_pred_model_unit: np.ndarray,
    out_cols: List[str],
    npz_data,
    model_path: Path,
    model_type: str,
    save_raw_unit: bool,
) -> pd.DataFrame:
    """Convert prediction array with shape (N, T, M) into a time-series CSV table."""
    n_trials, n_time, n_outputs = Y_pred_model_unit.shape
    if len(out_cols) != n_outputs:
        raise ValueError(f"len(out_cols)={len(out_cols)} does not match prediction dimension={n_outputs}.")

    df = pd.DataFrame({
        "trial_idx": np.repeat(np.arange(n_trials), n_time),
        "time_idx": np.tile(np.arange(n_time), n_trials),
        "gait_cycle_percent": np.tile(np.linspace(0.0, 100.0, n_time), n_trials),
    })

    # Add trial-level metadata and repeat it for each time step.
    for key in ["subject_id", "height_m", "weight_kg", "speed_label"]:
        if key in npz_data.files:
            df[key] = np.repeat(npz_data[key], n_time)

    # Add model-unit predictions, e.g., N/kg or Nm/kg.
    model_suffix = unit_suffix(MODEL_UNIT)
    pred_2d = Y_pred_model_unit.reshape(n_trials * n_time, n_outputs)
    for i, col in enumerate(out_cols):
        df[f"pred_{col}_{model_suffix}"] = pred_2d[:, i]

    # Optionally add predictions restored to the original raw unit, e.g., N or Nm.
    if save_raw_unit:
        if "weight_kg" not in npz_data.files:
            raise KeyError("weight_kg is required to restore predictions to the original raw unit.")
        weight_kg = npz_data["weight_kg"].astype(np.float32)
        pred_raw = (Y_pred_model_unit * weight_kg[:, None, None]).reshape(n_trials * n_time, n_outputs)
        raw_suffix = unit_suffix(RAW_UNIT)
        for i, col in enumerate(out_cols):
            df[f"pred_{col}_{raw_suffix}"] = pred_raw[:, i]

    # Add lightweight provenance columns at the end.
    df["model_type"] = model_type
    df["model_file"] = model_path.name
    df["model_prediction_unit"] = MODEL_UNIT
    df["raw_target_unit"] = RAW_UNIT

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Predict {TASK_NAME} using a trained Gait2Hip-60 model.")
    parser.add_argument("--npz_path", type=str, default=str(DEFAULT_NPZ_PATH), help="Path to Gait2Hip_JM60.npz.")
    parser.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL_PATH), help="Path to a trained .pt checkpoint.")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Directory for CSV prediction outputs.")
    parser.add_argument("--output_csv", type=str, default=None, help="Optional explicit CSV output path.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for prediction.")
    parser.add_argument("--save_raw_unit", action="store_true", help=f"Also save predictions restored to {RAW_UNIT} in the CSV.")
    parser.add_argument("--evaluate", action="store_true", help="Compute overall metrics if Y is available in the NPZ file.")
    args = parser.parse_args()

    npz_path = Path(args.npz_path)
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output_csv is None:
        output_csv = output_dir / f"{model_path.stem}_predictions.csv"
    else:
        output_csv = Path(args.output_csv)
        if output_csv.suffix.lower() != ".csv":
            output_csv = output_csv.with_suffix(".csv")
        output_csv.parent.mkdir(parents=True, exist_ok=True)

    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Task: {TASK_NAME}")
    print(f"Input NPZ: {npz_path}")
    print(f"Checkpoint: {model_path}")

    # Load input data.
    npz_data = np.load(npz_path, allow_pickle=True)
    X = npz_data["X"].astype(np.float32)
    print("X shape:", X.shape)

    # Load checkpoint and rebuild model.
    checkpoint = load_checkpoint(model_path, device)
    model_type = infer_model_type(model_path, checkpoint)
    hparams = checkpoint.get("hparams", {})
    state_dict = checkpoint.get("state_dict", checkpoint)

    model = build_model(model_type, hparams).to(device)
    model.load_state_dict(state_dict, strict=True)
    print(f"Loaded model type: {model_type}")

    # Load normalization parameters from the checkpoint.
    mu_x = np.asarray(checkpoint["mu_x"], dtype=np.float32)
    sd_x = np.asarray(checkpoint["sd_x"], dtype=np.float32)
    X_norm = apply_x_norm(X, mu_x, sd_x)

    # Run prediction.
    Y_pred = predict_numpy(model, X_norm, device=device, batch_size=args.batch_size)
    print("Y_pred shape:", Y_pred.shape)
    print(f"Prediction unit: {MODEL_UNIT}")

    # Use output column names from checkpoint if available; otherwise use NPZ columns.
    if "out_cols" in checkpoint:
        out_cols = decode_columns(checkpoint["out_cols"])
    elif "out_cols" in npz_data.files:
        out_cols = decode_columns(npz_data["out_cols"])
    else:
        out_cols = [f"output_{i}" for i in range(Y_pred.shape[-1])]

    # Save predictions as CSV only.
    df_pred = predictions_to_dataframe(
        Y_pred_model_unit=Y_pred,
        out_cols=out_cols,
        npz_data=npz_data,
        model_path=model_path,
        model_type=model_type,
        save_raw_unit=args.save_raw_unit,
    )
    df_pred.to_csv(output_csv, index=False, float_format="%.6f")
    print("Saved prediction CSV to:", output_csv)

    # Optionally evaluate against Y in the NPZ file and save metrics as CSV.
    if args.evaluate:
        if "Y" not in npz_data.files:
            raise KeyError("Y is not available in the NPZ file, so evaluation cannot be computed.")
        if "weight_kg" not in npz_data.files:
            raise KeyError("weight_kg is required to convert Y to the model target unit for evaluation.")

        Y_true_raw = npz_data["Y"].astype(np.float32)
        weight_kg = npz_data["weight_kg"].astype(np.float32)
        Y_true_model_unit = (Y_true_raw / weight_kg[:, None, None]).astype(np.float32)
        metrics = regression_metrics(Y_true_model_unit, Y_pred)

        metrics_csv = output_dir / f"{model_path.stem}_metrics.csv"
        df_metrics = pd.DataFrame([
            {
                "dataset": DATASET_ID,
                "model_type": model_type,
                "model_file": model_path.name,
                "unit": MODEL_UNIT,
                "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"],
                "R2": metrics["R2"],
            }
        ])
        df_metrics.to_csv(metrics_csv, index=False, float_format="%.6f")

        print(
            f"Evaluation in {MODEL_UNIT} | "
            f"RMSE={metrics['RMSE']:.4f} | MAE={metrics['MAE']:.4f} | R2={metrics['R2']:.4f}"
        )
        print("Saved metrics CSV to:", metrics_csv)


if __name__ == "__main__":
    main()
