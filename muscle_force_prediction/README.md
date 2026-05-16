# Muscle Force Prediction (MF60)

This folder provides training and prediction scripts for the right hip-related muscle force prediction task in **Gait2Hip-60**.

## Task

- Dataset file: `Gait2Hip_MF60.npz`
- Input: lower-limb kinematic sequences `X`
- Output: right hip-related muscle force sequences `Y`
- Raw target unit in the dataset: **N**
- Model training target unit: **N/kg**

The original target values in `Gait2Hip_MF60.npz` are in **N**. During training, the script converts them to body-weight-normalized values:

```python
Y_train = Y_raw / weight_kg[:, None, None]
```

Therefore, the default model prediction unit is **N/kg**.

## Expected folder structure

```text
Gait2Hip-60/
├── data/
│   └── Gait2Hip_MF60.npz
│
└── muscle_force_prediction/
    ├── LSTM_MF60.py
    ├── Mamba_MF60.py
    ├── Transformer_MF60.py
    ├── predict.py
    ├── trained_models/
    │   ├── LSTM_MF60.pt
    │   ├── Mamba_MF60.pt
    │   └── Transformer_MF60.pt
    ├── results/
    └── predictions/
```

The full dataset should be downloaded from Zenodo and placed in the `data/` folder. The GitHub repository only provides code, example scripts, training scripts, and trained models.

## Training

Run the training scripts from the repository root.

### Train LSTM

```bash
python muscle_force_prediction/LSTM_MF60.py
```

### Train Mamba

```bash
python muscle_force_prediction/Mamba_MF60.py
```

### Train Transformer

```bash
python muscle_force_prediction/Transformer_MF60.py
```

Each training script uses the same protocol:

```text
Subject-holdout TEST split
+ 5-fold GroupKFold benchmark on the training pool
+ median best epoch selection
+ final training on the full pool
+ independent TEST evaluation
```

The trained models are saved to:

```text
muscle_force_prediction/trained_models/
```

The evaluation CSV files are saved to:

```text
muscle_force_prediction/results/
```

Typical output files are:

```text
muscle_force_prediction/trained_models/LSTM_MF60.pt
muscle_force_prediction/trained_models/Mamba_MF60.pt
muscle_force_prediction/trained_models/Transformer_MF60.pt

muscle_force_prediction/results/LSTM_MF60_cv_5fold_baseline_metrics.csv
muscle_force_prediction/results/LSTM_MF60_test_metrics_by_speed_mean.csv
```

The same naming pattern is used for Mamba and Transformer.

## Prediction

The `predict.py` script loads a trained model and saves prediction results directly as **CSV**. It does not generate `.npz` prediction files.

### Default prediction

By default, the script uses `Transformer_MF60.pt`:

```bash
python muscle_force_prediction/predict.py
```

Default input:

```text
data/Gait2Hip_MF60.npz
```

Default model:

```text
muscle_force_prediction/trained_models/Transformer_MF60.pt
```

Default output:

```text
muscle_force_prediction/predictions/Transformer_MF60_predictions.csv
```

The prediction columns are saved in **N/kg**.

### Use a specific model

```bash
python muscle_force_prediction/predict.py \
  --model_path muscle_force_prediction/trained_models/LSTM_MF60.pt
```

```bash
python muscle_force_prediction/predict.py \
  --model_path muscle_force_prediction/trained_models/Mamba_MF60.pt
```

```bash
python muscle_force_prediction/predict.py \
  --model_path muscle_force_prediction/trained_models/Transformer_MF60.pt
```

### Save predictions restored to N

To include both **N/kg** and restored **N** predictions in the CSV:

```bash
python muscle_force_prediction/predict.py \
  --model_path muscle_force_prediction/trained_models/Transformer_MF60.pt \
  --save_raw_unit
```

The restored raw-unit prediction is computed as:

```python
Y_pred_N = Y_pred_N_per_kg * weight_kg[:, None, None]
```

### Evaluate predictions

If `Y` is available in the NPZ file, you can compute overall metrics:

```bash
python muscle_force_prediction/predict.py \
  --model_path muscle_force_prediction/trained_models/Transformer_MF60.pt \
  --evaluate
```

The metrics are saved as CSV:

```text
muscle_force_prediction/predictions/Transformer_MF60_metrics.csv
```

The metrics are computed in **N/kg**, because this is the model training and prediction unit.

## Prediction CSV format

The output CSV contains one row per trial and time point:

```text
trial_idx,time_idx,gait_cycle_percent,subject_id,height_m,weight_kg,speed_label,...
```

Prediction columns use this naming style:

```text
pred_<output_name>_N_per_kg
```

If `--save_raw_unit` is used, additional columns are included:

```text
pred_<output_name>_N
```

## Mamba dependency

The Mamba model requires `mamba_ssm`. If you only use LSTM or Transformer, this dependency is not required for prediction.
