# Joint Moment Prediction (JM60)

This folder provides training and prediction scripts for the right hip joint moment prediction task in **Gait2Hip-60**.

## Task

- Dataset file: `Gait2Hip_JM60.npz`
- Input: lower-limb kinematic sequences `X`
- Output: right hip joint moment sequences `Y`
- Raw target unit in the dataset: **Nm**
- Model training target unit: **Nm/kg**

The original target values in `Gait2Hip_JM60.npz` are in **Nm**. During training, the script converts them to body-weight-normalized values:

```python
Y_train = Y_raw / weight_kg[:, None, None]
```

Therefore, the default model prediction unit is **Nm/kg**.

## Expected folder structure

```text
Gait2Hip-60/
├── data/
│   └── Gait2Hip_JM60.npz
│
└── joint_moment_prediction/
    ├── LSTM_JM60.py
    ├── Mamba_JM60.py
    ├── Transformer_JM60.py
    ├── predict.py
    ├── trained_models/
    │   ├── LSTM_JM60.pt
    │   ├── Mamba_JM60.pt
    │   └── Transformer_JM60.pt
    ├── results/
    └── predictions/
```

The full dataset should be downloaded from Zenodo and placed in the `data/` folder. The GitHub repository only provides code, example scripts, training scripts, and trained models.

## Training

Run the training scripts from the repository root.

### Train LSTM

```bash
python joint_moment_prediction/LSTM_JM60.py
```

### Train Mamba

```bash
python joint_moment_prediction/Mamba_JM60.py
```

### Train Transformer

```bash
python joint_moment_prediction/Transformer_JM60.py
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
joint_moment_prediction/trained_models/
```

The evaluation CSV files are saved to:

```text
joint_moment_prediction/results/
```

Typical output files are:

```text
joint_moment_prediction/trained_models/LSTM_JM60.pt
joint_moment_prediction/trained_models/Mamba_JM60.pt
joint_moment_prediction/trained_models/Transformer_JM60.pt

joint_moment_prediction/results/LSTM_JM60_cv_5fold_baseline_metrics.csv
joint_moment_prediction/results/LSTM_JM60_test_metrics_by_speed_mean.csv
```

The same naming pattern is used for Mamba and Transformer.

## Prediction

The `predict.py` script loads a trained model and saves prediction results directly as **CSV**. It does not generate `.npz` prediction files.

### Default prediction

By default, the script uses `Transformer_JM60.pt`:

```bash
python joint_moment_prediction/predict.py
```

Default input:

```text
data/Gait2Hip_JM60.npz
```

Default model:

```text
joint_moment_prediction/trained_models/Transformer_JM60.pt
```

Default output:

```text
joint_moment_prediction/predictions/Transformer_JM60_predictions.csv
```

The prediction columns are saved in **Nm/kg**.

### Use a specific model

```bash
python joint_moment_prediction/predict.py \
  --model_path joint_moment_prediction/trained_models/LSTM_JM60.pt
```

```bash
python joint_moment_prediction/predict.py \
  --model_path joint_moment_prediction/trained_models/Mamba_JM60.pt
```

```bash
python joint_moment_prediction/predict.py \
  --model_path joint_moment_prediction/trained_models/Transformer_JM60.pt
```

### Save predictions restored to Nm

To include both **Nm/kg** and restored **Nm** predictions in the CSV:

```bash
python joint_moment_prediction/predict.py \
  --model_path joint_moment_prediction/trained_models/Transformer_JM60.pt \
  --save_raw_unit
```

The restored raw-unit prediction is computed as:

```python
Y_pred_Nm = Y_pred_Nm_per_kg * weight_kg[:, None, None]
```

### Evaluate predictions

If `Y` is available in the NPZ file, you can compute overall metrics:

```bash
python joint_moment_prediction/predict.py \
  --model_path joint_moment_prediction/trained_models/Transformer_JM60.pt \
  --evaluate
```

The metrics are saved as CSV:

```text
joint_moment_prediction/predictions/Transformer_JM60_metrics.csv
```

The metrics are computed in **Nm/kg**, because this is the model training and prediction unit.

## Prediction CSV format

The output CSV contains one row per trial and time point:

```text
trial_idx,time_idx,gait_cycle_percent,subject_id,height_m,weight_kg,speed_label,...
```

Prediction columns use this naming style:

```text
pred_<output_name>_Nm_per_kg
```

If `--save_raw_unit` is used, additional columns are included:

```text
pred_<output_name>_Nm
```

## Mamba dependency

The Mamba model requires `mamba_ssm`. If you only use LSTM or Transformer, this dependency is not required for prediction.
