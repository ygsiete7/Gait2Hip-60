# Gait2Hip-60: A Multi-Cadence Gait Dynamics Dataset

**Gait2Hip-60** is a multi-cadence gait dynamics dataset from 60 healthy subjects. The dataset provides trial-level OpenSim-derived gait biomechanics data, including inverse kinematics (IK), inverse dynamics (ID), and static-optimization-derived muscle force outputs, together with NPZ files for machine learning and deep learning applications.

![image](workflow.png)

This GitHub repository provides example scripts and benchmark code for using the released NPZ files in machine learning and deep learning applications.

The full dataset is available on **Zenodo**.

```text
https://doi.org/10.5281/zenodo.20175768
```
> **Important note**  
> The muscle forces and joint moments provided in this dataset are derived from an OpenSim musculoskeletal modeling pipeline. They should be interpreted as simulation-based estimates rather than direct in vivo measurements.

---

## Repository structure

```text
Gait2Hip-60/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── data/
│   └── README.md
│
├── examples/
│   ├── load_Gait2Hip_MF60.py
│   └── load_Gait2Hip_JM60.py
│
├── muscle_force_prediction/
│   ├── train_lstm.py
│   ├── train_mamba.py
│   ├── train_transformer.py
│   ├── predict.py
│   ├── trained_models/
│   │   ├── lstm_mf60.pt
│   │   ├── mamba_mf60.pt
│   │   └── transformer_mf60.pt
│   └── README.md
│
└── joint_moment_prediction/
    ├── train_lstm.py
    ├── train_mamba.py
    ├── train_transformer.py
    ├── predict.py
    ├── trained_models/
    │   ├── lstm_jm60.pt
    │   ├── mamba_jm60.pt
    │   └── transformer_jm60.pt
    └── README.md
```

---

## Data availability
The Zenodo release includes:

| File | Description |
|---|---|
| `Gait2Hip-60.zip` | Complete dataset archive containing trial-level OpenSim-derived outputs |
| `Gait2Hip_MF60.npz` | NPZ file for right-hip-related muscle force analysis and prediction |
| `Gait2Hip_JM60.npz` | NPZ file for right hip joint moment analysis and prediction |
| `subject_info.csv` | Subject information |

Large dataset files are not stored directly in this GitHub repository.

After downloading the NPZ files from Zenodo, place them in the `data/` folder:

```text
data/
├── Gait2Hip_MF60.npz
└── Gait2Hip_JM60.npz
```

---

## Dataset overview

Gait2Hip-60 contains gait data from 60 healthy adults labeled from `H01` to `H60`.

Walking trials were performed under three metronome-paced cadence conditions:

| Condition label | Cadence |
|---|---:|
| `slow` | 78 steps/min |
| `med` | 115 steps/min |
| `fast` | 135 steps/min |

Each subject was expected to have up to four repeated trials under each of the three cadence conditions, resulting in a maximum of 12 trials per subject. After quality-control screening, trials that did not meet the predefined quality-control criteria were excluded.

As a result, 57 subjects retained 12 valid trials, whereas three subjects had fewer valid trials:

| Subject | Number of valid trials |
|---|---:|
| `H17` | 11 |
| `H47` | 10 |
| `H52` | 8 |

The public NPZ files contain 713 valid samples after quality-control and file consistency checks.

---

## NPZ file format

The released NPZ files contain the following fields:

| Key | Description |
|---|---|
| `X` | Input sequence array |
| `Y` | Target sequence array |
| `subject_id` | Subject ID for each sample |
| `height_m` | Subject height in meters |
| `weight_kg` | Subject body mass in kilograms |
| `speed_label` | Legacy condition label: `slow`, `med`, or `fast` |
| `in_cols` | Names of input variables |
| `out_cols` | Names of output variables |

Typical array dimensions are:

```text
X: (N, 180, D_in)
Y: (N, 180, D_out)
```

where:

- `N` is the number of valid gait samples;
- `180` is the sequence length;
- `D_in` is the number of input kinematic variables;
- `D_out` is the number of output variables.

> Note: `speed_label` is a legacy field name retained for compatibility with the released code and scripts. In this dataset, it represents the metronome-paced cadence condition rather than directly measured walking speed. The labels `slow`, `med`, and `fast` correspond to 78, 115, and 135 steps/min, respectively.

---

## Input and output variables

### Input variables

The NPZ input array contains 10 lower-limb kinematic variables:

```text
hip_flexion_r
hip_adduction_r
hip_rotation_r
hip_flexion_l
hip_adduction_l
hip_rotation_l
knee_angle_r
ankle_angle_r
knee_angle_l
ankle_angle_l
```

### Muscle force outputs

`Gait2Hip_MF60.npz` contains 14 right-hip-related muscle force outputs:

```text
addbrev_r
addlong_r
addmag_r
grac_r
iliopsoas_r
recfem_r
sart_r
glmax_r
bflh_r
hamstring_r
glmed_r
glmin_r
tfl_r
piri_r
```

### Joint moment outputs

`Gait2Hip_JM60.npz` contains 3 right hip joint moment outputs:

```text
hip_flexion_r_moment
hip_adduction_r_moment
hip_rotation_r_moment
```

---

## Units

The released NPZ files store the biomechanical outputs in their original units:

- input joint kinematics: degrees;
- muscle force outputs: Newtons (N);
- joint moment outputs: Newton-meters (N·m);
- subject body mass: kilograms (kg);
- subject height: meters (m).

In the provided training scripts, target outputs are normalized by body mass before model training:

- muscle force prediction: `N/kg`;
- joint moment prediction: `Nm/kg`.

Users may choose either the original-unit targets or body-mass-normalized targets depending on their own research purpose.

---

## Requirements
This repository contains Python scripts for loading the NPZ files and running the baseline models. The main dependencies include:

```text
numpy
pandas
torch
scikit-learn
mamba-ssm
```
>Note: `mamba-ssm` is required only for running the Mamba baseline.

---

## Quick start

### Load the muscle force NPZ file

```bash
python examples/load_Gait2Hip_MF60.py
```

### Load the joint moment NPZ file

```bash
python examples/load_Gait2Hip_JM60.py
```

The loading scripts print the available keys, array shapes, input variables, output variables, condition labels, and a preview of the first trial.

---

## Benchmark protocol

The provided baseline scripts follow the same general protocol:

1. Load the corresponding NPZ file.
2. Normalize the input kinematic variables using statistics computed from the training data.
3. Normalize target outputs by body mass:
   - `Gait2Hip_MF60.npz`: muscle forces are converted from `N` to `N/kg`;
   - `Gait2Hip_JM60.npz`: joint moments are converted from `N·m` to `N·m/kg`.
4. Split subjects into a training/validation pool and an independent test set.
5. Use 5-fold GroupKFold cross-validation on the training/validation pool for epoch selection.
6. Train the final model on the full training/validation pool.
7. Evaluate the final model on the independent subject-holdout test set.
8. Report RMSE, MAE, and R², including subject-level mean metrics and cadence-specific metrics.

The implemented baseline models are:

- LSTM;
- Transformer;
- Mamba.

---

## Model training

Before running the scripts, check and modify the dataset path and output directory in each script if needed.

### Muscle force prediction

```bash
cd muscle_force_prediction

python LSTM.py
python Transformer.py
python Mamba.py
```

### Joint moment prediction

```bash
cd joint_moment_prediction

python LSTM.py
python Transformer.py
python Mamba.py
```

Each script saves:

- 5-fold cross-validation metrics;
- final test metrics;
- cadence-specific test metrics;
- final model checkpoint.

---

## Reference results

The full benchmark results are reported in the associated manuscript.

```text
[Paper DOI or arXiv URL, if available]
```
---

## Recommended use

Gait2Hip-60 can be used for research on:

- gait biomechanics;
- musculoskeletal modeling;
- hip dynamics analysis;
- machine learning-based biomechanical estimation;
- time-series modeling;
- deep learning model development and evaluation.

---



## Citation

If you use this dataset or code, please cite the associated dataset record and publication when available.

Dataset title:

```text
Gait2Hip-60: A Multi-Cadence Gait Dynamics Dataset
```

Zenodo record:

```text
[Zenodo DOI or URL]
```

Publication:

```text
[Paper DOI or arXiv URL, if available]
```

---

## License

The dataset is released under the license specified on the Zenodo record.

The source code in this repository is released under the license specified in this GitHub repository.

Please check the license files before reuse.

---

## Contact

For questions about the dataset or code, please contact:

```text
Jiaqi Zhang
Capital University of Physical Education and Sports
Email: jackie4real@outlook.com
```
