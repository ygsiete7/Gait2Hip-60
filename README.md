# Gait2Hip-60: A Multi-Cadence Gait Dynamics Dataset

**Gait2Hip-60** is a multi-cadence gait dynamics dataset from 60 healthy subjects. The dataset provides trial-level OpenSim-derived gait biomechanics data, including inverse kinematics (IK), inverse dynamics (ID), and static-optimization-derived muscle force outputs, together with NPZ files for machine learning and deep learning applications.

The release includes:

- `Gait2Hip-60.zip`: the complete dataset archive containing trial-level OpenSim-derived outputs;
- `Gait2Hip_MF60.npz`: an NPZ dataset for right-hip-related muscle force analysis and prediction from gait kinematics;
- `Gait2Hip_JM60.npz`: an NPZ dataset for right hip joint moment analysis and prediction from gait kinematics;
- `subject_info.csv`: anonymized subject-level information.

> **Important note**  
> The muscle forces and joint moments provided in this dataset are derived from an OpenSim musculoskeletal modeling pipeline. They should be interpreted as simulation-based estimates rather than direct in vivo measurements.

---

## 1. Dataset overview

Gait2Hip-60 contains gait data from 60 healthy adults labeled from `H01` to `H60`.

Walking trials were performed under three metronome-paced cadence conditions:

| Condition label | Cadence |
|---|---:|
| `slow` | 78 steps/min |
| `med` | 115 steps/min |
| `fast` | 135 steps/min |

Each subject was expected to have up to four repeated trials under each of the three cadence conditions, resulting in a maximum of 12 trials per subject. After quality-control screening, trials that did not meet the predefined quality-control criteria were excluded. As a result, 57 subjects retained 12 valid trials, whereas three subjects had fewer valid trials: H17 retained 11 trials, H47 retained 10 trials, and H52 retained 8 trials.

The released data include both trial-level OpenSim-derived files and two NPZ datasets that can be directly loaded for computational modeling, machine learning, and deep learning applications.

---

## 2. Uploaded files

| File | Description |
|---|---|
| `Gait2Hip-60.zip` | Complete dataset archive containing trial-level OpenSim-derived outputs |
| `Gait2Hip_MF60.npz` | NPZ dataset for right-hip-related muscle force analysis and prediction |
| `Gait2Hip_JM60.npz` | NPZ dataset for right hip joint moment analysis and prediction |
| `subject_info.csv` | Anonymized subject-level information |

---

## 3. Dataset structure

After extracting `Gait2Hip-60.zip`, the dataset is organized as follows:

```text
Gait2Hip-60/
└── opensim_outputs/
    ├── ik/
    ├── id/
    └── mf/
```

### 3.1 `opensim_outputs/`

This folder contains trial-level OpenSim-derived outputs.

| Folder | Description | File type |
|---|---|---|
| `ik/` | Inverse kinematics outputs | `.mot` |
| `id/` | Inverse dynamics outputs | `.sto` |
| `mf/` | Muscle force outputs derived from static optimization | `.sto` |

Example filenames:

```text
slow001.mot
med002.sto
fast003.sto
```

where:

- `slow`, `med`, and `fast` are cadence condition labels corresponding to 78, 115, and 135 steps/min, respectively;
- `001`, `002`, `003`, etc. indicate repeated trial indices within each condition;
- `.mot` files are used for inverse kinematics outputs;
- `.sto` files are used for inverse dynamics and muscle force outputs.

---

## 4. NPZ datasets

Two NPZ datasets are provided for direct use in machine learning and deep learning studies.

### 4.1 `Gait2Hip_MF60.npz`

This file is designed for right-hip-related muscle force analysis and prediction from gait kinematics.

- Input: lower-limb joint kinematic sequences
- Output: right-hip-related muscle force trajectories
- Sequence length: 180 frames

### 4.2 `Gait2Hip_JM60.npz`

This file is designed for right hip joint moment analysis and prediction from gait kinematics.

- Input: lower-limb joint kinematic sequences
- Output: right hip joint moment trajectories
- Sequence length: 180 frames

The public NPZ files contain 713 valid samples after quality-control and file consistency checks.

---

## 5. Input and output variables

### 5.1 Input variables

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

### 5.2 Muscle force outputs

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

### 5.3 Joint moment outputs

`Gait2Hip_JM60.npz` contains 3 right hip joint moment outputs:

```text
hip_flexion_r_moment
hip_adduction_r_moment
hip_rotation_r_moment
```

---

## 6. Units

- Input joint kinematics are provided in degrees.
- Muscle force outputs are provided in Newtons (N).
- Joint moment outputs are provided in Newton-meters (N·m).
- Subject body mass is provided in kilograms (kg), and height is provided in meters (m).

Note: The released muscle force and joint moment outputs are not normalized by body mass. Users may perform body-mass normalization if required for their own analyses.
---
## 7. NPZ file format

The released NPZ files contain the following fields:

| Key | Description |
|---|---|
| `X` | Input sequence array |
| `Y` | Target sequence array |
| `subject_id` | Subject ID for each sample |
| `height_m` | Subject height in meters |
| `weight_kg` | Subject body mass in kilograms |
| `speed_label` | Legacy condition label for the metronome-paced cadence condition: `slow`, `med`, or `fast` |
| `in_cols` | Names of input variables |
| `out_cols` | Names of output variables |

Note: `speed_label` is a legacy field name retained for compatibility with the released code and scripts. In this dataset, it represents the metronome-paced cadence condition rather than directly measured walking speed. The labels `slow`, `med`, and `fast` correspond to 78, 115, and 135 steps/min, respectively.

Typical array dimensions are:

```text
X: (N, 180, D_in)
Y: (N, 180, D_out)
```

where:

- `N` is the number of gait-cycle samples;
- `180` is the sequence length after temporal resampling or alignment;
- `D_in` is the number of input kinematic variables;
- `D_out` is the number of output variables.

---

## 8. Loading example

Example in Python:

```python
import numpy as np

data = np.load("Gait2Hip_MF60.npz", allow_pickle=True)

print(data.files)
print(data["X"].shape)
print(data["Y"].shape)
print(data["in_cols"])
print(data["out_cols"])

X = data["X"]
Y = data["Y"]
subject_id = data["subject_id"]
cadence_condition = data["speed_label"]
```

---

## 9. Subject information

The file `subject_info.csv` provides anonymized subject-level information for the 60 healthy subjects.

The public release does not include personally identifiable information.

---

## 10. Ethics and privacy

The data were collected and processed for research purposes. All released subject identifiers are anonymized as `H01` to `H60`.

Users should not attempt to re-identify individual participants.

---

## 11. Recommended use

Gait2Hip-60 can be used for research on:

- gait biomechanics;
- musculoskeletal modeling;
- hip dynamics analysis;
- machine learning-based biomechanical estimation;
- time-series modeling;
- deep learning model development and evaluation.

---

## 12. Citation

If you use this dataset, please cite the associated dataset record and publication when available.

Dataset title:

```text
Gait2Hip-60: A Multi-Cadence Gait Dynamics Dataset
```

---

## 13. License

The dataset is released under the license specified on the Zenodo record.

Please check the Zenodo license field before reuse.

---

## 14. Contact

For questions about the dataset, please contact:

```text
Jiaqi Zhang
Capital University of Physical Education and Sports
Email: jackie4real@outlook.com
```