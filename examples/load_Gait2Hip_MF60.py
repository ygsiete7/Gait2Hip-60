from pathlib import Path

import numpy as np
import pandas as pd


# Set the default path to the NPZ file.
# Users should download Gait2Hip_MF60.npz from Zenodo and place it in the data/ folder.
NPZ_PATH = Path(__file__).resolve().parents[1] / "data" / "Gait2Hip_MF60.npz"


def decode_columns(columns):
    """Convert column names to a standard Python string list."""
    return [
        col.decode("utf-8") if isinstance(col, bytes) else str(col)
        for col in columns
    ]


def main():
    """Load and inspect the Gait2Hip_MF60 dataset."""

    if not NPZ_PATH.exists():
        raise FileNotFoundError(
            f"File not found: {NPZ_PATH}\n"
            "Please download Gait2Hip_MF60.npz from Zenodo and place it in the data/ folder."
        )

    # Load the NPZ file.
    data = np.load(NPZ_PATH, allow_pickle=True)

    # Print all available keys in the NPZ file.
    print("Available keys:", data.files)

    # Load input features and target outputs.
    X = data["X"]  # Shape: (N, T, D_in)
    Y = data["Y"]  # Shape: (N, T, M)

    # Load trial-level metadata.
    subject_id = data["subject_id"]      # Shape: (N,)
    height_m = data["height_m"]          # Shape: (N,)
    weight_kg = data["weight_kg"]        # Shape: (N,)
    speed_label = data["speed_label"]    # Shape: (N,)

    # Load input and output column names.
    in_cols = decode_columns(data["in_cols"])
    out_cols = decode_columns(data["out_cols"])

    # Print basic dataset information.
    print("\nDataset: Gait2Hip_MF60")
    print("Task: Right hip-related muscle force prediction")
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)
    print("Number of trials:", X.shape[0])
    print("Number of time steps:", X.shape[1])
    print("Number of input variables:", X.shape[2])
    print("Number of output variables:", Y.shape[2])

    # Print metadata examples.
    print("\nFirst 5 subject IDs:", subject_id[:5])
    print("First 5 heights:", height_m[:5])
    print("First 5 body weights:", weight_kg[:5])
    print("First 10 speed labels:", speed_label[:10])

    # Print column names.
    print("\nInput columns:")
    print(in_cols)

    print("\nOutput columns:")
    print(out_cols)

    # Preview the first trial.
    trial_idx = 0

    # Convert the first trial input and output arrays to pandas DataFrames.
    df_X = pd.DataFrame(X[trial_idx], columns=in_cols)
    df_Y = pd.DataFrame(Y[trial_idx], columns=out_cols)

    print(f"\nPreview of input data for trial {trial_idx}:")
    print(df_X.head())

    print(f"\nPreview of muscle force outputs for trial {trial_idx}:")
    print(df_Y.head())


if __name__ == "__main__":
    main()