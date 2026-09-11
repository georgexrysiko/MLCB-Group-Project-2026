import numpy as np
import pandas as pd
import scipy.sparse as sp
from typing import Dict, Tuple
from sklearn.metrics import (accuracy_score,average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, matthews_corrcoef,precision_score,
                             recall_score,roc_auc_score)


# ======================================================================
# Convert sparse or dense matrices to dense float32 arrays --> Preprocessing step requires dense input.
def to_float32_dense(matrix) -> np.ndarray:

    if sp.issparse(matrix):
        matrix = matrix.toarray()

    return np.asarray(matrix,dtype=np.float32,)

# ======================================================================
# Combining two feature matrices while keeping sparse input sparse.
def combine_features(left, right):
    right = np.asarray(right, dtype=np.float32,)

    if sp.issparse(left):
        return sp.hstack(
            [left.astype(np.float32), sp.csr_matrix(right)],
            format="csr",)
    return np.hstack([np.asarray(left,dtype=np.float32), right,]
                     ).astype(np.float32,copy=False,)


# ======================================================================
# Function for aggregating cell-level features by taking the mean across all cells
# of each biological sample (patient_timepoint).

def aggregate_matrix_by_sample(matrix,biological_sample_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    biological_sample_ids = np.asarray(biological_sample_ids).astype(str)

    # The matrix and sample IDs must refer to the same cells.
    if matrix.shape[0] != len(biological_sample_ids):
        raise ValueError(f"The matrix contains {matrix.shape[0]} rows, but biological_sample_ids contains {len(biological_sample_ids)} values.")

    unique_samples = np.sort(pd.unique(biological_sample_ids))

    rows = []

    for sample_id in unique_samples:

        sample_matrix = matrix[biological_sample_ids == sample_id]

        if sp.issparse(sample_matrix):
            sample_mean = np.asarray(sample_matrix.mean(axis=0)).ravel()
        else:
            sample_mean = np.asarray(sample_matrix,dtype=np.float32).mean(axis=0)

        rows.append(sample_mean.astype(np.float32,copy=False))

    return (np.vstack(rows),unique_samples)


# ======================================================================
# Obtaining one binary response label for each biological sample.
# Remember: Every patient_timepoint (samples) must contain only one response label.

def labels_for_samples(biological_sample_ids: np.ndarray,labels: np.ndarray,unique_samples: np.ndarray) -> np.ndarray:

    biological_sample_ids = np.asarray(biological_sample_ids).astype(str)
    labels = np.asarray(labels,dtype=np.int8)
    unique_samples = np.asarray(unique_samples).astype(str)

    # The sample IDs and labels must refer to the same cells.
    if len(biological_sample_ids) != len(labels):
        raise ValueError("Biological_sample_ids and labels must have the same length.")

    # Checking that the response is binary.
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Labels must contain only 0 and 1.")

    sample_labels = pd.DataFrame({"sample_id": biological_sample_ids,"label": labels,})

    # Biological sample --> only one response label.
    label_counts = (sample_labels.groupby("sample_id",observed=True,)["label"].nunique())
    mixed_samples = label_counts[label_counts > 1]

    if not mixed_samples.empty:
        raise ValueError("The following biological samples contain multiple response labels")

    label_mapping = (sample_labels.drop_duplicates(subset="sample_id")
                     .set_index("sample_id")["label"])

    missing_samples = (set(unique_samples) - set(label_mapping.index.astype(str)))
    if missing_samples:
        raise ValueError(f"No labels were found for the following biological samples: {sorted(missing_samples)}")

    return (label_mapping.loc[unique_samples].to_numpy(dtype=np.int8))


# ======================================================================
# Function for calculating classification metrics at sample or patient level.
# Worth mentioning: Mixed-response patients are excluded only from patient-level evaluation.

def calculate_metrics(predictions: pd.DataFrame,level: str,threshold: float = 0.5,) -> Dict[str, object]:

    level = level.lower().strip()

    if level == "sample":
        score_column = "sample_score"
        count_name = "N_samples"

    elif level == "patient":
        score_column = "patient_score"
        count_name = "N_patients"

    else:
        raise ValueError("level must be either 'sample' or 'patient'.")

    required_columns = {"true_label",score_column}
    missing_columns = (required_columns - set(predictions.columns))
    if missing_columns:
        raise KeyError(f"Missing columns for {level}-level metrics: {sorted(missing_columns)}")

    evaluation_predictions = predictions.copy()
    n_total_rows = len(evaluation_predictions)

    # Mixed-response patients do not have one valid patient-level label
    # Therefore they remain in sample-level evaluation but are excluded here.
    n_excluded_mixed_patients = 0
    if level == "patient":
        valid_mask = (evaluation_predictions["true_label"].notna())
        n_excluded_mixed_patients = int((~valid_mask).sum())
        evaluation_predictions = (evaluation_predictions.loc[valid_mask].copy())

    elif evaluation_predictions["true_label"].isna().any():
        raise ValueError("Sample-level predictions contain missing true labels.")

    if evaluation_predictions.empty:
        raise ValueError(f"No valid rows remain for {level}-level evaluation.")

    y_true = (evaluation_predictions["true_label"].astype(int).to_numpy())
    scores = (evaluation_predictions[score_column].astype(float).to_numpy())

    if not np.isfinite(scores).all():
        raise ValueError(f"{level.capitalize()} scores contain non-finite values.")
    if not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError("true_label must contain only 0 and 1.")

    predicted_labels = (scores >= threshold).astype(int)

    # AUC and Average Precision --> Both response classes are required.
    if np.unique(y_true).size == 2:

        auc = float(roc_auc_score( y_true,scores))
        average_precision = float(average_precision_score(y_true,scores))

    else:
        auc = np.nan
        average_precision = np.nan

    tn, fp, fn, tp = confusion_matrix(y_true,predicted_labels,labels=[0, 1],).ravel()

    return {"Level": level,"AUC": auc,"Average_precision": average_precision,
            "Accuracy": float(accuracy_score( y_true,predicted_labels)),
            "Balanced_accuracy": float(balanced_accuracy_score( y_true,predicted_labels)),
            "Precision": float(precision_score(y_true,predicted_labels,zero_division=0)),
            "Sensitivity": float(recall_score(y_true,predicted_labels,pos_label=1,zero_division=0)),
            "Specificity": float(recall_score(y_true,predicted_labels,pos_label=0,zero_division=0)),
            "F1": float(f1_score(y_true,predicted_labels,zero_division=0)),
            "MCC": float(matthews_corrcoef(y_true,predicted_labels)),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
            count_name: int(len(evaluation_predictions)),
            "N_total_rows": int(n_total_rows),
            "N_excluded_mixed_patients": (n_excluded_mixed_patients if level == "patient" else 0),
            "Threshold": float(threshold)
            }


# ======================================================================
# In this function we create fold-specific cell weights so that each biological sample
# contributes equally and the two response classes are balanced.

def make_sample_balanced_class_weights(training_sample_ids: np.ndarray,training_labels: np.ndarray,) -> np.ndarray:

    training_sample_ids = np.asarray(training_sample_ids).astype(str)
    training_labels = np.asarray(training_labels,dtype=np.int8,)

    unique_samples = np.sort(pd.unique(training_sample_ids))

    # Obtaining one response label for each biological sample.
    sample_labels = labels_for_samples(biological_sample_ids=training_sample_ids,
                                       labels=training_labels,
                                       unique_samples=unique_samples,)

    # Counting biological samples in each response class.
    class_counts = (pd.Series(sample_labels).value_counts().to_dict())

    if len(class_counts) < 2:
        raise ValueError("Both response classes are required to calculate balanced weights."
)

    n_samples = len(unique_samples)
    n_classes = len(class_counts)

    # Balancing the two classes at biological sample level.
    class_weight_by_label = {int(label): n_samples / (n_classes * count) for label, count in class_counts.items()}

    # Dividing each sample's total weight equally across its cells.
    cells_per_sample = ( pd.Series(training_sample_ids).value_counts().to_dict())

    weights = np.asarray( [class_weight_by_label[int(label)] / cells_per_sample[str(sample_id)]
                           for sample_id, label in zip(training_sample_ids,training_labels,)],
                           dtype=np.float64)

    # Normalizing the weights --> their mean = 1.
    weights *= len(weights) / weights.sum()

    return weights
