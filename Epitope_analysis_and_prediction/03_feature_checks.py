# ==========================================================
# File: 03_feature_checks.py
# Purpose: Test the open claims from EDA before modeling.
# Input: Data/B_Cells_cleaned.csv, Data/Sars_cleaned.csv
# ==========================================================

import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 100)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"

SCORES = [
    "chou_fasman",
    "emini",
    "kolaskar_tongaonkar",
    "parker",
]

PEPTIDE = SCORES + ["peptide_length"]

PROTEIN = [
    "isoelectric_point",
    "aromaticity",
    "hydrophobicity",
    "stability",
]

SCORES_LOG = [
    "chou_fasman",
    "emini_log",
    "kolaskar_tongaonkar",
    "parker",
]

CORR_COLS = SCORES_LOG + ["peptide_length"]

REQUIRED_COLUMNS = [
    "protein_seq",
    "start_position",
    "end_position",
    "peptide_seq",
    "target",
] + SCORES + PROTEIN


# ==========================================================
# GENERAL FUNCTIONS
# ==========================================================

def banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def validate_auc_data(df, feature, dataset_name):
    """Check whether an AUC calculation has valid input data."""

    valid = df[[feature, "target"]].dropna()

    if valid.empty:
        raise ValueError(
            f"{dataset_name}: no valid observations for {feature}."
        )

    if valid["target"].nunique() != 2:
        raise ValueError(
            f"{dataset_name}: both target classes are required "
            f"for AUC calculation of {feature}."
        )

    return valid


def calculate_auc(y_true, scores):
    """Calculate ROC-AUC only when both classes are present."""

    if pd.Series(y_true).nunique() != 2:
        return np.nan

    return roc_auc_score(y_true, scores)


# ==========================================================
# LOAD AND CLEAN DATA
# ==========================================================

def load(name):

    file_path = DATA_DIR / name

    if not file_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{name} is missing required columns: {missing_columns}"
        )

    start_rows = len(df)

    # Check missing values in required columns.
    missing_counts = df[REQUIRED_COLUMNS].isna().sum()

    if missing_counts.any():
        problematic = missing_counts[missing_counts > 0]
        raise ValueError(
            f"{name} contains missing required values:\n{problematic}"
        )

    # Convert coordinates and numerical features.
    numeric_columns = [
        "start_position",
        "end_position",
        "target",
    ] + SCORES + PROTEIN

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    if df[numeric_columns].isna().any().any():
        raise ValueError(
            f"{name} contains non-numeric or missing values "
            "in required numerical columns."
        )

    # Validate target labels.
    if not set(df["target"].unique()).issubset({0, 1}):
        raise ValueError(
            f"{name} contains target values other than 0 and 1."
        )

    if df["target"].nunique() != 2:
        raise ValueError(
            f"{name} must contain both target classes."
        )

    # Validate coordinates.
    invalid_coordinates = (
        (df["start_position"] < 1)
        | (df["end_position"] < df["start_position"])
        | (df["start_position"] % 1 != 0)
        | (df["end_position"] % 1 != 0)
    )

    if invalid_coordinates.any():
        raise ValueError(
            f"{name} contains invalid peptide coordinates."
        )

    # Calculate peptide length using 1-based inclusive coordinates.
    df["peptide_length"] = (
        df["end_position"] - df["start_position"] + 1
    ).astype(int)

    # Check peptide sequence length against coordinates.
    sequence_lengths = df["peptide_seq"].astype(str).str.len()

    length_mismatches = (
        sequence_lengths != df["peptide_length"]
    )

    if length_mismatches.any():
        raise ValueError(
            f"{name} contains peptide sequence and coordinate "
            "length mismatches."
        )

    # Emini must be non-negative for log1p.
    if (df["emini"] < 0).any():
        raise ValueError(
            f"{name} contains negative Emini values. "
            "log1p transformation is not valid for these values."
        )

    # Identify peptide sequences with conflicting target labels.
    labels_per_peptide = (
        df.groupby("peptide_seq")["target"].nunique()
    )

    conflicting_peptides = labels_per_peptide[
        labels_per_peptide > 1
    ].index

    conflicting_rows = df["peptide_seq"].isin(
        conflicting_peptides
    )

    number_conflicting_peptides = len(conflicting_peptides)
    number_conflicting_rows = int(conflicting_rows.sum())

    # Remove all rows belonging to conflicting peptide sequences.
    df = df.loc[~conflicting_rows].copy()

    rows_after_conflict_removal = len(df)

    # Remove repeated records for the same protein region.
    rows_before_deduplication = len(df)

    df = df.drop_duplicates(
        subset=[
            "protein_seq",
            "start_position",
            "end_position",
        ],
        keep="first",
    ).copy()

    duplicate_rows_removed = (
        rows_before_deduplication - len(df)
    )

    df = df.reset_index(drop=True)

    # Log-transform Emini.
    df["emini_log"] = np.log1p(df["emini"])

    print(f"\n{name}")
    print("Original rows:", start_rows)
    print(
        "Conflicting peptide sequences:",
        number_conflicting_peptides
    )
    print(
        "Rows removed due to conflicting labels:",
        number_conflicting_rows
    )
    print(
        "Rows remaining after conflict removal:",
        rows_after_conflict_removal
    )
    print(
        "Repeated protein-coordinate records removed:",
        duplicate_rows_removed
    )
    print("Final rows:", len(df))
    print(
        "Final positive rate:",
        round(df["target"].mean(), 3)
    )

    return df


# ==========================================================
# STEP 0: LOAD AND RESOLVE CONFLICTS
# ==========================================================

banner("STEP 0: LOAD AND RESOLVE CONFLICTS")

b = load("B_Cells_cleaned.csv")
s = load("Sars_cleaned.csv")

print("\nB_Cells positive rate:", round(b["target"].mean(), 3))
print("SARS positive rate   :", round(s["target"].mean(), 3))


# ==========================================================
# CHECK 1: EPITOPE SHARE BY EXACT PEPTIDE LENGTH
# ==========================================================

banner("CHECK 1: EPITOPE SHARE BY PEPTIDE LENGTH")

for name, df in [
    ("B_Cells", b),
    ("SARS (up to 25 aa)", s[s["peptide_length"] <= 25]),
]:

    table = df.groupby("peptide_length")["target"].agg(
        peptides="size",
        epitopes="sum",
        epitope_share="mean",
    )

    table["epitope_share"] = (
        table["epitope_share"].round(3)
    )

    print(f"\n{name}")
    print(table)

print("\nSARS length bands by target")

bands = pd.cut(
    s["peptide_length"],
    bins=[0, 15, 20, 50, 1000],
    labels=["5-15", "16-20", "21-50", "over 50"],
    include_lowest=True,
)

print(
    pd.crosstab(
        bands,
        s["target"],
        margins=True,
    )
)


# ==========================================================
# CHECK 2: PROTEIN SIZE VERSUS POSITIVE RATE
# ==========================================================

banner("CHECK 2: PROTEIN SIZE VERSUS POSITIVE RATE")

per_protein = b.groupby("protein_seq").agg(
    peptides=("target", "size"),
    positives=("target", "sum"),
)

per_protein["positive_rate"] = (
    per_protein["positives"] / per_protein["peptides"]
)

rho = per_protein[
    ["peptides", "positive_rate"]
].corr(method="spearman").iloc[0, 1]

print(
    "Spearman, peptides per protein vs positive rate:",
    round(rho, 3),
)

size_bins = pd.cut(
    per_protein["peptides"],
    bins=[0, 1, 2, 5, 20, 100, 10000],
)

summary = per_protein.groupby(
    size_bins,
    observed=True,
).agg(
    proteins=("peptides", "size"),
    peptides=("peptides", "sum"),
    positives=("positives", "sum"),
)

summary["peptide_weighted_positive_rate"] = (
    summary["positives"] / summary["peptides"]
).round(3)

print("\nProtein-size group summary")
print(summary)


# ==========================================================
# CHECK 3: SINGLE-FEATURE AUC
# ==========================================================

banner("CHECK 3: SINGLE-FEATURE AUC")


def auc_table(df, features, dataset_name):

    rows = []

    for feature in features:

        valid = validate_auc_data(
            df,
            feature,
            dataset_name,
        )

        pooled_auc = roc_auc_score(
            valid["target"],
            valid[feature],
        )

        per_protein_auc = []

        for _, group in df.groupby("protein_seq"):

            if (
                len(group) >= 10
                and group["target"].nunique() == 2
            ):

                group_valid = group[
                    [feature, "target"]
                ].dropna()

                if group_valid["target"].nunique() == 2:

                    per_protein_auc.append(
                        roc_auc_score(
                            group_valid["target"],
                            group_valid[feature],
                        )
                    )

        rows.append({
            "feature": feature,
            "pooled_auc": pooled_auc,
            "macro_auc": (
                np.mean(per_protein_auc)
                if per_protein_auc
                else np.nan
            ),
            "proteins_used": len(per_protein_auc),
        })

    return pd.DataFrame(rows).round(3)


print("\nB_Cells, peptide-level features")

print(
    auc_table(
        b,
        PEPTIDE,
        "B_Cells",
    )
)

print("\nB_Cells, protein-level features")

print(
    auc_table(
        b,
        PROTEIN,
        "B_Cells",
    )
)

s15 = s[s["peptide_length"] <= 15]

print("\nSARS 5 to 15 aa")

print(
    auc_table(
        s15,
        PEPTIDE,
        "SARS 5 to 15 aa",
    )
)

print("\nSARS all lengths")

print(
    auc_table(
        s,
        PEPTIDE,
        "SARS all lengths",
    )
)

print(
    "\nInterpretation: AUC = 0.50 indicates no ranking "
    "discrimination. AUC below 0.50 indicates the score "
    "ranks the classes in the opposite direction."
)


# ==========================================================
# CHECK 4: GROUPED CROSS-VALIDATION
# ==========================================================

banner(
    "CHECK 4: GROUPED CROSS-VALIDATION "
    "(5 folds, grouped by protein_seq)"
)

logit = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
    ),
)

tree = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=50,
    class_weight="balanced",
    random_state=42,
)

experiments = [
    (
        "length only, tree",
        ["peptide_length"],
        tree,
    ),
    (
        "4 scores, logistic",
        SCORES_LOG,
        logit,
    ),
    (
        "4 scores + length, tree",
        SCORES_LOG + ["peptide_length"],
        tree,
    ),
    (
        "4 protein features, logistic",
        PROTEIN,
        logit,
    ),
    (
        "4 protein features, tree",
        PROTEIN,
        tree,
    ),
]


def grouped_cv(df, features, model):

    X = df[features].to_numpy(dtype=float)
    y = df["target"].to_numpy(dtype=int)
    groups = df["protein_seq"].to_numpy()

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    oof = np.full(len(df), np.nan)
    fold_auc = []

    for fold_number, (train_idx, test_idx) in enumerate(
        cv.split(X, y, groups),
        start=1,
    ):

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        if np.unique(y_train).size != 2:
            raise ValueError(
                f"Fold {fold_number} training data "
                "does not contain both target classes."
            )

        fitted_model = clone(model)

        fitted_model.fit(
            X_train,
            y_train,
        )

        probabilities = fitted_model.predict_proba(
            X_test
        )[:, 1]

        oof[test_idx] = probabilities

        fold_auc.append(
            calculate_auc(
                y_test,
                probabilities,
            )
        )

    if np.isnan(oof).any():
        raise RuntimeError(
            "Some observations did not receive "
            "out-of-fold predictions."
        )

    overall_auc = calculate_auc(y, oof)

    return {
        "roc_auc_oof": overall_auc,
        "pr_auc_oof": average_precision_score(y, oof),
        "fold_auc_mean": np.nanmean(fold_auc),
        "fold_auc_std": np.nanstd(fold_auc),
        "valid_folds": int(np.isfinite(fold_auc).sum()),
    }


rows = []

for label, features, model in experiments:

    result = grouped_cv(
        b,
        features,
        model,
    )

    result["experiment"] = label
    rows.append(result)

cv_table = (
    pd.DataFrame(rows)
    .set_index("experiment")
    .round(3)
)

print(cv_table)

print(
    "\nPR-AUC baseline, equal to positive prevalence:",
    round(b["target"].mean(), 3),
)


# ==========================================================
# CHECK 5: EXTERNAL TEST
# TRAIN ON B_CELLS, TEST ON SARS
# ==========================================================

banner("CHECK 5: TRAIN ON B_CELLS, TEST ON SARS")

external = [
    (
        "length only, tree",
        ["peptide_length"],
        tree,
    ),
    (
        "4 scores, logistic",
        SCORES_LOG,
        logit,
    ),
    (
        "4 scores + length, tree",
        SCORES_LOG + ["peptide_length"],
        tree,
    ),
]

subsets = {
    "SARS 5-15 aa": s[s["peptide_length"] <= 15],
    "SARS 5-20 aa": s[s["peptide_length"] <= 20],
    "SARS all": s,
}

rows = []

for label, features, model in external:

    fitted_model = clone(model)

    fitted_model.fit(
        b[features].to_numpy(dtype=float),
        b["target"].to_numpy(dtype=int),
    )

    row = {
        "experiment": label,
    }

    for subset_name, subset in subsets.items():

        if subset["target"].nunique() != 2:
            row[f"{subset_name} AUC"] = np.nan
            row[f"{subset_name} n"] = len(subset)
            continue

        probabilities = fitted_model.predict_proba(
            subset[features].to_numpy(dtype=float)
        )[:, 1]

        row[f"{subset_name} AUC"] = roc_auc_score(
            subset["target"],
            probabilities,
        )

        row[f"{subset_name} n"] = len(subset)

    rows.append(row)

external_table = (
    pd.DataFrame(rows)
    .set_index("experiment")
    .round(3)
)

print(external_table)


# ==========================================================
# CHECK 6: CORRELATIONS
# ==========================================================

banner("CHECK 6: CORRELATIONS")

labeled = pd.concat(
    [
        b,
        s[s["peptide_length"] <= 25],
    ],
    ignore_index=True,
)

print(
    "\nPearson with log1p(Emini), "
    "labeled peptides up to 25 aa"
)

print(
    labeled[CORR_COLS]
    .corr(method="pearson")
    .round(3)
)

print("\nSpearman with log1p(Emini)")

print(
    labeled[CORR_COLS]
    .corr(method="spearman")
    .round(3)
)


# ----------------------------------------------------------
# Within-protein Spearman correlations
# B_Cells proteins with at least 50 peptides
# ----------------------------------------------------------

print(
    "\nMedian Spearman inside proteins "
    "with 50+ peptides (B_Cells)"
)

matrices = []

for _, group in b.groupby("protein_seq"):

    if len(group) >= 50:

        correlation_matrix = group[
            CORR_COLS
        ].corr(method="spearman")

        matrices.append(
            correlation_matrix.to_numpy()
        )

print("Proteins used:", len(matrices))

if matrices:

    stacked_matrices = np.stack(matrices)

    median_matrix = np.nanmedian(
        stacked_matrices,
        axis=0,
    )

    print(
        pd.DataFrame(
            median_matrix,
            index=CORR_COLS,
            columns=CORR_COLS,
        ).round(3)
    )

else:

    print(
        "No B_Cells protein groups contain "
        "at least 50 peptides."
    )


# ----------------------------------------------------------
# SARS correlation matrices
# ----------------------------------------------------------

print("\nSpearman, SARS 5 to 15 aa")

s15 = s[s["peptide_length"] <= 15]

print("n =", len(s15))

print(
    s15[CORR_COLS]
    .corr(method="spearman")
    .round(3)
)


print("\nSpearman, SARS up to 25 aa")

s25 = s[s["peptide_length"] <= 25]

print("n =", len(s25))

print(
    s25[CORR_COLS]
    .corr(method="spearman")
    .round(3)
)


# ==========================================================
# COMPLETION
# ==========================================================

print("\n" + "=" * 70)
print("FILE 03 COMPLETED")
print("=" * 70)