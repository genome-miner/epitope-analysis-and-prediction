# ==========================================================
# File: 10_all_models.py
# Purpose: Compare all models on identical nested, cluster-grouped splits.
# Models: logistic, RBF-SVM, boosting, random forest, decision tree,
#         Emini-only baseline, and dummy baseline.
# Outputs: results tables, paired protein-bootstrap comparisons,
#          SARS retest without P59594, and Figures 16 to 20.
# Inputs: Data/B_Cells_cleaned.csv, Data/Sars_cleaned.csv
# Features: chou_fasman, log1p(emini), parker
# ==========================================================

from pathlib import Path
from collections import Counter, defaultdict
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# -----------------------------
# Configuration
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
RESULTS_DIR = BASE_DIR / "Results"
FIG_DIR = BASE_DIR / "Figures"

RUN_SVM = True
RUN_FOREST = True
OUTER_SEED = 42
INNER_SEED = 43
CLUSTER_LINK = 0.20
FEATURES = ["chou_fasman", "emini_log", "parker"]
N_BOOT = 2000
N_PAIRED_PERM = 10000
RANDOM_SEED = 2026

GRIDS = {
    "logistic": {"C": [0.01, 0.1, 1.0, 10.0]},
    "svm_rbf": {"C": [1.0, 4.0], "gamma": [0.1, 0.3333]},
    "boosting": {"max_depth": [2, 3], "learning_rate": [0.05, 0.1]},
    "forest": {"max_depth": [4, 8], "min_samples_leaf": [20, 50]},
    "tree": {"max_depth": [2, 4], "min_samples_leaf": [20, 50]},
}

DEFAULTS = {
    "logistic": {"C": 1.0},
    "svm_rbf": {"C": 1.0, "gamma": 0.3333},
    "boosting": {"max_depth": 3, "learning_rate": 0.05},
    "forest": {"max_depth": 8, "min_samples_leaf": 20},
    "tree": {"max_depth": 4, "min_samples_leaf": 50},
}

TUNED_NAMES = ["logistic"]
if RUN_SVM:
    TUNED_NAMES.append("svm_rbf")
TUNED_NAMES.append("boosting")
if RUN_FOREST:
    TUNED_NAMES.append("forest")
TUNED_NAMES.append("tree")
BASELINES = ["emini_alone", "dummy"]
ALL_NAMES = TUNED_NAMES + BASELINES

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 300)


# -----------------------------
# General helpers
# -----------------------------

def banner(title):
    print("\n" + "=" * 72, flush=True)
    print(title, flush=True)
    print("=" * 72, flush=True)


def require_columns(df, required, filename):
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{filename} is missing required columns: {missing}")


def load_dataset(filename):
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path)
    required = [
        "protein_seq", "peptide_seq", "start_position", "end_position",
        "target", "emini", "chou_fasman", "parker", "parent_protein_id",
    ]
    require_columns(df, required, filename)

    if df["protein_seq"].isna().any() or df["peptide_seq"].isna().any():
        raise ValueError(f"{filename}: protein_seq or peptide_seq contains missing values.")

    df["protein_seq"] = df["protein_seq"].astype(str).str.strip().str.upper()
    df["peptide_seq"] = df["peptide_seq"].astype(str).str.strip().str.upper()
    if (df["protein_seq"].str.len() == 0).any() or (df["peptide_seq"].str.len() == 0).any():
        raise ValueError(f"{filename}: empty protein or peptide sequence found.")

    for col in ["start_position", "end_position", "target", "emini", "chou_fasman", "parker"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if not np.isfinite(df[col].to_numpy(dtype=float)).all():
            raise ValueError(f"{filename}: {col} contains missing or non-finite values.")

    for col in ["start_position", "end_position", "target"]:
        vals = df[col].to_numpy(dtype=float)
        if not np.equal(vals, np.floor(vals)).all():
            raise ValueError(f"{filename}: {col} must contain integer values.")
        df[col] = df[col].astype(int)

    if not set(df["target"].unique()).issubset({0, 1}):
        raise ValueError(f"{filename}: target must contain only 0 and 1.")
    if df["target"].nunique() != 2:
        raise ValueError(f"{filename}: both target classes are required.")
    if (df["emini"] < 0).any():
        raise ValueError(f"{filename}: Emini values must be non-negative before log1p.")
    if (df["start_position"] < 1).any() or (df["end_position"] < df["start_position"]).any():
        raise ValueError(f"{filename}: invalid 1-based inclusive peptide coordinates.")

    df["peptide_length"] = df["end_position"] - df["start_position"] + 1
    if not (df["peptide_length"].to_numpy() == df["peptide_seq"].str.len().to_numpy()).all():
        raise ValueError(f"{filename}: peptide length does not match its coordinates.")
    if (df["end_position"].to_numpy() > df["protein_seq"].str.len().to_numpy()).any():
        raise ValueError(f"{filename}: peptide coordinates exceed parent protein length.")

    # Verify every peptide against its parent sequence using 1-based inclusive coordinates.
    matches = []
    for protein, peptide, start, end in zip(
        df["protein_seq"], df["peptide_seq"], df["start_position"], df["end_position"]
    ):
        matches.append(protein[start - 1:end] == peptide)
    if not all(matches):
        bad = int(np.sum(~np.asarray(matches)))
        raise ValueError(f"{filename}: {bad} peptide sequence(s) do not match their parent coordinates.")

    # Remove peptide sequences with conflicting labels, as in the preceding analyses.
    label_counts = df.groupby("peptide_seq")["target"].nunique()
    conflicting_peptides = label_counts[label_counts > 1].index
    n_conflicts = int(df["peptide_seq"].isin(conflicting_peptides).sum())
    if len(conflicting_peptides):
        df = df[~df["peptide_seq"].isin(conflicting_peptides)].copy()

    # Duplicate coordinates with inconsistent labels are a data error, not silently resolved.
    coordinate_keys = ["protein_seq", "start_position", "end_position"]
    coordinate_label_counts = df.groupby(coordinate_keys, dropna=False)["target"].nunique()
    if (coordinate_label_counts > 1).any():
        raise ValueError(f"{filename}: identical protein coordinates have conflicting labels.")

    before = len(df)
    df = df.drop_duplicates(coordinate_keys).reset_index(drop=True)
    n_duplicates = before - len(df)
    df["emini_log"] = np.log1p(df["emini"].astype(float))

    for col in FEATURES:
        if not np.isfinite(df[col].to_numpy(dtype=float)).all():
            raise ValueError(f"{filename}: feature {col} contains non-finite values after cleaning.")

    print(
        f"{filename}: retained {len(df):,} rows, removed {n_conflicts:,} rows from "
        f"conflicting peptide labels and {n_duplicates:,} duplicate coordinate rows.",
        flush=True,
    )
    return df


def group_indices(groups):
    groups = np.asarray(groups)
    if len(groups) == 0:
        return []
    mapping = {}
    for i, value in enumerate(groups):
        mapping.setdefault(value, []).append(i)
    return [np.asarray(idx, dtype=int) for idx in mapping.values()]


def fast_auc(y, scores):
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(y) != len(scores) or len(y) == 0:
        raise ValueError("AUC requires equally sized, non-empty label and score arrays.")
    if not np.isfinite(scores).all():
        raise ValueError("AUC scores contain non-finite values.")
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC is undefined unless both target classes are present.")
    ranks = rankdata(scores, method="average")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def macro_auc(y, scores, exact_groups, min_n=10):
    values = []
    for idx in group_indices(exact_groups):
        if len(idx) >= min_n and 0 < np.sum(y[idx] == 1) < len(idx):
            values.append(fast_auc(y[idx], scores[idx]))
    if not values:
        raise ValueError("No eligible protein groups were available for macro AUC.")
    return float(np.mean(values))


def make_model(name, params):
    if name == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, **params),
        )
    if name == "svm_rbf":
        return make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", class_weight="balanced", **params),
        )
    if name == "forest":
        return RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            n_jobs=1,
            random_state=42,
            **params,
        )
    if name == "tree":
        return DecisionTreeClassifier(class_weight="balanced", random_state=42, **params)
    if name == "boosting":
        return HistGradientBoostingClassifier(
            max_iter=150,
            min_samples_leaf=50,
            l2_regularization=1.0,
            random_state=42,
            **params,
        )
    raise ValueError(f"Unknown model name: {name}")


def fit_model(name, params, X, y):
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) != 2:
        raise ValueError(f"Cannot train {name}: training fold does not contain both classes.")
    model = make_model(name, params)
    if name == "boosting":
        counts = np.bincount(y, minlength=2)
        if np.any(counts == 0):
            raise ValueError("Boosting training fold is missing a target class.")
        class_weights = len(y) / (2.0 * counts)
        model.fit(X, y, sample_weight=class_weights[y])
    else:
        model.fit(X, y)
    return model


def score_of(model, X):
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    return np.asarray(model.decision_function(X), dtype=float)


def best_threshold(y, scores):
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(y)) != 2 or not np.isfinite(scores).all():
        raise ValueError("Threshold selection requires both classes and finite scores.")
    candidates = np.unique(np.quantile(scores, np.linspace(0.40, 0.95, 56)))
    values = [matthews_corrcoef(y, (scores >= threshold).astype(int)) for threshold in candidates]
    return float(candidates[int(np.argmax(values))])


def threshold_metrics(y, predictions):
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    return {
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "accuracy": (tp + tn) / len(y),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "f1": f1_score(y, predictions, zero_division=0),
        "macro_f1": f1_score(y, predictions, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y, predictions),
    }


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    if len(p_values) == 0:
        return np.asarray([], dtype=float)
    if not np.isfinite(p_values).all() or np.any((p_values < 0) | (p_values > 1)):
        raise ValueError("P-values for Holm correction must be finite and between 0 and 1.")
    order = np.argsort(p_values)
    m = len(p_values)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def bootstrap_ci(values, rng, n_boot=N_BOOT):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        raise ValueError("Cannot bootstrap an empty set of values.")
    draws = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[draws].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def paired_signflip_p(differences, rng, n_perm=N_PAIRED_PERM):
    """Two-sided paired sign-flip permutation test on per-protein differences."""
    differences = np.asarray(differences, dtype=float)
    if len(differences) == 0 or not np.isfinite(differences).all():
        raise ValueError("Paired test requires finite, non-empty differences.")
    observed = abs(float(np.mean(differences)))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, len(differences)))
    null_means = np.mean(signs * differences[None, :], axis=1)
    return float((1 + np.sum(np.abs(null_means) >= observed)) / (n_perm + 1))


def topk_gain(scores, y, eligible_groups, fraction=0.10):
    gains = []
    for idx in eligible_groups:
        k = max(1, int(np.ceil(fraction * len(idx))))
        selected = idx[np.argsort(-scores[idx], kind="stable")[:k]]
        gains.append(float(np.mean(y[selected]) - np.mean(y[idx])))
    return np.asarray(gains, dtype=float)


# -----------------------------
# Load and validate data
# -----------------------------

banner("STEP 0: LOAD AND VALIDATE")
b = load_dataset("B_Cells_cleaned.csv")
s = load_dataset("Sars_cleaned.csv")

y_b = b["target"].to_numpy(dtype=int)
g_exact = b["protein_seq"].to_numpy()
X_b = b[FEATURES].to_numpy(dtype=float)
emini_b = b["emini_log"].to_numpy(dtype=float)
is_p59594 = b["parent_protein_id"].astype(str).to_numpy() == "P59594"

X_s = s[FEATURES].to_numpy(dtype=float)
y_s = s["target"].to_numpy(dtype=int)
len_s = s["peptide_length"].to_numpy(dtype=int)
emini_s = s["emini_log"].to_numpy(dtype=float)

print(f"B_Cells rows: {len(b):,}", flush=True)
print(f"SARS rows: {len(s):,}", flush=True)
print(f"P59594 B_Cells rows: {int(is_p59594.sum()):,}", flush=True)
print(f"Models: {TUNED_NAMES}", flush=True)

# Main B-Cells nested CV retains P59594, matching File 9's main Check 2.
# P59594 is excluded only from the separate SARS retest, matching File 9's Check 5.

eligible_groups = [
    idx for idx in group_indices(g_exact)
    if len(idx) >= 10 and 0 < np.sum(y_b[idx] == 1) < len(idx)
]
if len(eligible_groups) < 2:
    raise ValueError("Too few eligible protein groups for protein-level evaluation.")
print(f"Proteins eligible for macro AUC: {len(eligible_groups)}", flush=True)


# -----------------------------
# Cluster grouping, same 5-mer overlap method and 0.20 threshold as File 9
# -----------------------------

banner("CHECK 1: CLUSTER GROUPING FOR ALL SPLITS")

def kmers(sequence, k=5):
    if len(sequence) < k:
        return set()
    return {sequence[i:i + k] for i in range(len(sequence) - k + 1)}


def pair_overlaps(sequences):
    kmer_sets = [kmers(sequence) for sequence in sequences]
    inverted = defaultdict(list)
    for i, kmer_set in enumerate(kmer_sets):
        for kmer in kmer_set:
            inverted[kmer].append(i)
    shared_counts = Counter()
    for members in inverted.values():
        if len(members) > 1:
            for a in range(len(members)):
                for c in range(a + 1, len(members)):
                    shared_counts[(members[a], members[c])] += 1
    overlaps = {}
    for (i, j), count in shared_counts.items():
        denominator = min(len(kmer_sets[i]), len(kmer_sets[j]))
        if denominator:
            overlaps[(i, j)] = count / denominator
    return overlaps


unique_sequences = list(pd.unique(g_exact))
sequence_to_index = {sequence: i for i, sequence in enumerate(unique_sequences)}
overlap_fractions = pair_overlaps(unique_sequences)
parents = list(range(len(unique_sequences)))


def find_root(index):
    while parents[index] != index:
        parents[index] = parents[parents[index]]
        index = parents[index]
    return index


for (i, j), fraction in overlap_fractions.items():
    if fraction >= CLUSTER_LINK:
        root_i, root_j = find_root(i), find_root(j)
        if root_i != root_j:
            parents[root_i] = root_j

cluster_labels = np.asarray([find_root(i) for i in range(len(unique_sequences))])
G_SPLIT = np.asarray([cluster_labels[sequence_to_index[sequence]] for sequence in g_exact])
print(f"Unique protein sequences: {len(unique_sequences)}", flush=True)
print(f"Clusters at overlap threshold {CLUSTER_LINK:.2f}: {len(np.unique(G_SPLIT))}", flush=True)
print("All related protein sequences within each cluster stay in the same CV fold.", flush=True)


# -----------------------------
# Reproducible random generators and protein-level metrics
# -----------------------------

rng_boot = np.random.default_rng(RANDOM_SEED)
rng_tests = np.random.default_rng(RANDOM_SEED + 1)


def per_protein_auc(scores):
    return np.asarray([fast_auc(y_b[idx], scores[idx]) for idx in eligible_groups])


# -----------------------------
# Nested CV helpers
# -----------------------------

def grid_search(name, X, y, split_groups, exact_groups, n_splits, seed):
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    try:
        splits = list(cv.split(X, y, split_groups))
    except ValueError as exc:
        raise ValueError(f"{name}: grouped CV could not create {n_splits} folds: {exc}") from exc

    best_score = -np.inf
    best_params = None
    best_oof = None
    for params in ParameterGrid(GRIDS[name]):
        oof_scores = np.full(len(y), np.nan, dtype=float)
        for train_idx, valid_idx in splits:
            if len(np.unique(y[train_idx])) != 2:
                raise ValueError(f"{name}: an inner training fold contains only one class.")
            model = fit_model(name, params, X[train_idx], y[train_idx])
            oof_scores[valid_idx] = score_of(model, X[valid_idx])
        if not np.isfinite(oof_scores).all():
            raise ValueError(f"{name}: inner out-of-fold predictions are incomplete.")
        score = macro_auc(y, oof_scores, exact_groups)
        if score > best_score:
            best_score = score
            best_params = params
            best_oof = oof_scores.copy()
    return best_score, best_params, best_oof


def nested_cv(name, outer_splits):
    tuned_scores = np.full(len(y_b), np.nan, dtype=float)
    default_scores = np.full(len(y_b), np.nan, dtype=float)
    predictions = np.full(len(y_b), -1, dtype=int)
    chosen_params = []

    for fold_number, (train_idx, test_idx) in enumerate(outer_splits, start=1):
        if len(np.unique(y_b[train_idx])) != 2:
            raise ValueError(f"{name}: outer training fold {fold_number} lacks a class.")
        _, params, inner_oof = grid_search(
            name,
            X_b[train_idx],
            y_b[train_idx],
            G_SPLIT[train_idx],
            g_exact[train_idx],
            n_splits=3,
            seed=INNER_SEED,
        )
        threshold = best_threshold(y_b[train_idx], inner_oof)
        chosen_params.append(params)

        tuned_model = fit_model(name, params, X_b[train_idx], y_b[train_idx])
        tuned_fold_scores = score_of(tuned_model, X_b[test_idx])
        tuned_scores[test_idx] = tuned_fold_scores
        predictions[test_idx] = (tuned_fold_scores >= threshold).astype(int)

        default_model = fit_model(name, DEFAULTS[name], X_b[train_idx], y_b[train_idx])
        default_scores[test_idx] = score_of(default_model, X_b[test_idx])
        print(f"    {name}: outer fold {fold_number}/5 complete", flush=True)

    if not np.isfinite(tuned_scores).all() or not np.isfinite(default_scores).all():
        raise ValueError(f"{name}: incomplete outer out-of-fold predictions.")
    if np.any(predictions < 0):
        raise ValueError(f"{name}: incomplete outer-fold class predictions.")
    return tuned_scores, default_scores, predictions, chosen_params


# -----------------------------
# Nested grouped CV for every tuned model
# -----------------------------

banner("CHECK 2: NESTED GROUPED CV FOR ALL MODELS")
outer_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=OUTER_SEED)
try:
    OUTER_SPLITS = list(outer_cv.split(X_b, y_b, G_SPLIT))
except ValueError as exc:
    raise ValueError(f"Outer grouped CV could not create five folds: {exc}") from exc

FOLD_ID = np.full(len(y_b), -1, dtype=int)
for fold_id, (_, test_idx) in enumerate(OUTER_SPLITS):
    FOLD_ID[test_idx] = fold_id
if np.any(FOLD_ID < 0):
    raise ValueError("Outer CV did not assign every row to exactly one test fold.")

TUNED_SCORES = {}
DEFAULT_SCORES = {}
PREDICTIONS = {}
CHOSEN_PARAMS = {}

for model_name in TUNED_NAMES:
    print(f"\nStarting model: {model_name}", flush=True)
    (TUNED_SCORES[model_name], DEFAULT_SCORES[model_name],
     PREDICTIONS[model_name], CHOSEN_PARAMS[model_name]) = nested_cv(model_name, OUTER_SPLITS)
    print(f"Finished model: {model_name}", flush=True)

# Emini-only baseline uses the same outer folds and training-fold threshold selection.
TUNED_SCORES["emini_alone"] = emini_b.copy()
PREDICTIONS["emini_alone"] = np.full(len(y_b), -1, dtype=int)
for train_idx, test_idx in OUTER_SPLITS:
    threshold = best_threshold(y_b[train_idx], emini_b[train_idx])
    PREDICTIONS["emini_alone"][test_idx] = (emini_b[test_idx] >= threshold).astype(int)

# A true constant-score dummy baseline has AUC 0.5 and AP equal to prevalence.
TUNED_SCORES["dummy"] = np.full(len(y_b), 0.5, dtype=float)

print("\nChosen hyperparameters by outer fold:", flush=True)
for model_name in TUNED_NAMES:
    print(model_name, flush=True)
    for params in CHOSEN_PARAMS[model_name]:
        print("   ", params, flush=True)


# -----------------------------
# Fold-wise score ranking for pooled descriptive metrics
# -----------------------------

def rank_within_outer_fold(scores):
    ranked = np.full(len(scores), np.nan, dtype=float)
    for fold_id in range(5):
        idx = np.where(FOLD_ID == fold_id)[0]
        ranked[idx] = rankdata(scores[idx], method="average") / len(idx)
    if not np.isfinite(ranked).all():
        raise ValueError("Fold-wise ranking produced missing values.")
    return ranked


FOLD_RANKS = {name: rank_within_outer_fold(scores) for name, scores in TUNED_SCORES.items()}
# Keep the dummy baseline truly constant across all folds.
FOLD_RANKS["dummy"] = np.full(len(y_b), 0.5, dtype=float)
PER_PROTEIN = {
    name: per_protein_auc(TUNED_SCORES[name])
    for name in TUNED_NAMES + ["emini_alone"]
}
PER_PROTEIN["dummy"] = np.full(len(eligible_groups), 0.5, dtype=float)


# -----------------------------
# Results table
# -----------------------------

banner("CHECK 3: RESULTS TABLE")
result_rows = []
for model_name in ALL_NAMES:
    protein_aucs = PER_PROTEIN[model_name]
    low, high = bootstrap_ci(protein_aucs, rng_boot)
    row = {
        "model": model_name,
        "macro_auc": float(np.mean(protein_aucs)),
        "low": float(low),
        "high": float(high),
        "pooled_auc_foldrank": float(roc_auc_score(y_b, FOLD_RANKS[model_name])),
        "pr_auc_foldrank": float(average_precision_score(y_b, FOLD_RANKS[model_name])),
    }
    if model_name != "dummy":
        gains = topk_gain(TUNED_SCORES[model_name], y_b, eligible_groups)
        gain_low, gain_high = bootstrap_ci(gains, rng_boot)
        row.update({
            "top10_gain": float(np.mean(gains)),
            "gain_low": float(gain_low),
            "gain_high": float(gain_high),
        })
    if model_name in DEFAULT_SCORES:
        default_protein_aucs = per_protein_auc(DEFAULT_SCORES[model_name])
        row["macro_default"] = float(np.mean(default_protein_aucs))
        row["tuned_minus_default"] = float(np.mean(protein_aucs - default_protein_aucs))
    result_rows.append(row)

results = pd.DataFrame(result_rows)
print("Positive-class prevalence / PR-AUC floor:", round(float(y_b.mean()), 4), flush=True)
print(results.round(4).to_string(index=False), flush=True)
print("Pooled AUC and PR-AUC use scores ranked within each outer fold.", flush=True)
print("Protein-level macro AUC is the primary comparison.", flush=True)
results.to_csv(RESULTS_DIR / "10_results_table.csv", index=False)


# -----------------------------
# Threshold metrics
# -----------------------------

banner("CHECK 4: THRESHOLD METRICS")
threshold_rows = []
for label, prediction in [
    ("all_negative", np.zeros(len(y_b), dtype=int)),
    ("all_positive", np.ones(len(y_b), dtype=int)),
]:
    threshold_rows.append({"model": label, **threshold_metrics(y_b, prediction)})
for model_name in TUNED_NAMES + ["emini_alone"]:
    threshold_rows.append({"model": model_name, **threshold_metrics(y_b, PREDICTIONS[model_name])})

threshold_table = pd.DataFrame(threshold_rows)
threshold_table["share_called_positive"] = (
    threshold_table["TP"] + threshold_table["FP"]
) / len(y_b)
print(threshold_table.round(4).to_string(index=False), flush=True)
print("Thresholds are selected using training-fold data only.", flush=True)
print("Use ranking metrics, not accuracy alone, to compare model discrimination.", flush=True)
threshold_table.to_csv(RESULTS_DIR/ "10_threshold_metrics.csv", index=False)


# -----------------------------
# Paired protein-level comparisons with Holm correction
# -----------------------------

banner("CHECK 5: PAIRED PROTEIN-LEVEL TESTS WITH HOLM CORRECTION")
comparison_names = TUNED_NAMES + ["emini_alone"]
pair_rows = []
for i in range(len(comparison_names)):
    for j in range(i + 1, len(comparison_names)):
        first, second = comparison_names[i], comparison_names[j]
        differences = PER_PROTEIN[first] - PER_PROTEIN[second]
        low, high = bootstrap_ci(differences, rng_boot)
        p_value = paired_signflip_p(differences, rng_tests)
        pair_rows.append({
            "first": first,
            "second": second,
            "mean_macro_auc_difference": float(np.mean(differences)),
            "bootstrap_low": float(low),
            "bootstrap_high": float(high),
            "paired_signflip_p": p_value,
        })

pairwise = pd.DataFrame(pair_rows)
pairwise["p_holm"] = holm_adjust(pairwise["paired_signflip_p"].to_numpy())
pairwise["significant_after_holm"] = pairwise["p_holm"] < 0.05
print(pairwise.round(4).to_string(index=False), flush=True)
print(f"Holm correction covers {len(pairwise)} pairwise model comparisons.", flush=True)
pairwise.to_csv(RESULTS_DIR / "10_pairwise_tests.csv", index=False)

# Tuned-versus-default comparisons are a separate, explicitly labelled family.
tuned_default_rows = []
for model_name in TUNED_NAMES:
    differences = PER_PROTEIN[model_name] - per_protein_auc(DEFAULT_SCORES[model_name])
    low, high = bootstrap_ci(differences, rng_boot)
    tuned_default_rows.append({
        "model": model_name,
        "mean_tuned_minus_default": float(np.mean(differences)),
        "bootstrap_low": float(low),
        "bootstrap_high": float(high),
        "paired_signflip_p": paired_signflip_p(differences, rng_tests),
    })
tuned_default = pd.DataFrame(tuned_default_rows)
tuned_default["p_holm"] = holm_adjust(tuned_default["paired_signflip_p"].to_numpy())
print("\nTuned versus default settings:", flush=True)
print(tuned_default.round(4).to_string(index=False), flush=True)
tuned_default.to_csv(RESULTS_DIR/ "10_tuned_vs_default_tests.csv", index=False)

# Each model versus the dummy AUC of 0.5, with a separate Holm family.
dummy_rows = []
for model_name in comparison_names:
    differences = PER_PROTEIN[model_name] - 0.5
    low, high = bootstrap_ci(differences, rng_boot)
    dummy_rows.append({
        "model": model_name,
        "mean_macro_auc_minus_0.5": float(np.mean(differences)),
        "bootstrap_low": float(low),
        "bootstrap_high": float(high),
        "paired_signflip_p": paired_signflip_p(differences, rng_tests),
    })
dummy_tests = pd.DataFrame(dummy_rows)
dummy_tests["p_holm"] = holm_adjust(dummy_tests["paired_signflip_p"].to_numpy())
print("\nEach model versus dummy macro AUC 0.5:", flush=True)
print(dummy_tests.round(4).to_string(index=False), flush=True)
dummy_tests.to_csv(RESULTS_DIR / "10_model_vs_dummy_tests.csv", index=False)


# -----------------------------
# SARS retest, trained without P59594
# -----------------------------

banner("CHECK 6: SARS RETEST, TRAINED WITHOUT P59594")
train_mask = ~is_p59594
if train_mask.sum() == 0:
    raise ValueError("No B_Cells rows remain after excluding P59594.")
if len(np.unique(y_b[train_mask])) != 2:
    raise ValueError("Both B_Cells target classes are required after excluding P59594.")

FINAL_MODELS = {}
for model_name in TUNED_NAMES:
    print(f"Tuning SARS-ret test model: {model_name}", flush=True)
    _, params, _ = grid_search(
        model_name,
        X_b[train_mask],
        y_b[train_mask],
        G_SPLIT[train_mask],
        g_exact[train_mask],
        n_splits=5,
        seed=44,
    )
    fitted = fit_model(model_name, params, X_b[train_mask], y_b[train_mask])
    FINAL_MODELS[model_name] = (params, fitted)
    print(f"  {model_name} selected settings: {params}", flush=True)

SARS_SCORES = {name: score_of(FINAL_MODELS[name][1], X_s) for name in TUNED_NAMES}
SARS_SCORES["emini_alone"] = emini_s.copy()

SARS_SUBSETS = {
    "5-15 aa": (len_s >= 5) & (len_s <= 15),
    "5-14 aa": (len_s >= 5) & (len_s <= 14),
    "17 aa": len_s == 17,
    "all": np.ones(len_s.shape, dtype=bool),
}

sars_rng = np.random.default_rng(RANDOM_SEED + 2)
sars_rows = []
for subset_name, subset_mask in SARS_SUBSETS.items():
    indices = np.where(subset_mask)[0]
    if len(indices) == 0:
        raise ValueError(f"SARS subset {subset_name} is empty.")
    subset_y = y_s[indices]
    if len(np.unique(subset_y)) != 2:
        raise ValueError(f"SARS subset {subset_name} does not contain both target classes.")

    # Paired peptide bootstrap: every scorer uses the same sampled peptide indices.
    bootstrap_auc = {name: [] for name in SARS_SCORES}
    bootstrap_differences = {name: [] for name in TUNED_NAMES}
    for _ in range(N_BOOT):
        sampled = sars_rng.choice(indices, size=len(indices), replace=True)
        sampled_y = y_s[sampled]
        if len(np.unique(sampled_y)) != 2:
            continue
        for scorer_name, scorer_values in SARS_SCORES.items():
            bootstrap_auc[scorer_name].append(fast_auc(sampled_y, scorer_values[sampled]))
        reference_auc = fast_auc(sampled_y, SARS_SCORES["emini_alone"][sampled])
        for scorer_name in TUNED_NAMES:
            model_auc = fast_auc(sampled_y, SARS_SCORES[scorer_name][sampled])
            bootstrap_differences[scorer_name].append(model_auc - reference_auc)

    valid_bootstraps = len(bootstrap_auc["emini_alone"])
    if valid_bootstraps < int(0.90 * N_BOOT):
        raise ValueError(
            f"SARS subset {subset_name}: only {valid_bootstraps}/{N_BOOT} valid paired bootstrap samples."
        )

    for scorer_name, scorer_values in SARS_SCORES.items():
        auc_value = fast_auc(y_s[indices], scorer_values[indices])
        auc_low, auc_high = np.percentile(bootstrap_auc[scorer_name], [2.5, 97.5])
        row = {
            "subset": subset_name,
            "rows": int(len(indices)),
            "positives": int(np.sum(y_s[indices] == 1)),
            "scorer": scorer_name,
            "auc": auc_value,
            "low": float(auc_low),
            "high": float(auc_high),
            "valid_paired_bootstraps": valid_bootstraps,
        }
        if scorer_name in TUNED_NAMES:
            diffs = np.asarray(bootstrap_differences[scorer_name], dtype=float)
            diff_low, diff_high = np.percentile(diffs, [2.5, 97.5])
            row["difference_vs_emini"] = auc_value - fast_auc(
                y_s[indices], SARS_SCORES["emini_alone"][indices]
            )
            row["difference_low"] = float(diff_low)
            row["difference_high"] = float(diff_high)
        sars_rows.append(row)

sars_results = pd.DataFrame(sars_rows)
print(sars_results.round(4).to_string(index=False), flush=True)
print("SARS intervals use paired peptide bootstrap samples.", flush=True)
print("SARS contains one protein with overlapping peptides, so these intervals", flush=True)
print("describe peptide-level resampling and do not represent independent-protein uncertainty.", flush=True)
print("The 5-15 aa and 5-14 aa subsets have different sample sizes.", flush=True)
print("Interpret the smaller 5-14 aa subset cautiously.", flush=True)
sars_results.to_csv(RESULTS_DIR / "10_sars.csv", index=False)


# -----------------------------
# Figures 16 to 20
# -----------------------------

banner("FIGURES 16 TO 20")
color_values = plt.cm.tab10(np.linspace(0, 1, len(ALL_NAMES)))
COLORS = dict(zip(ALL_NAMES, color_values))

# Figure 16: protein-level macro AUC with protein-bootstrap intervals.
fig, ax = plt.subplots(figsize=(10, 5.5))
ordered = results.sort_values("macro_auc", ascending=False).reset_index(drop=True)
x_values = np.arange(len(ordered))
lower_errors = np.maximum(0, ordered["macro_auc"].to_numpy() - ordered["low"].to_numpy())
upper_errors = np.maximum(0, ordered["high"].to_numpy() - ordered["macro_auc"].to_numpy())
ax.errorbar(
    x_values,
    ordered["macro_auc"],
    yerr=[lower_errors, upper_errors],
    fmt="o", capsize=5, markersize=8, color="black",
)
ax.axhline(0.5, color="gray", linestyle="--", label="chance")
ax.set_xticks(x_values)
ax.set_xticklabels(ordered["model"], rotation=20, ha="right")
ax.set_ylim(0.35, 1.0)
ax.set_ylabel("Macro within-protein AUC")
ax.set_title("Figure 16: All Models, 95% Protein-Bootstrap Intervals")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "Fig16_All_Models.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Figure 17: pooled ROC and PR curves from fold-wise ranked scores.
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for model_name in TUNED_NAMES + ["emini_alone"]:
    fold_scores = FOLD_RANKS[model_name]
    fpr, tpr, _ = roc_curve(y_b, fold_scores)
    axes[0].plot(
        fpr, tpr, color=COLORS[model_name],
        label=f"{model_name} ({roc_auc_score(y_b, fold_scores):.3f})",
    )
    precision, recall, _ = precision_recall_curve(y_b, fold_scores)
    axes[1].plot(
        recall, precision, color=COLORS[model_name],
        label=f"{model_name} ({average_precision_score(y_b, fold_scores):.3f})",
    )
axes[0].plot([0, 1], [0, 1], "k--", label="chance")
axes[0].set_xlabel("False positive rate")
axes[0].set_ylabel("True positive rate")
axes[0].set_title("ROC")
axes[0].legend(loc="lower right", fontsize=8)
axes[1].axhline(y_b.mean(), color="black", linestyle="--", label=f"prevalence {y_b.mean():.3f}")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall")
axes[1].legend(loc="upper right", fontsize=8)
fig.suptitle("Figure 17: ROC and PR, Scores Ranked Within Each Outer Fold", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(FIG_DIR / "Fig17_ROC_PR.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Figure 18: confusion matrices using training-fold-selected thresholds.
confusion_names = TUNED_NAMES + ["emini_alone"]
n_columns = 3
n_rows = int(np.ceil(len(confusion_names) / n_columns))
fig, axes = plt.subplots(
    n_rows, n_columns,
    figsize=(4.6 * n_columns, 4.4 * n_rows),
    squeeze=False,
)
for axis in axes.ravel():
    axis.axis("off")
for axis, model_name in zip(axes.ravel(), confusion_names):
    axis.axis("on")
    matrix = confusion_matrix(y_b, PREDICTIONS[model_name], labels=[0, 1])
    row_totals = matrix.sum(axis=1, keepdims=True)
    percentages = np.divide(matrix, row_totals, out=np.zeros_like(matrix, dtype=float), where=row_totals != 0)
    axis.imshow(percentages, cmap="Blues", vmin=0, vmax=1)
    for row in range(2):
        for column in range(2):
            axis.text(
                column, row,
                f"{matrix[row, column]}\n{percentages[row, column]:.0%}",
                ha="center", va="center", fontsize=12,
                color="white" if percentages[row, column] > 0.6 else "black",
            )
    axis.set_xticks([0, 1])
    axis.set_xticklabels(["Predicted 0", "Predicted 1"])
    axis.set_yticks([0, 1])
    axis.set_yticklabels(["True 0", "True 1"])
    axis.set_title(model_name)
fig.suptitle("Figure 18: Confusion Matrices, Row Percentages", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(FIG_DIR / "Fig18_Confusion_All.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Figure 19: tuned versus default settings.
fig, ax = plt.subplots(figsize=(10, 5.5))
x_values = np.arange(len(TUNED_NAMES))
for offset, (label, column) in enumerate([
    ("default", "macro_default"),
    ("tuned (nested)", "macro_auc"),
]):
    values = results.set_index("model").loc[TUNED_NAMES, column].to_numpy(dtype=float)
    ax.plot(x_values + (offset - 0.5) * 0.15, values, "o", markersize=9, label=label)
ax.axhline(0.5, color="gray", linestyle="--", label="chance")
ax.set_xticks(x_values)
ax.set_xticklabels(TUNED_NAMES, rotation=15, ha="right")
ax.set_ylim(0.35, 1.0)
ax.set_ylabel("Macro within-protein AUC")
ax.set_title("Figure 19: Tuned Against Default Settings")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "Fig19_Tuned_vs_Default.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Figure 20: SARS retest with paired peptide-bootstrap intervals.
fig, ax = plt.subplots(figsize=(11, 5.5))
subset_names = list(SARS_SUBSETS)
scorer_names = TUNED_NAMES + ["emini_alone"]
bar_width = 0.8 / len(scorer_names)
for scorer_index, scorer_name in enumerate(scorer_names):
    part = sars_results[sars_results["scorer"] == scorer_name].set_index("subset").loc[subset_names]
    positions = np.arange(len(subset_names)) + scorer_index * bar_width
    lower = np.maximum(0, part["auc"].to_numpy() - part["low"].to_numpy())
    upper = np.maximum(0, part["high"].to_numpy() - part["auc"].to_numpy())
    ax.errorbar(
        positions, part["auc"], yerr=[lower, upper],
        fmt="o", capsize=3, color=COLORS[scorer_name], label=scorer_name,
    )
ax.axhline(0.5, color="gray", linestyle="--")
ax.set_xticks(np.arange(len(subset_names)) + bar_width * (len(scorer_names) - 1) / 2)
ax.set_xticklabels(subset_names)
ax.set_ylim(0.35, 1.0)
ax.set_ylabel("AUC on SARS")
ax.set_title("Figure 20: SARS Retest Without P59594, Peptide Bootstrap")
ax.legend(fontsize=8, ncol=3)
fig.tight_layout()
fig.savefig(FIG_DIR / "Fig20_SARS_All.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("Saved Figures 16 to 20.", flush=True)

banner("FILE 10 COMPLETED SUCCESSFULLY")
print("Tables saved in Data/ and Figures 16 to 20 saved in Figures/.", flush=True)
