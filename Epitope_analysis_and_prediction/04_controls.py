# ==========================================================
# File: 04_controls.py
# Purpose: Compare feature sets and models using controlled tests.
# Input: Data/B_Cells_cleaned.csv, Data/Sars_cleaned.csv
# ==========================================================

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ----------------------------------------------------------
# Settings
# ----------------------------------------------------------

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 100)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"

AA = list("ACDEFGHIKLMNPQRSTVWY")
COMP = ["aa_" + a for a in AA]

SCORES_LOG = [
    "chou_fasman",
    "emini_log",
    "kolaskar_tongaonkar",
    "parker"
]

PROTEIN = [
    "isoelectric_point",
    "aromaticity",
    "hydrophobicity",
    "stability"
]

SEEDS = [42, 7, 2024]


# ----------------------------------------------------------
# Helper functions
# ----------------------------------------------------------

def banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load(name):

    path = DATA_DIR / name

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    required = [
        "peptide_seq",
        "protein_seq",
        "start_position",
        "end_position",
        "target",
        "emini",
        "chou_fasman",
        "kolaskar_tongaonkar",
        "parker",
        "isoelectric_point",
        "aromaticity",
        "hydrophobicity",
        "stability"
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns in {name}: {missing}")

    if df[required].isnull().any().any():
        raise ValueError(f"Missing values found in {name}")

    df["target"] = pd.to_numeric(df["target"], errors="coerce")

    if df["target"].isnull().any():
        raise ValueError(f"Invalid target values in {name}")

    if not set(df["target"].unique()).issubset({0, 1}):
        raise ValueError(f"Target must contain only 0 and 1 in {name}")

    df["start_position"] = pd.to_numeric(
        df["start_position"], errors="coerce"
    )

    df["end_position"] = pd.to_numeric(
        df["end_position"], errors="coerce"
    )

    if df[["start_position", "end_position"]].isnull().any().any():
        raise ValueError(f"Invalid coordinates in {name}")

    df["peptide_length"] = (
        df["end_position"] - df["start_position"] + 1
    )

    if (df["peptide_length"] <= 0).any():
        raise ValueError(f"Invalid peptide lengths in {name}")

    if (df["emini"] < 0).any():
        raise ValueError(f"Negative Emini values found in {name}")

    # Remove peptides with conflicting target labels.
    label_counts = df.groupby("peptide_seq")["target"].nunique()
    conflicts = label_counts[label_counts > 1].index

    df = df[~df["peptide_seq"].isin(conflicts)].copy()

    # Remove repeated protein-coordinate records.
    df = df.drop_duplicates(
        ["protein_seq", "start_position", "end_position"]
    )

    df = df.reset_index(drop=True)

    # Log-transform Emini scores.
    df["emini_log"] = np.log1p(df["emini"])

    # Calculate amino acid fractions.
    seq_len = df["peptide_seq"].str.len()

    if (seq_len == 0).any():
        raise ValueError(f"Empty peptide sequence found in {name}")

    for aa in AA:
        df["aa_" + aa] = (
            df["peptide_seq"].str.count(aa) / seq_len
        )

    return df


def safe_auc(y, scores):

    if len(np.unique(y)) < 2:
        return np.nan

    return roc_auc_score(y, scores)


def macro_auc(y, scores, groups, min_n=10):

    frame = pd.DataFrame({
        "y": y,
        "scores": scores,
        "group": groups
    })

    values = []

    for _, group in frame.groupby("group"):
        if len(group) >= min_n and group["y"].nunique() == 2:
            values.append(
                roc_auc_score(group["y"], group["scores"]))

    if not values:
        return np.nan, 0
    return float(np.mean(values)), len(values)


# ----------------------------------------------------------
# STEP 0: Load data
# ----------------------------------------------------------

banner("STEP 0: LOAD")

b = load("B_Cells_cleaned.csv")
s = load("Sars_cleaned.csv")

y_b = b["target"].to_numpy()
g_b = b["protein_seq"].to_numpy()

y_s = s["target"].to_numpy()

print("B_Cells rows:", len(b),
      "positive rate:", round(y_b.mean(), 3))

print("SARS rows:", len(s),
      "positive rate:", round(y_s.mean(), 3))


# Proteins eligible for macro AUC.
ELIG = [
    idx
    for idx in b.groupby("protein_seq").indices.values()
    if len(idx) >= 10 and len(np.unique(y_b[idx])) == 2
]

print("Proteins with 10+ peptides and both classes:", len(ELIG))

def pp_auc(scores):

    values = [
        roc_auc_score(y_b[idx], scores[idx])
        for idx in ELIG
    ]

    return np.array(values)


# ----------------------------------------------------------
# Models and feature sets
# ----------------------------------------------------------

MODELS = {

    "logistic": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000
        )
    ),

    "forest": RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        min_samples_leaf=20,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )
}


FEATURE_SETS = {
    "length": ["peptide_length"],
    "scores": SCORES_LOG,
    "scores+length": SCORES_LOG + ["peptide_length"],
    "scores+composition": SCORES_LOG + COMP,
    "scores+composition+length":
        SCORES_LOG + COMP + ["peptide_length"],
    "protein": PROTEIN,
    "scores+protein": SCORES_LOG + PROTEIN
}


# ----------------------------------------------------------
# Grouped cross-validation
# ----------------------------------------------------------

def oof_predict(df, features, model, seed):

    X = df[features].to_numpy()
    y = df["target"].to_numpy()
    groups = df["protein_seq"].to_numpy()

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed
    )

    oof = np.zeros(len(df))
    for train_idx, test_idx in cv.split(X, y, groups):
        fitted_model = clone(model)
        fitted_model.fit(
            X[train_idx],
            y[train_idx]
        )
        oof[test_idx] = fitted_model.predict_proba(
            X[test_idx]
        )[:, 1]
    return oof


# ----------------------------------------------------------
# CHECK 1: Grouped CV
# ----------------------------------------------------------

banner("CHECK 1: GROUPED CV, 5 FOLDS, 3 SEEDS")

OOF = {}
rows = []

for model_name, model in MODELS.items():
    for fs_name, features in FEATURE_SETS.items():
        pooled = []
        pr_scores = []
        macro_scores = []
        for seed in SEEDS:
            oof = oof_predict(
                b,
                features,
                model,
                seed
            )

            if seed == SEEDS[0]:
                OOF[(model_name, fs_name)] = oof
            pooled.append(
                safe_auc(y_b, oof)
            )
            pr_scores.append(
                average_precision_score(y_b, oof)
            )
            macro_scores.append(
                pp_auc(oof).mean()
            )

        rows.append({
            "model": model_name,
            "features": fs_name,
            "pooled_auc": np.mean(pooled),
            "pr_auc": np.mean(pr_scores),
            "macro_auc": np.mean(macro_scores),
            "macro_seed_std": np.std(macro_scores)
        })

        print("Done:", model_name, fs_name)


cv_table = pd.DataFrame(rows).round(3)

print("\nProteins in macro AUC:", len(ELIG))
print("PR-AUC baseline:", round(y_b.mean(), 3))
print(cv_table.to_string(index=False))


# ----------------------------------------------------------
# CHECK 2: Bootstrap by protein
# ----------------------------------------------------------

banner("CHECK 2: BOOTSTRAP BY PROTEIN")
rng = np.random.default_rng(0)

prot_idx = list(
    b.groupby("protein_seq").indices.values()
)

key = ("logistic", "scores")
oof_key = OOF[key]


# Pooled AUC bootstrap.
pooled_boot = []

for _ in range(300):

    pick = rng.integers(
        0,
        len(prot_idx),
        len(prot_idx)
    )

    idx = np.concatenate([
        prot_idx[i] for i in pick
    ])

    auc = safe_auc(
        y_b[idx],
        oof_key[idx]
    )

    if not np.isnan(auc):
        pooled_boot.append(auc)


# Macro AUC bootstrap.
per_key = pp_auc(oof_key)

macro_boot = per_key[
    rng.integers(
        0,
        len(per_key),
        (2000, len(per_key))
    )
].mean(axis=1)


print("Logistic, 4 scores")

print(
    "Pooled AUC:",
    round(safe_auc(y_b, oof_key), 3),
    "interval:",
    np.round(
        np.percentile(pooled_boot, [2.5, 97.5]),
        3
    )
)

print(
    "Macro AUC:",
    round(per_key.mean(), 3),
    "interval:",
    np.round(
        np.percentile(macro_boot, [2.5, 97.5]),
        3
    )
)


# Paired differences between models and feature sets.
print("\nPaired difference in macro AUC, seed 42")

pairs = [

    (("logistic", "scores+length"),
     ("logistic", "scores")),

    (("forest", "scores+length"),
     ("forest", "scores")),

    (("logistic", "scores+composition"),
     ("logistic", "scores")),

    (("forest", "scores+composition"),
     ("forest", "scores")),

    (("logistic", "scores+protein"),
     ("logistic", "scores")),

    (("forest", "scores+protein"),
     ("forest", "scores")),

    (("forest", "scores"),
     ("logistic", "scores"))
]


diff_rows = []

for a_key, b_key in pairs:

    diff = pp_auc(OOF[a_key]) - pp_auc(OOF[b_key])

    boot = diff[
        rng.integers(
            0,
            len(diff),
            (2000, len(diff))
        )
    ].mean(axis=1)

    low, high = np.percentile(
        boot,
        [2.5, 97.5]
    )

    diff_rows.append({
        "first": f"{a_key[0]} {a_key[1]}",
        "second": f"{b_key[0]} {b_key[1]}",
        "mean_diff": diff.mean(),
        "low": low,
        "high": high,
        "interval_excludes_zero": bool(
            low > 0 or high < 0
        )
    })


print(
    pd.DataFrame(diff_rows)
    .round(3)
    .to_string(index=False)
)


# ----------------------------------------------------------
# CHECK 3: Length-stratified AUC in B_Cells
# ----------------------------------------------------------

banner("CHECK 3: LENGTH-STRATIFIED AUC IN B_CELLS")
length_b = b["peptide_length"].to_numpy()
strata_b = {
    "8-10 aa":
        (length_b >= 8) & (length_b <= 10),
    "15 aa":
        length_b == 15,
    "5-7 and 11-14 aa":
        (
            ((length_b >= 5) & (length_b <= 7))
            |
            ((length_b >= 11) & (length_b <= 14))
        )
}


scorers = {
    "logistic scores (OOF)":
        OOF[("logistic", "scores")],
    "forest scores (OOF)":
        OOF[("forest", "scores")],
    "emini_log":
        b["emini_log"].to_numpy(),
    "chou_fasman":
        b["chou_fasman"].to_numpy(),
    "kolaskar_tongaonkar":
        b["kolaskar_tongaonkar"].to_numpy(),
    "parker":
        b["parker"].to_numpy()
}


rows = []
for stratum, mask in strata_b.items():
    for scorer_name, scores in scorers.items():
        pooled = safe_auc(
            y_b[mask],
            scores[mask]
        )
        macro, n_prot = macro_auc(
            y_b[mask],
            scores[mask],
            g_b[mask]
        )

        rows.append({
            "stratum": stratum,
            "rows": int(mask.sum()),
            "positives": int(y_b[mask].sum()),
            "scorer": scorer_name,
            "pooled_auc": pooled,
            "macro_auc": macro,
            "proteins": n_prot
        })


print(
    pd.DataFrame(rows)
    .round(3)
    .to_string(index=False)
)


# ----------------------------------------------------------
# CHECK 4: Train on B_Cells, test on SARS
# ----------------------------------------------------------

banner("CHECK 4: TRAIN ON B_CELLS, TEST ON SARS")

EXTERNAL_SETS = [
    "length",
    "scores",
    "scores+length",
    "scores+composition",
    "scores+composition+length"
]


subsets_s = {
    "5-15 aa":
        s["peptide_length"] <= 15,
    "5-20 aa":
        s["peptide_length"] <= 20,
    "all":
        s["peptide_length"] >= 0
}


print(
    "SARS subset sizes:",
    {
        name: int(mask.sum())
        for name, mask in subsets_s.items()
    }
)


fitted = {}
rows = []

for model_name, model in MODELS.items():
    for fs_name in EXTERNAL_SETS:
        features = FEATURE_SETS[fs_name]
        fitted_model = clone(model)
        fitted_model.fit(
            b[features].to_numpy(),
            y_b
        )

        fitted[(model_name, fs_name)] = fitted_model
        probabilities = fitted_model.predict_proba(
            s[features].to_numpy()
        )[:, 1]

        row = {
            "model": model_name,
            "features": fs_name
        }

        for sub_name, mask in subsets_s.items():

            row[f"AUC {sub_name}"] = safe_auc(
                y_s[mask],
                probabilities[mask]
            )

        rows.append(row)


print(
    pd.DataFrame(rows)
    .round(3)
    .to_string(index=False)
)


# ----------------------------------------------------------
# CHECK 5: SARS AUC by length group
# ----------------------------------------------------------

banner("CHECK 5: SARS AUC BY LENGTH STRATUM")
length_s = s["peptide_length"].to_numpy()
strata_s = {
    "5-14 aa":
        length_s <= 14,
    "15 aa":
        length_s == 15,
    "17 aa":
        length_s == 17,
    "16-25 aa":
        (length_s >= 16) & (length_s <= 25)
}


for stratum, mask in strata_s.items():
    print(
        stratum,
        "rows:",
        int(mask.sum()),
        "positives:",
        int(y_s[mask].sum())
    )


rows = []

proba_logit = fitted[
    ("logistic", "scores")
].predict_proba(
    s[SCORES_LOG].to_numpy()
)[:, 1]


proba_forest = fitted[
    ("forest", "scores")
].predict_proba(
    s[SCORES_LOG].to_numpy()
)[:, 1]


s_scorers = {
    "logistic scores": proba_logit,
    "forest scores": proba_forest,
    "emini_log": s["emini_log"].to_numpy(),
    "chou_fasman": s["chou_fasman"].to_numpy()
}


for stratum, mask in strata_s.items():
    for scorer_name, scores in s_scorers.items():
        positives = int(y_s[mask].sum())
        negatives = int(mask.sum() - positives)
        if min(positives, negatives) >= 5:

            auc = safe_auc(
                y_s[mask],
                scores[mask]
            )

        else:
            auc = np.nan

        rows.append({
            "stratum": stratum,
            "scorer": scorer_name,
            "AUC": auc
        })


print(
    "\nAUC is blank when either class has fewer than 5 peptides."
)

print(
    pd.DataFrame(rows)
    .round(3)
    .to_string(index=False)
)


# ----------------------------------------------------------
# CHECK 6: Bootstrap by peptide on SARS
# ----------------------------------------------------------

banner("CHECK 6: SARS BOOTSTRAP BY PEPTIDE")

rng = np.random.default_rng(1)

boot_sets = {
    "logistic scores": proba_logit,
    "forest scores": proba_forest,
    "emini_log alone": s["emini_log"].to_numpy()
}


for sub_name in ["5-15 aa", "all"]:
    mask = subsets_s[sub_name].to_numpy()
    idx_all = np.where(mask)[0]

    for scorer_name, scores in boot_sets.items():
        values = []
        for _ in range(1000):

            idx = rng.choice(
                idx_all,
                len(idx_all),
                replace=True
            )

            auc = safe_auc(
                y_s[idx],
                scores[idx]
            )

            if not np.isnan(auc):
                values.append(auc)

        original_auc = safe_auc(
            y_s[mask],
            scores[mask]
        )

        if values:

            interval = np.round(
                np.percentile(values, [2.5, 97.5]),
                3
            )

        else:
            interval = "Not available"

        print(
            f"SARS {sub_name}, {scorer_name}:",
            "AUC:", round(original_auc, 3),
            "interval:", interval
        )


# ----------------------------------------------------------
# CHECK 7: Variance inflation factors
# ----------------------------------------------------------

banner("CHECK 7: VARIANCE INFLATION FACTORS")

Z = StandardScaler().fit_transform(
    b[SCORES_LOG]
)

for i, name in enumerate(SCORES_LOG):

    others = np.delete(Z, i, axis=1)

    r2 = LinearRegression().fit(
        others,
        Z[:, i]
    ).score(
        others,
        Z[:, i]
    )

    if r2 >= 1:
        vif = np.inf
    else:
        vif = 1 / (1 - r2)

    print(f"{name}: VIF {vif:.2f}")


# ----------------------------------------------------------
# CHECK 8: Protein length versus positive rate
# ----------------------------------------------------------

banner("CHECK 8: PROTEIN LENGTH VERSUS POSITIVE RATE")

pp = b.groupby("protein_seq").agg(
    peptides=("target", "size"),
    positives=("target", "sum")
)


pp["protein_len"] = [
    len(sequence)
    for sequence in pp.index
]

pp["rate"] = (
    pp["positives"] / pp["peptides"]
)


print(
    "Spearman, protein length vs positive rate:",
    round(
        pp[["protein_len", "rate"]]
        .corr(method="spearman")
        .iloc[0, 1],
        3
    )
)


print(
    "Spearman, protein length vs peptides per protein:",
    round(
        pp[["protein_len", "peptides"]]
        .corr(method="spearman")
        .iloc[0, 1],
        3
    )
)


bins = pd.cut(
    pp["protein_len"],
    [0, 150, 300, 600, 1000, 10000]
)


summary = pp.groupby(
    bins,
    observed=True
).agg(

    proteins=("peptides", "size"),
    peptides=("peptides", "sum"),
    positives=("positives", "sum")
)


summary["peptide_weighted_rate"] = (
    summary["positives"] / summary["peptides"]
).round(3)


print(summary)


# ----------------------------------------------------------
# CHECK 9: Single amino acid fraction AUC
# ----------------------------------------------------------

banner("CHECK 9: SINGLE AMINO ACID FRACTION AUC")
s15 = s[s["peptide_length"] <= 15]
rows = []

for col in COMP:
    b_pooled = safe_auc(
        y_b,
        b[col].to_numpy()
    )

    b_macro, n_prot = macro_auc(
        y_b,
        b[col].to_numpy(),
        g_b
    )

    s_auc = safe_auc(
        s15["target"].to_numpy(),
        s15[col].to_numpy()
    )

    rows.append({
        "feature": col,
        "B_Cells_pooled": b_pooled,
        "B_Cells_macro": b_macro,
        "SARS_5_15": s_auc
    })


aa_table = pd.DataFrame(rows).set_index("feature")


aa_table["same_side_of_0.5"] = (
    (aa_table["B_Cells_macro"] - 0.5)
    * (aa_table["SARS_5_15"] - 0.5)

    > 0
)


print(
    aa_table
    .sort_values("B_Cells_macro", ascending=False)
    .round(3)
)


# ----------------------------------------------------------
# COMPLETED
# ----------------------------------------------------------

print("\n" + "=" * 70)
print("FILE 04 COMPLETED")
print("=" * 70)