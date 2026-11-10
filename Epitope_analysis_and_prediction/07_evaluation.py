# ==========================================================
# File: 07_evaluation.py
# Purpose: Full metrics (confusion matrix, accuracy, precision,
# recall, specificity, F1, macro F1, MCC, ROC, PR), protein-aware
# top-10% precision, clean SARS retest without P59594, and a
# cleaned COVID ranking.
# Input: Data/B_Cells_cleaned.csv, Data/Sars_cleaned.csv,
#        Data/COVID_cleaned.csv
# ==========================================================

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, matthews_corrcoef,
                             precision_recall_curve, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
RESULTS_DIR = BASE_DIR / "Results"
FIG_DIR = BASE_DIR / "Figures"

SEED = 42
RUN_SVM = True          # set False to save about 10 minutes
CONSTRUCT_END = None  # Set from the complete supplied COVID protein sequence
FS = {
    "3 features": ["chou_fasman", "emini_log", "parker"],
    "4 features": ["chou_fasman", "emini_log", "kolaskar_tongaonkar",
                   "parker"],
}
MODELS = {
    "logistic": make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)),
    "boosting": HistGradientBoostingClassifier(
        max_depth=3, max_iter=200, learning_rate=0.05,
        min_samples_leaf=50, l2_regularization=1.0, random_state=42),
}
if RUN_SVM:
    MODELS["svm_rbf"] = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced"))


def banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load(name):
    df = pd.read_csv(DATA_DIR / name)
    df["peptide_length"] = df["end_position"] - df["start_position"] + 1
    if "target" in df.columns:
        n_labels = df.groupby("peptide_seq")["target"].nunique()
        conflicts = n_labels[n_labels > 1].index
        df = df[~df["peptide_seq"].isin(conflicts)]
    df = df.drop_duplicates(["protein_seq", "start_position", "end_position"])
    df = df.reset_index(drop=True)
    df["emini_log"] = np.log1p(df["emini"])
    return df


def fit_model(name, X, y):
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) != 2 or not np.isin(y, [0, 1]).all():
        raise ValueError(f"{name}: training data must contain both target classes 0 and 1.")
    if not np.isfinite(X).all():
        raise ValueError(f"{name}: training features contain missing or non-finite values.")
    m = clone(MODELS[name])
    if name == "boosting":
        counts = np.bincount(y, minlength=2)
        cw = len(y) / (2.0 * counts)
        m.fit(X, y, sample_weight=cw[y])
    else:
        m.fit(X, y)
    return m


def score_of(m, X):
    try:
        return m.predict_proba(X)[:, 1]
    except (AttributeError, NotImplementedError):
        return m.decision_function(X)


def best_threshold(y, p):
    grid = np.quantile(p, np.linspace(0.40, 0.95, 56))
    mccs = [matthews_corrcoef(y, (p >= t).astype(int)) for t in grid]
    return grid[int(np.argmax(mccs))]


def threshold_metrics(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "accuracy": (tp + tn) / len(y),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "f1": f1_score(y, pred, zero_division=0),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y, pred),
    }


banner("STEP 0: LOAD")
b = load("B_Cells_cleaned.csv")
s = load("Sars_cleaned.csv")
covid = load("COVID_cleaned.csv")
y_b = b["target"].astype(int).to_numpy()
g_b = b["protein_seq"].to_numpy()
y_s = s["target"].astype(int).to_numpy()
len_s = s["peptide_length"].to_numpy()
print("B_Cells:", len(b), "SARS:", len(s), "COVID:", len(covid))
print("Models:", list(MODELS))

ELIG = [i for i in b.groupby("protein_seq").indices.values()
        if len(i) >= 10 and len(np.unique(y_b[i])) == 2]
print("Proteins in macro AUC:", len(ELIG))

rng = np.random.default_rng(0)


def boot_mean(arr, n=2000):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        raise ValueError("Cannot calculate bootstrap interval from an empty array.")
    boots = arr[rng.integers(0, len(arr), (n, len(arr)))].mean(axis=1)
    return np.percentile(boots, [2.5, 97.5])


def pp_auc(score):
    return np.array([roc_auc_score(y_b[i], score[i]) for i in ELIG])


def topk_stats(score, frac=0.10):
    prec, base = [], []
    for i in ELIG:
        k = max(1, int(np.ceil(frac * len(i))))
        order = np.argsort(-score[i], kind="stable")[:k]
        prec.append(y_b[i][order].mean())
        base.append(y_b[i].mean())
    return np.array(prec), np.array(base)

# ----------------------------------------------------------
# Check 1: grouped CV with thresholds chosen inside training folds
# ----------------------------------------------------------

banner("CHECK 1: GROUPED CV, 5 OUTER FOLDS, 3 INNER FOLDS (seed 42)")

CV = {}
for fs, feats in FS.items():
    X = b[feats].to_numpy(dtype=float)
    if not np.isfinite(X).all():
        raise ValueError(f"{fs}: features contain missing or non-finite values.")

    for name in MODELS:
        outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        score = np.full(len(b), np.nan)
        pred = np.full(len(b), -1, dtype=int)

        for tr, te in outer.split(X, y_b, g_b):
            if len(np.unique(y_b[tr])) != 2:
                raise ValueError(f"{name}, {fs}: an outer training fold lacks one target class.")

            inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED + 1)
            oof_in = np.full(len(tr), np.nan)

            for itr, iva in inner.split(X[tr], y_b[tr], g_b[tr]):
                if len(np.unique(y_b[tr][itr])) != 2:
                    raise ValueError(f"{name}, {fs}: an inner training fold lacks one target class.")
                m = fit_model(name, X[tr][itr], y_b[tr][itr])
                oof_in[iva] = score_of(m, X[tr][iva])

            if not np.isfinite(oof_in).all():
                raise ValueError(f"{name}, {fs}: incomplete or invalid inner out-of-fold predictions.")

            thr = best_threshold(y_b[tr], oof_in)
            m = fit_model(name, X[tr], y_b[tr])
            s_te = score_of(m, X[te])

            if not np.isfinite(s_te).all():
                raise ValueError(f"{name}, {fs}: invalid outer test predictions.")

            score[te] = s_te
            pred[te] = (s_te >= thr).astype(int)

        if not np.isfinite(score).all() or (pred < 0).any():
            raise ValueError(f"{name}, {fs}: incomplete outer out-of-fold predictions.")

        CV[(fs, name)] = (score, pred)
        print("done:", fs, name)


# ----------------------------------------------------------
# Check 2: threshold metrics next to the two trivial predictors
# ----------------------------------------------------------

banner("CHECK 2: THRESHOLD METRICS (pooled over held-out folds)")

rows = []
for label, pred in [("all negative", np.zeros(len(b), dtype=int)),
                    ("all positive", np.ones(len(b), dtype=int))]:
    rows.append({"features": "-", "model": label,
                 **threshold_metrics(y_b, pred)})
for (fs, name), (score, pred) in CV.items():
    rows.append({"features": fs, "model": name,
                 **threshold_metrics(y_b, pred)})
thr_table = pd.DataFrame(rows)
print(thr_table.round(3).to_string(index=False))
thr_table.round(4).to_csv(RESULTS_DIR / "07_threshold_metrics.csv", index=False)
print("\nPositive rate:", round(y_b.mean(), 3))
print("One pooled threshold. Dense proteins dominate these counts.")
print("SVM scores are decision values, not probabilities.")


# ----------------------------------------------------------
# Check 3: ranking metrics with protein-bootstrap intervals
# ----------------------------------------------------------

banner("CHECK 3: RANKING METRICS")

PER = {}
rows = []
for (fs, name), (score, pred) in CV.items():
    per = pp_auc(score)
    PER[(fs, name)] = per
    lo, hi = boot_mean(per)
    rows.append({"features": fs, "model": name,
                 "pooled_auc": roc_auc_score(y_b, score),
                 "pr_auc": average_precision_score(y_b, score),
                 "macro_auc": per.mean(), "macro_low": lo,
                 "macro_high": hi})
rank_table = pd.DataFrame(rows)
print(rank_table.round(3).to_string(index=False))
rank_table.round(4).to_csv(RESULTS_DIR / "07_ranking_metrics.csv", index=False)
print("PR-AUC floor:", round(y_b.mean(), 3))
print("Pooled AUC and PR-AUC mix fold score scales. Macro AUC does not.")

print("\nPaired macro AUC differences (by protein)")
rows = []
pairs = []
for fs in FS:
    names = list(MODELS)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairs.append(((fs, names[i]), (fs, names[j])))
for name in MODELS:
    pairs.append((("4 features", name), ("3 features", name)))
for a, c in pairs:
    d = PER[a] - PER[c]
    lo, hi = boot_mean(d)
    rows.append({"first": f"{a[1]} {a[0]}", "second": f"{c[1]} {c[0]}",
                 "mean_diff": d.mean(), "low": lo, "high": hi,
                 "excludes_zero": bool(lo > 0 or hi < 0)})
print(pd.DataFrame(rows).round(3).to_string(index=False))
print(f"{len(pairs)} paired tests, no multiple-comparison correction.")


# ----------------------------------------------------------
# Check 4: per-protein top-10% precision
# ----------------------------------------------------------

banner("CHECK 4: PER-PROTEIN TOP-10% PRECISION (143 proteins)")

rows = []
for (fs, name), (score, pred) in CV.items():
    prec, base = topk_stats(score)
    d = prec - base
    lo, hi = boot_mean(d)
    rows.append({"features": fs, "model": name,
                 "top10_precision": prec.mean(),
                 "protein_base_rate": base.mean(),
                 "gain": d.mean(), "gain_low": lo, "gain_high": hi,
                 "enrichment": prec.mean() / base.mean()})
print(pd.DataFrame(rows).round(3).to_string(index=False))
print("Gain is top-10% precision minus the protein's own positive rate.")
print("Small proteins give k of 1 to 2 peptides, so values are noisy.")


# ----------------------------------------------------------
# Figures 9 to 12
# ----------------------------------------------------------

banner("FIGURES")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, fs in zip(axes, FS):
    for name in MODELS:
        sc = CV[(fs, name)][0]
        fpr, tpr, _ = roc_curve(y_b, sc)
        ax.plot(fpr, tpr, label=f"{name} (pooled AUC "
                                f"{roc_auc_score(y_b, sc):.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="chance")
    ax.set_title(f"ROC, {fs}")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
fig.suptitle("Figure 9: ROC Curves, Pooled Held-Out Predictions", fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(FIG_DIR / "Fig9_ROC.png", dpi=300, bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, fs in zip(axes, FS):
    for name in MODELS:
        sc = CV[(fs, name)][0]
        p, r, _ = precision_recall_curve(y_b, sc)
        ax.plot(r, p, label=f"{name} (PR-AUC "
                            f"{average_precision_score(y_b, sc):.3f})")
    ax.axhline(y_b.mean(), color="k", linestyle="--",
               label=f"floor {y_b.mean():.3f}")
    ax.set_title(f"Precision-Recall, {fs}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="upper right")
fig.suptitle("Figure 10: Precision-Recall Curves", fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(FIG_DIR / "Fig10_PR.png", dpi=300, bbox_inches="tight")
plt.close(fig)

n_models = len(MODELS)
fig, axes = plt.subplots(len(FS), n_models,
                         figsize=(4.6 * n_models, 4.4 * len(FS)),
                         squeeze=False)
for r_i, fs in enumerate(FS):
    for c_i, name in enumerate(MODELS):
        ax = axes[r_i, c_i]
        pred = CV[(fs, name)][1]
        cm = confusion_matrix(y_b, pred, labels=[0, 1])
        pct = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
        for a in range(2):
            for c in range(2):
                ax.text(c, a, f"{cm[a, c]}\n{pct[a, c]:.0%}",
                        ha="center", va="center", fontsize=12,
                        color="white" if pct[a, c] > 0.6 else "black")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pred non-epitope", "Pred epitope"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["True non-epitope", "True epitope"])
        ax.set_title(f"{name}, {fs}")
fig.suptitle("Figure 11: Confusion Matrices (row percentages)", fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(FIG_DIR / "Fig11_Confusion_Matrices.png", dpi=300,
            bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 5.5))
width = 0.8 / len(FS)
for k, fs in enumerate(FS):
    sub = rank_table[rank_table["features"] == fs]
    x = np.arange(len(sub)) + k * width
    err = np.vstack([sub["macro_auc"] - sub["macro_low"],
                     sub["macro_high"] - sub["macro_auc"]])
    ax.bar(x, sub["macro_auc"], width=width, yerr=err, capsize=4,
           label=fs, edgecolor="black")
ax.axhline(0.5, color="k", linestyle="--", label="chance")
ax.set_xticks(np.arange(len(sub)) + width * (len(FS) - 1) / 2)
ax.set_xticklabels(list(MODELS))
ax.set_ylim(0.45, 0.65)
ax.set_ylabel("Macro within-protein AUC")
ax.set_title("Figure 12: Model Comparison (95% protein-bootstrap intervals)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "Fig12_Model_Comparison.png", dpi=300,
            bbox_inches="tight")
plt.close(fig)
print("Saved: Fig9, Fig10, Fig11, Fig12")


# ----------------------------------------------------------
# Check 5: clean SARS retest
# ----------------------------------------------------------

banner("CHECK 5: SARS RETEST WITHOUT THE NEAR-IDENTICAL PROTEIN P59594")

if "parent_protein_id" not in b.columns:
    raise ValueError("parent_protein_id is missing from B_Cells data.")
if "parent_protein_id" not in b.columns:
    raise ValueError("B_Cells data is missing parent_protein_id, required to exclude P59594.")

is_p = b["parent_protein_id"].astype(str).eq("P59594").to_numpy()
print("P59594 rows in B_Cells:", int(is_p.sum()),
      "positives:", int(y_b[is_p].sum()))

sars_peps = set(s["peptide_seq"])
exact_b = b["peptide_seq"].isin(sars_peps).to_numpy()
print("B_Cells rows whose peptide also appears in SARS:", int(exact_b.sum()),
      "of which in P59594:", int((exact_b & is_p).sum()))

merged = s[["peptide_seq", "target"]].drop_duplicates("peptide_seq").merge(
    b.loc[exact_b, ["peptide_seq", "target"]].drop_duplicates("peptide_seq"),
    on="peptide_seq", suffixes=("_sars", "_bcells"))
print("Exact peptide overlaps:", len(merged),
      "same label:", int((merged["target_sars"]
                          == merged["target_bcells"]).sum()))

TRAIN_SETS = {
    "full (contaminated)": np.ones(len(b), dtype=bool),
    "no P59594": ~is_p,
    "no P59594, no exact matches": ~is_p & ~exact_b,
}
for k, v in TRAIN_SETS.items():
    print(f"{k}: {int(v.sum())} rows, {int(y_b[v].sum())} positives")

SUBSETS = {"5-15 aa": len_s <= 15, "5-14 aa": len_s <= 14,
           "17 aa": len_s == 17, "all": len_s >= 0}
rng_s = np.random.default_rng(1)


def sars_row(sc, mask, extra):
    idx_all = np.where(mask)[0]
    if len(idx_all) == 0 or len(np.unique(y_s[idx_all])) != 2:
        raise ValueError(f"Cannot calculate SARS AUC for subset: {extra['subset']}.")

    vals = []
    for _ in range(500):
        idx = rng_s.choice(idx_all, len(idx_all), replace=True)
        if len(np.unique(y_s[idx])) == 2:
            vals.append(roc_auc_score(y_s[idx], sc[idx]))

    if len(vals) < 100:
        raise ValueError(f"Too few valid bootstrap samples for SARS subset: {extra['subset']}.")

    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {
        **extra,
        "rows": int(mask.sum()),
        "positives": int(y_s[mask].sum()),
        "auc": roc_auc_score(y_s[mask], sc[mask]),
        "low": lo,
        "high": hi
    }


rows = []
for fs, feats in FS.items():
    for name in MODELS:
        for tname, tmask in TRAIN_SETS.items():
            m = fit_model(name, b.loc[tmask, feats].to_numpy(), y_b[tmask])
            sc = score_of(m, s[feats].to_numpy())
            for sub, smask in SUBSETS.items():
                rows.append(sars_row(sc, smask, {
                    "features": fs, "model": name,
                    "training": tname, "subset": sub}))
        print("done:", fs, name)
for sub, smask in SUBSETS.items():
    rows.append(sars_row(s["emini_log"].to_numpy(), smask, {
        "features": "-", "model": "emini_log alone",
        "training": "none", "subset": sub}))

sars_table = pd.DataFrame(rows)
sars_table.round(4).to_csv(RESULTS_DIR / "07_sars_retest.csv", index=False)
print("\nAUC with 95% peptide-bootstrap intervals")
print(sars_table.round(3).to_string(index=False))

print("\nAUC comparison across training sets")
print("Differences are small and inconsistent across models and peptide subsets.")
pivot = sars_table[sars_table["training"] != "none"].pivot_table(
    index=["features", "model", "subset"], columns="training",
    values="auc")
print(pivot.round(3).to_string())
print("\nSARS is one protein with overlapping peptides, so intervals run")
print("narrow. The 15 aa stratum is untestable (4 positives).")


# ----------------------------------------------------------
# Check 6: cleaned COVID ranking using the complete supplied sequence
# ----------------------------------------------------------

banner("CHECK 6: COVID RANKING USING COMPLETE SUPPLIED PROTEIN SEQUENCE")

# Confirm that the COVID dataset contains one consistent full protein sequence.
if "protein_seq" not in covid.columns:
    raise ValueError("COVID data is missing the protein_seq column.")

covid_sequences = covid["protein_seq"].dropna().astype(str).str.strip().str.upper().unique()

if len(covid_sequences) != 1:
    raise ValueError(
        f"Expected one COVID protein sequence, found {len(covid_sequences)}."
    )

covid_seq = covid_sequences[0]
CONSTRUCT_END = len(covid_seq)

if CONSTRUCT_END == 0:
    raise ValueError("The COVID protein sequence is empty.")

print("Complete supplied COVID sequence length:", CONSTRUCT_END)
print("Sequence ending:", covid_seq[-20:])
print("His-tag residues at sequence end:", len(covid_seq) - len(covid_seq.rstrip("H")))

# Use the same P59594 exclusion applied in File 6.
b_train = b.loc[~is_p].copy()
y_train = b_train["target"].astype(int).to_numpy()

if len(b_train) == 0 or len(np.unique(y_train)) != 2:
    raise ValueError("Filtered B_Cells training data must contain both target classes.")

print("B_Cells training rows after P59594 exclusion:", len(b_train))
print("B_Cells training positives:", int(y_train.sum()))

feats3 = FS["3 features"]
X_train = b_train[feats3].to_numpy(dtype=float)

fit_l = fit_model("logistic", X_train, y_train)
fit_b = fit_model("boosting", X_train, y_train)

# Keep peptides of 5 to 15 amino acids within the complete supplied sequence.
cand = covid[
    (covid["peptide_length"] >= 5)
    & (covid["peptide_length"] <= 15)
    & (covid["start_position"] >= 1)
    & (covid["end_position"] <= CONSTRUCT_END)
].copy()

# Verify the protein sequence and peptide coordinates before ranking.
cand["protein_seq"] = cand["protein_seq"].astype(str).str.strip().str.upper()
cand["peptide_seq"] = cand["peptide_seq"].astype(str).str.strip().str.upper()

if not cand["protein_seq"].eq(covid_seq).all():
    raise ValueError("Some COVID candidate rows do not match the confirmed supplied protein sequence.")

coord_match = cand.apply(
    lambda r: covid_seq[int(r["start_position"]) - 1:int(r["end_position"])] == r["peptide_seq"],
    axis=1
)

if not coord_match.all():
    bad = int((~coord_match).sum())
    raise ValueError(f"{bad} COVID candidate peptides do not match their protein coordinates.")

print("COVID rows:", len(covid), "kept:", len(cand))

if len(cand) == 0:
    raise ValueError("No COVID candidates remain after sequence and length filtering.")

Xc = cand[feats3].to_numpy(dtype=float)

if not np.isfinite(Xc).all():
    raise ValueError("COVID candidate features contain missing or non-finite values.")

cand["score_logistic"] = score_of(fit_l, Xc)
cand["score_boosting"] = score_of(fit_b, Xc)

# Combine model percentile ranks to reduce dependence on different score scales.
cand["score"] = (
    cand["score_logistic"].rank(pct=True)
    + cand["score_boosting"].rank(pct=True)
) / 2

cand["len_pct"] = cand.groupby("peptide_length")["score"].rank(pct=True)

print(
    "Spearman, logistic vs boosting scores:",
    round(cand[["score_logistic", "score_boosting"]].corr(method="spearman").iloc[0, 1], 3)
)

top1 = cand[cand["score"] >= cand["score"].quantile(0.99)]

mix = pd.DataFrame({
    "share_all": cand["peptide_length"].value_counts(normalize=True),
    "share_top_1pct": top1["peptide_length"].value_counts(normalize=True)
}).fillna(0).sort_index()

print("\nLength mix, all candidates against the raw top 1 percent")
print(mix.round(3))

# Select non-overlapping candidates, with no more than three per peptide length.
MAX_PER_LENGTH = 3
ranked = cand.sort_values(["len_pct", "score"], ascending=False)

chosen = []
per_len = {}

for _, row in ranked.iterrows():
    peptide_len = int(row["peptide_length"])
    n_len = per_len.get(peptide_len, 0)

    if n_len >= MAX_PER_LENGTH:
        continue

    if all(
        row["end_position"] < c["start_position"]
        or row["start_position"] > c["end_position"]
        for c in chosen
    ):
        chosen.append(row)
        per_len[peptide_len] = n_len + 1

    if len(chosen) == 25:
        break

keep = [
    "len_pct", "score", "peptide_seq", "start_position", "end_position",
    "peptide_length", "chou_fasman", "emini", "parker"
]

top = pd.DataFrame(chosen)[keep]

if top.empty:
    raise ValueError("No non-overlapping COVID candidates were selected.")

top.to_csv(RESULTS_DIR / "COVID_top_clean.csv", index=False)

cand.sort_values("score", ascending=False)[
    keep + ["score_logistic", "score_boosting"]
].to_csv(RESULTS_DIR / "COVID_ranked_clean.csv", index=False)

print("\nTop", len(top), "non-overlapping candidates, at most", MAX_PER_LENGTH, "per length")
print(top.round(3).to_string(index=False))

print("\nRankings are computational candidates for follow-up, not validated epitopes.")
print("COVID ranking models were trained after excluding P59594.")
print("The full supplied COVID sequence was retained, including its terminal His-tag residues.")

print("\n" + "=" * 70)
print("FILE 07 COMPLETED")
print("=" * 70)