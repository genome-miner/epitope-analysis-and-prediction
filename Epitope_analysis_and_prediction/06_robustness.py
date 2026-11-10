# ==========================================================
# File: 06_robustness.py
# Purpose: Paired feature tests, label-shuffle null, train versus
# test gap, Dummy and SVM rows, paired SARS tests, similarity
# check, and a cleaned COVID ranking with partial dependence.
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
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 100)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
RESULTS_DIR = BASE_DIR / "Results"
FIG_DIR = BASE_DIR / "Figures"
FIG_DIR.mkdir(exist_ok=True)

FEATS = ["chou_fasman", "emini_log", "kolaskar_tongaonkar", "parker"]
SEEDS = [42, 7, 2024]
N_SHUFFLES = 300

MODELS = {
    "dummy": DummyClassifier(strategy="prior"),
    "logistic": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)),
    "svm_rbf": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")),
    "boosting": HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05, min_samples_leaf=50, l2_regularization=1.0, random_state=42)
}


def banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load(name):
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    df = pd.read_csv(path)
    required = ["peptide_seq", "protein_seq", "start_position", "end_position", "emini"] + FEATS[:1] + FEATS[2:]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
    df["start_position"] = pd.to_numeric(df["start_position"], errors="coerce")
    df["end_position"] = pd.to_numeric(df["end_position"], errors="coerce")
    df = df.dropna(subset=["peptide_seq", "protein_seq", "start_position", "end_position", "emini"])
    df["peptide_length"] = df["end_position"] - df["start_position"] + 1
    if "target" in df.columns:
        df["target"] = pd.to_numeric(df["target"], errors="coerce")
        df = df.dropna(subset=["target"])
        n_labels = df.groupby("peptide_seq")["target"].nunique()
        conflicts = n_labels[n_labels > 1].index
        df = df[~df["peptide_seq"].isin(conflicts)]
        df["target"] = df["target"].astype(int)
        if not set(df["target"].unique()).issubset({0, 1}):
            raise ValueError(f"{name} contains target values other than 0 and 1.")
    df = df.drop_duplicates(["protein_seq", "start_position", "end_position"])
    df = df.reset_index(drop=True)
    df["emini_log"] = np.log1p(df["emini"])
    feature_cols = ["chou_fasman", "emini_log", "kolaskar_tongaonkar", "parker"]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=feature_cols + ["peptide_length"])
    df = df.reset_index(drop=True)
    if df.empty:
        raise ValueError(f"{name} has no usable rows after cleaning.")
    return df


def macro_auc(y, score, groups, min_n=10):
    frame = pd.DataFrame({"y": y, "score": score, "group": groups})
    vals = []
    for _, d in frame.groupby("group"):
        if len(d) >= min_n and d["y"].nunique() == 2:
            vals.append(roc_auc_score(d["y"], d["score"]))
    return float(np.mean(vals)) if vals else np.nan


def fit_model(name, X, y):
    m = clone(MODELS[name])
    if name == "boosting":
        counts = np.bincount(y, minlength=2)
        if np.any(counts == 0):
            raise ValueError("Boosting training data must contain both target classes.")
        weights = len(y) / (2.0 * counts)
        m.fit(X, y, sample_weight=weights[y])
    else:
        m.fit(X, y)
    return m


def score_of(m, X):
    if hasattr(m, "predict_proba"):
        return m.predict_proba(X)[:, 1]
    return m.decision_function(X)


def oof_predict(feats, name, seed, y_override=None, track=False):
    X = b[feats].to_numpy()
    y_work = y_b.copy() if y_override is None else np.asarray(y_override, dtype=int)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.full(len(y_work), np.nan)
    train_auc, test_auc = [], []
    for tr, te in cv.split(X, y_work, g_b):
        if len(np.unique(y_work[tr])) < 2:
            raise ValueError("A training fold contains only one target class.")
        m = fit_model(name, X[tr], y_work[tr])
        oof[te] = score_of(m, X[te])
        if track:
            train_auc.append(macro_auc(y_work[tr], score_of(m, X[tr]), g_b[tr]))
            test_auc.append(macro_auc(y_work[te], oof[te], g_b[te]))
    if np.isnan(oof).any():
        raise ValueError("Some rows did not receive out-of-fold predictions.")
    return oof, train_auc, test_auc


banner("STEP 0: LOAD")
b = load("B_Cells_cleaned.csv")
s = load("Sars_cleaned.csv")
covid = load("COVID_cleaned.csv")

if "target" not in b.columns or "target" not in s.columns:
    raise ValueError("B_Cells and SARS files must contain a target column.")

if b["target"].nunique() != 2 or s["target"].nunique() != 2:
    raise ValueError("B_Cells and SARS data must each contain both target classes.")

X_b = b[FEATS].to_numpy()
y_b = b["target"].astype(int).to_numpy()
g_b = b["protein_seq"].astype(str).to_numpy()
X_s = s[FEATS].to_numpy()
y_s = s["target"].astype(int).to_numpy()
len_s = s["peptide_length"].to_numpy()

if b["protein_seq"].nunique() < 5:
    raise ValueError("Not enough B_Cells protein groups for 5-fold grouped CV.")

covid_seq = str(covid["protein_seq"].iloc[0])
if covid["protein_seq"].nunique() != 1:
    raise ValueError("COVID data should contain one parent protein sequence.")

NATIVE_LEN = len(covid_seq)

banner("SEQUENCE CHECK: P59594 VS SARS")

if "parent_protein_id" not in b.columns:
    raise ValueError("parent_protein_id is missing from B_Cells data.")

p59594 = b.loc[b["parent_protein_id"].astype(str) == "P59594", "protein_seq"].drop_duplicates()

if p59594.empty:
    print("P59594 was not found in B_Cells.")
else:
    pseq = str(p59594.iloc[0])
    sars_seq = str(s["protein_seq"].iloc[0])
    print("P59594 length:", len(pseq))
    print("SARS length:", len(sars_seq))
    print("Exact sequence match:", pseq == sars_seq)
    if len(pseq) == len(sars_seq):
        print("Same-position identity:", round(sum(a == c for a, c in zip(pseq, sars_seq)) / len(pseq), 4))

print("B_Cells:", len(b), "SARS:", len(s), "COVID:", len(covid))
print("COVID native protein length:", NATIVE_LEN)

ELIG = [i for i in b.groupby("protein_seq").indices.values() if len(i) >= 10 and len(np.unique(y_b[i])) == 2]
print("Proteins in macro AUC:", len(ELIG))

if not ELIG:
    raise ValueError("No eligible proteins are available for macro AUC.")


def pp_auc(score, y_values=y_b):
    return np.array([roc_auc_score(y_values[i], score[i]) for i in ELIG])


rng = np.random.default_rng(0)


def boot_mean(arr, n=2000, pct=(2.5, 97.5)):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan
    boots = arr[rng.integers(0, len(arr), (n, len(arr)))].mean(axis=1)
    return np.percentile(boots, pct)


# ----------------------------------------------------------
# Check 1: paired feature tests, 3 seeds, same model class
# ----------------------------------------------------------

banner("CHECK 1: FEATURE SET TESTS (3 seeds, paired by protein)")

FS = {
    "all4": FEATS,
    "emini only": ["emini_log"],
    "emini+chou_fasman": ["emini_log", "chou_fasman"],
    "no kolaskar": [f for f in FEATS if f != "kolaskar_tongaonkar"],
    "no parker": [f for f in FEATS if f != "parker"]
}

PP = {}
OOF42 = {}

for name in ["logistic", "boosting"]:
    for fs, feats in FS.items():
        arrs = []
        for seed in SEEDS:
            oof, _, _ = oof_predict(feats, name, seed)
            if seed == 42 and fs == "all4":
                OOF42[name] = oof
            arrs.append(pp_auc(oof))
        PP[(name, fs)] = np.mean(arrs, axis=0)
        print("done:", name, fs)

rows = []
n_tests = 8
corr_pct = (100 * 0.05 / n_tests / 2, 100 - 100 * 0.05 / n_tests / 2)

for name in ["logistic", "boosting"]:
    for fs in FS:
        per = PP[(name, fs)]
        lo, hi = boot_mean(per)
        row = {"model": name, "features": fs, "macro_auc": per.mean(), "low": lo, "high": hi}
        if fs != "all4":
            diff = per - PP[(name, "all4")]
            dlo, dhi = boot_mean(diff)
            clo, chi = boot_mean(diff, pct=corr_pct)
            row.update({"diff_vs_all4": diff.mean(), "diff_low": dlo, "diff_high": dhi, "excludes_zero": bool(dlo > 0 or dhi < 0), "excludes_zero_corrected": bool(clo > 0 or chi < 0)})
        rows.append(row)

print(pd.DataFrame(rows).round(3).to_string(index=False))
print("Eight paired tests. The corrected column splits 0.05 across eight.")


# ----------------------------------------------------------
# Check 2: within-protein label shuffle null
# ----------------------------------------------------------

banner("CHECK 2: WITHIN-PROTEIN LABEL SHUFFLE (300 shuffles)")

rng_n = np.random.default_rng(5)

for name in ["logistic", "boosting"]:
    observed_oof = OOF42[name]
    observed = pp_auc(observed_oof).mean()
    null = []
    for shuffle_num in range(N_SHUFFLES):
        y_perm = y_b.copy()
        for indices in b.groupby("protein_seq").indices.values():
            y_perm[indices] = rng_n.permutation(y_perm[indices])
        perm_oof, _, _ = oof_predict(FEATS, name, 42, y_override=y_perm)
        perm_auc = pp_auc(perm_oof, y_perm).mean()
        null.append(perm_auc)
        if (shuffle_num + 1) % 50 == 0:
            print(f"{name}: completed {shuffle_num + 1}/{N_SHUFFLES} shuffles")
    null = np.asarray(null)
    p = (1 + int((null >= observed).sum())) / (1 + len(null))
    print(f"{name}: observed {observed:.3f}, null mean {null.mean():.3f}, null 95th pct {np.percentile(null, 95):.3f}, p {p:.4f}")


# ----------------------------------------------------------
# Check 3: train versus test macro AUC
# ----------------------------------------------------------

banner("CHECK 3: TRAIN VERSUS TEST MACRO AUC (seed 42)")

rows = []

for name in ["logistic", "boosting"]:
    _, tr_auc, te_auc = oof_predict(FEATS, name, 42, track=True)
    rows.append({"model": name, "train_macro_mean": np.nanmean(tr_auc), "test_macro_mean": np.nanmean(te_auc), "gap": np.nanmean(tr_auc) - np.nanmean(te_auc), "test_fold_std": np.nanstd(te_auc)})

print(pd.DataFrame(rows).round(3).to_string(index=False))
print("Train proteins are scored in-sample. A gap above 0.05 is used as a project-level warning threshold.")


# ----------------------------------------------------------
# Check 4: Dummy and SVM rows
# ----------------------------------------------------------

banner("CHECK 4: DUMMY AND SVM ROWS (seed 42, four scores)")

rate = y_b.mean()
rows = []

for name in ["dummy", "logistic", "svm_rbf", "boosting"]:
    if name == "dummy":
        oof = np.full(len(y_b), rate)
    elif name in OOF42:
        oof = OOF42[name]
    else:
        oof, _, _ = oof_predict(FEATS, name, 42)

    rows.append({
        "model": name,
        "pooled_auc": roc_auc_score(y_b, oof),
        "pr_auc": average_precision_score(y_b, oof),
        "macro_auc": pp_auc(oof).mean()
    })
    print("done:", name)

print(pd.DataFrame(rows).round(3).to_string(index=False))
print("PR-AUC floor:", round(rate, 3))
print("F1 of an all-positive predictor:", round(2 * rate / (1 + rate), 3))


# ----------------------------------------------------------
# Check 5: paired SARS tests, excluding highly similar protein
# ----------------------------------------------------------

banner("CHECK 5: PAIRED SARS TESTS (excluding P59594)")

if "parent_protein_id" not in b.columns:
    raise ValueError("parent_protein_id is required for the SARS similarity exclusion.")

b_train = b[b["parent_protein_id"].astype(str) != "P59594"].copy()

if b_train.empty or b_train["target"].nunique() != 2:
    raise ValueError("Training data after P59594 exclusion must contain both target classes.")

X_train = b_train[FEATS].to_numpy()
y_train = b_train["target"].astype(int).to_numpy()

print("Original B_Cells rows:", len(b))
print("Rows after P59594 exclusion:", len(b_train))
print("Excluded rows:", len(b) - len(b_train))
print("Remaining protein groups:", b_train["protein_seq"].nunique())

fit_log = fit_model("logistic", X_train, y_train)
fit_boost = fit_model("boosting", X_train, y_train)

sc = {
    "boosting": score_of(fit_boost, X_s),
    "logistic": score_of(fit_log, X_s),
    "emini_log": s["emini_log"].to_numpy()
}

pairs = [("boosting", "emini_log"), ("logistic", "emini_log"), ("boosting", "logistic")]
subsets = {"5-15 aa": len_s <= 15, "5-20 aa": len_s <= 20, "all": np.ones_like(len_s, dtype=bool)}

rng_s = np.random.default_rng(1)
rows = []

for sub_name, mask in subsets.items():
    idx_all = np.where(mask)[0]
    if len(idx_all) == 0 or len(np.unique(y_s[idx_all])) < 2:
        print("Skipping subset without both target classes:", sub_name)
        continue
    diffs = {p: [] for p in pairs}
    for _ in range(1000):
        idx = rng_s.choice(idx_all, len(idx_all), replace=True)
        if len(np.unique(y_s[idx])) < 2:
            continue
        auc = {k: roc_auc_score(y_s[idx], v[idx]) for k, v in sc.items()}
        for a, c in pairs:
            diffs[(a, c)].append(auc[a] - auc[c])
    for a, c in pairs:
        d = np.asarray(diffs[(a, c)])
        if len(d) == 0:
            continue
        lo, hi = np.percentile(d, [2.5, 97.5])
        obs = roc_auc_score(y_s[mask], sc[a][mask]) - roc_auc_score(y_s[mask], sc[c][mask])
        rows.append({"subset": sub_name, "first": a, "second": c, "auc_diff": obs, "low": lo, "high": hi, "excludes_zero": bool(lo > 0 or hi < 0)})

print(pd.DataFrame(rows).round(3).to_string(index=False))
print("Subsets nest, so the rows are not independent tests.")


# ----------------------------------------------------------
# Check 6: sequence similarity by shared 5-mers
# ----------------------------------------------------------

banner("CHECK 6: SHARED 5-MERS BETWEEN PROTEINS")


def kmers(seq, k=5):
    seq = str(seq)
    return {seq[i:i + k] for i in range(len(seq) - k + 1)} if len(seq) >= k else set()


if "parent_protein_id" in b.columns:
    id_of = b.drop_duplicates("protein_seq").set_index("protein_seq")["parent_protein_id"]
else:
    id_of = pd.Series({seq: f"Protein_{i + 1}" for i, seq in enumerate(b["protein_seq"].drop_duplicates())})

b_seqs = list(id_of.index)
sars_seq = str(s["protein_seq"].iloc[0])

print("SARS protein length:", len(sars_seq), "COVID protein length:", len(covid_seq))

for label, seq in [("SARS", sars_seq), ("COVID", covid_seq)]:
    km = kmers(seq)
    if not km:
        print(f"{label}: sequence is too short for 5-mer comparison.")
        continue
    fr = pd.Series({id_of[p]: len(km & kmers(p)) / len(km) for p in b_seqs})
    print(f"\n{label}: fraction of its 5-mers found in each B_Cells protein")
    print("Top 5:")
    print(fr.sort_values(ascending=False).head(5).round(3).to_string())
    print("B_Cells proteins above 0.05:", int((fr > 0.05).sum()), "above 0.20:", int((fr > 0.20).sum()))

sars_kmers = kmers(sars_seq)
covid_kmers = kmers(covid_seq)

if covid_kmers:
    shared = len(sars_kmers & covid_kmers) / len(covid_kmers)
    print("\nFraction of COVID 5-mers found in the SARS protein:", round(shared, 3))
else:
    print("\nCOVID sequence is too short for 5-mer comparison.")

print("A high fraction against a B_Cells protein means the external test")
print("is not independent. Random sequences share almost no 5-mers.")


# ----------------------------------------------------------
# Check 7: cleaned COVID ranking and partial dependence
# ----------------------------------------------------------

banner("CHECK 7: COVID RANKING, CLEANED")

print("COVID construct sequence length:", len(covid_seq))
print("COVID ranking boundary:", NATIVE_LEN)
print("Last 30 residues:", covid_seq[-30:])

if NATIVE_LEN <= 0 or NATIVE_LEN > len(covid_seq):
    raise ValueError("Invalid COVID sequence boundary.")

if not np.isfinite(covid["start_position"]).all() or not np.isfinite(covid["end_position"]).all():
    raise ValueError("COVID positions contain missing or invalid values.")

if not np.allclose(covid["peptide_length"], covid["end_position"] - covid["start_position"] + 1):
    raise ValueError("COVID peptide lengths do not match their coordinates.")

print("Peptides ending beyond ranking boundary:", int((covid["end_position"] > NATIVE_LEN).sum()))

cand = covid[(covid["peptide_length"] >= 5) & (covid["peptide_length"] <= 15)].copy()
cand = cand[(cand["start_position"] >= 1) & (cand["end_position"] <= NATIVE_LEN)].copy()

print("Candidates kept (5-15 aa, inside ranking boundary):", len(cand))

if cand.empty:
    raise ValueError("No COVID candidates remain after filtering.")

if not np.isfinite(cand[FEATS].to_numpy()).all():
    raise ValueError("COVID candidate features contain missing or infinite values.")

cand["score"] = score_of(fit_boost, cand[FEATS].to_numpy())
cand["len_pct"] = cand.groupby("peptide_length")["score"].rank(pct=True)

spearman = cand[["score", "peptide_length"]].corr(method="spearman").iloc[0, 1]
print("Spearman, score vs peptide length:", round(spearman, 3))

top1 = cand[cand["score"] >= cand["score"].quantile(0.99)]
share = pd.DataFrame({
    "share_all": cand["peptide_length"].value_counts(normalize=True),
    "share_top_1pct": top1["peptide_length"].value_counts(normalize=True)
}).fillna(0).sort_index()

print("\nLength mix, all candidates against the raw top 1 percent")
print(share.round(3))

ranked = cand.sort_values(["len_pct", "score"], ascending=False)
chosen = []
for _, row in ranked.iterrows():
    if all(row["end_position"] < c["start_position"] or row["start_position"] > c["end_position"] for c in chosen):
        chosen.append(row)
    if len(chosen) == 25:
        break

keep = ["len_pct", "score", "peptide_seq", "start_position", "end_position", "peptide_length", "chou_fasman", "emini", "kolaskar_tongaonkar", "parker"]
top = pd.DataFrame(chosen, columns=cand.columns)[keep]

if top.empty:
    raise ValueError("No non-overlapping COVID candidates were selected.")

top.to_csv(RESULTS_DIR / "COVID_top_by_length_percentile.csv", index=False)

print("\nTop non-overlapping candidates, ranked by percentile within length")
print(top.round(3).to_string(index=False))
print("\nThese are computational rankings, not experimentally validated epitopes.")

disp = PartialDependenceDisplay.from_estimator(fit_boost, X_train, features=[0, 1, 2, 3], feature_names=FEATS, kind="average", grid_resolution=30, method="brute")
disp.figure_.suptitle("Figure 7: Partial Dependence, Boosting on Filtered B_Cells")
disp.figure_.savefig(FIG_DIR / "Fig7_Partial_Dependence.png", dpi=300, bbox_inches="tight")
plt.close(disp.figure_)

fig, ax = plt.subplots(figsize=(9, 6))
disp2 = PartialDependenceDisplay.from_estimator(fit_boost, X_train, features=[(2, 3)], feature_names=FEATS, kind="average", grid_resolution=20, method="brute", ax=ax)
disp2.figure_.suptitle("Figure 8: Kolaskar and Parker, Joint Dependence", fontsize=15)
ax.set_xlabel("Kolaskar-Tongaonkar", fontsize=12)
ax.set_ylabel("Parker", fontsize=12)
ax.tick_params(axis="both", labelsize=10)
fig.tight_layout()
fig.savefig(FIG_DIR / "Fig8_Partial_Dependence_2D.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("Saved: Fig7_Partial_Dependence.png, Fig8_Partial_Dependence_2D.png")
print("\n" + "=" * 70)
print("FILE 06 COMPLETED")
print("=" * 70)