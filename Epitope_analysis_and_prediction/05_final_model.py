# ==========================================================
# File: 05_final_model.py
# Purpose: Nested grouped CV, one frozen model, SARS test,
# within-protein importance, ranked COVID candidates.
# Input: Data/B_Cells_cleaned.csv, Data/Sars_cleaned.csv,
#        Data/COVID_cleaned.csv
# Features: the four scores only, with log1p(emini).
# ==========================================================

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score,
                             matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
RESULTS_DIR = BASE_DIR / "Results"

FEATS = ["chou_fasman", "emini_log", "kolaskar_tongaonkar", "parker"]
SEED = 42

GRIDS = {
    "logistic": {"C": [0.01, 0.1, 1.0]},
    "forest": {
        "max_depth": [4, 6, 8],
        "min_samples_leaf": [20, 50],
    },
    "boosting": {
        "max_depth": [2, 3],
        "max_iter": [100, 200],
        "learning_rate": [0.05],
        "min_samples_leaf": [50],
        "l2_regularization": [1.0],
    },
}
MODES = ["none", "sqrt"]


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
        "emini",
        *FEATS[:1],
        "kolaskar_tongaonkar",
        "parker"
    ]

    if name != "COVID_cleaned.csv":
        required.append("target")
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {name}: {missing}")
    if df[required].isnull().any().any():
        raise ValueError(f"Missing values found in {name}")
    df["start_position"] = pd.to_numeric(
        df["start_position"], errors="coerce"
    )
    df["end_position"] = pd.to_numeric(
        df["end_position"], errors="coerce"
    )
    df["emini"] = pd.to_numeric(
        df["emini"], errors="coerce"
    )
    if df[["start_position", "end_position", "emini"]].isnull().any().any():
        raise ValueError(f"Invalid numeric values in {name}")
    df["peptide_length"] = (
        df["end_position"] - df["start_position"] + 1
    )
    if (df["peptide_length"] <= 0).any():
        raise ValueError(f"Invalid peptide lengths in {name}")
    if (df["emini"] < 0).any():
        raise ValueError(f"Negative Emini values in {name}")
    if "target" in df.columns:
        df["target"] = pd.to_numeric(
            df["target"], errors="coerce"
        )
        if df["target"].isnull().any():
            raise ValueError(f"Invalid target values in {name}")

        if not set(df["target"].unique()).issubset({0, 1}):
            raise ValueError(f"Target must contain only 0 and 1 in {name}")

        label_counts = df.groupby("peptide_seq")["target"].nunique()
        conflicts = label_counts[label_counts > 1].index

        df = df[~df["peptide_seq"].isin(conflicts)].copy()

    df = df.drop_duplicates(
        ["protein_seq", "start_position", "end_position"]
    )
    df = df.reset_index(drop=True)
    df["emini_log"] = np.log1p(df["emini"])
    return df


def per_protein_auc(y, s, g, min_n=10):
    frame = pd.DataFrame({"y": y, "s": s, "g": g})
    out = []
    for _, d in frame.groupby("g"):
        if len(d) >= min_n and d["y"].nunique() == 2:
            out.append(roc_auc_score(d["y"], d["s"]))
    return np.array(out)


def macro_auc(y, s, g, min_n=10):
    vals = per_protein_auc(y, s, g, min_n)
    return float(vals.mean()) if len(vals) else 0.5


def safe_auc(y, s):
    return np.nan if len(np.unique(y)) < 2 else roc_auc_score(y, s)


# ----------------------------------------------------------
# Weights: protein weight times class weight, mean 1
# ----------------------------------------------------------

def make_weights(g, y, mode):
    if mode == "none":
        w = np.ones(len(g), dtype=float)
    else:
        sizes = pd.Series(g).value_counts()
        size = pd.Series(g).map(sizes).to_numpy(float)
        w = 1.0 / np.sqrt(size)
    classes, counts = np.unique(y, return_counts=True)
    class_w = np.ones(int(np.max(y)) + 1)
    for cls, count in zip(classes, counts):
        class_w[cls] = len(y) / (len(classes) * count)
    w = w * class_w[y]
    return w / w.mean()


def fit_model(kind, params, X, y, w):
    if kind == "logistic":
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, **params))
        m.fit(X, y, logisticregression__sample_weight=w)
    elif kind == "forest":
        m = RandomForestClassifier(n_estimators=150, n_jobs=-1,
                                   random_state=SEED, **params)
        m.fit(X, y, sample_weight=w)
    else:
        m = HistGradientBoostingClassifier(random_state=SEED, **params)
        m.fit(X, y, sample_weight=w)
    return m


def search(kind, mode, X, y, g, n_splits, seed):
    """Grid search scored by macro within-protein AUC."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                              random_state=seed)
    splits = list(cv.split(X, y, g))
    best = None
    for params in ParameterGrid(GRIDS[kind]):
        oof = np.zeros(len(y))
        for itr, iva in splits:
            w = make_weights(g[itr], y[itr], mode)
            m = fit_model(kind, params, X[itr], y[itr], w)
            oof[iva] = m.predict_proba(X[iva])[:, 1]
        score = macro_auc(y, oof, g)
        if best is None or score > best[0]:
            best = (score, params, oof)
    return best


def best_threshold(y, p):
    grid = np.quantile(p, np.linspace(0.40, 0.95, 56))
    mccs = [matthews_corrcoef(y, (p >= t).astype(int)) for t in grid]
    return grid[int(np.argmax(mccs))]


# ----------------------------------------------------------
# Load
# ----------------------------------------------------------

banner("STEP 0: LOAD")
b = load("B_Cells_cleaned.csv")
s = load("Sars_cleaned.csv")
covid = load("COVID_cleaned.csv")

X_b = b[FEATS].to_numpy()
y_b = b["target"].astype(int).to_numpy()
g_b = b["protein_seq"].to_numpy()
X_s = s[FEATS].to_numpy()
y_s = s["target"].astype(int).to_numpy()
len_s = s["peptide_length"].to_numpy()

print("B_Cells:", len(b), "SARS:", len(s), "COVID:", len(covid))

ELIG = [i for i in b.groupby("protein_seq").indices.values()
        if len(i) >= 10 and len(np.unique(y_b[i])) == 2]
print("Proteins in macro AUC:", len(ELIG))


def pp_auc(score):
    return np.array([roc_auc_score(y_b[i], score[i]) for i in ELIG])


rng = np.random.default_rng(0)


def boot_mean(arr, n=2000):
    arr = np.asarray(arr)
    if len(arr) == 0:
        return np.nan, np.nan
    boots = arr[
        rng.integers(0, len(arr), (n, len(arr)))
    ].mean(axis=1)
    return np.percentile(boots, [2.5, 97.5])


# ----------------------------------------------------------
# Check 1: nested grouped CV, 5 outer folds, 3 inner folds
# ----------------------------------------------------------

banner("CHECK 1: NESTED GROUPED CV (seed 42, one split)")

RESULTS = {}
rows = []
for kind in GRIDS:
    for mode in MODES:
        outer = StratifiedGroupKFold(n_splits=5, shuffle=True,
                                     random_state=SEED)
        oof = np.zeros(len(y_b))
        pred = np.zeros(len(y_b), dtype=int)
        folds = []
        for tr, te in outer.split(X_b, y_b, g_b):
            _, params, oof_in = search(kind, mode, X_b[tr], y_b[tr],
                                       g_b[tr], 3, SEED + 1)
            thr = best_threshold(y_b[tr], oof_in)
            w = make_weights(g_b[tr], y_b[tr], mode)
            m = fit_model(kind, params, X_b[tr], y_b[tr], w)
            p = m.predict_proba(X_b[te])[:, 1]
            oof[te] = p
            pred[te] = (p >= thr).astype(int)
            folds.append((tr, te, params))
        per = pp_auc(oof)
        lo, hi = boot_mean(per)
        RESULTS[(kind, mode)] = {"oof": oof, "folds": folds, "per": per}
        rows.append({
            "model": kind, "weights": mode,
            "macro_auc": per.mean(), "macro_low": lo, "macro_high": hi,
            "pooled_auc": roc_auc_score(y_b, oof),
            "pr_auc": average_precision_score(y_b, oof),
            "mcc": matthews_corrcoef(y_b, pred),
            "f1": f1_score(y_b, pred),
        })
        print("done:", kind, mode)

table = pd.DataFrame(rows).round(3)
print("\nPR-AUC floor:", round(y_b.mean(), 3))
print(table.to_string(index=False))
table.to_csv(RESULTS_DIR / "05_model_comparison.csv", index=False)

print("\nMCC and F1 use one pooled threshold per fold, chosen inside")
print("the training folds. Dense proteins dominate these two numbers.")


# ----------------------------------------------------------
# Check 2: paired differences, protein weights versus none
# ----------------------------------------------------------

banner("CHECK 2: PAIRED MACRO DIFFERENCE, sqrt WEIGHTS MINUS NONE")

diff_rows = []
for kind in GRIDS:
    diff = RESULTS[(kind, "sqrt")]["per"] - RESULTS[(kind, "none")]["per"]
    lo, hi = boot_mean(diff)
    diff_rows.append({"model": kind, "mean_diff": diff.mean(),
                      "low": lo, "high": hi,
                      "excludes_zero": bool(lo > 0 or hi < 0)})
print(pd.DataFrame(diff_rows).round(3).to_string(index=False))
print("Three comparisons, no multiple-comparison correction.")


# ----------------------------------------------------------
# Choose the final configuration by the stated rule:
# highest outer macro AUC
# ----------------------------------------------------------

best_row = table.sort_values("macro_auc", ascending=False).iloc[0]
KIND, MODE = best_row["model"], best_row["weights"]
banner(f"FINAL CONFIGURATION: {KIND}, weights {MODE}")
print("Chosen by highest outer macro AUC. Gaps to the next models are")
print("often inside the intervals. Report the tie, not a winner.")

score, FINAL_PARAMS, _ = search(KIND, MODE, X_b, y_b, g_b, 5, SEED + 2)
print("Selected parameters:", FINAL_PARAMS)
final = fit_model(KIND, FINAL_PARAMS, X_b, y_b, make_weights(g_b, y_b, MODE))

baseline = fit_model("logistic", {"C": 1.0}, X_b, y_b,
                     make_weights(g_b, y_b, "none"))


# ----------------------------------------------------------
# Check 3: frozen model on SARS, with intervals
# ----------------------------------------------------------

banner("CHECK 3: FROZEN MODEL ON SARS (bootstrap by peptide)")

subsets = {
    "5-15 aa": len_s <= 15,
    "5-14 aa": len_s <= 14,
    "17 aa": len_s == 17,
    "5-20 aa": len_s <= 20,
    "all": len_s >= 0,
}
scorers = {
    "final model": final.predict_proba(X_s)[:, 1],
    "baseline logistic": baseline.predict_proba(X_s)[:, 1],
    "emini_log alone": s["emini_log"].to_numpy(),
}

rng_s = np.random.default_rng(1)
rows = []
for sub_name, mask in subsets.items():
    idx_all = np.where(mask)[0]
    pos = int(y_s[mask].sum())
    for name, sc in scorers.items():
        vals = []
        for _ in range(1000):
            idx = rng_s.choice(idx_all, len(idx_all), replace=True)
            if len(np.unique(y_s[idx])) == 2:
                vals.append(roc_auc_score(y_s[idx], sc[idx]))
        lo, hi = np.percentile(vals, [2.5, 97.5])
        rows.append({"subset": sub_name, "rows": int(mask.sum()),
                     "positives": pos, "scorer": name,
                     "auc": safe_auc(y_s[mask], sc[mask]),
                     "low": lo, "high": hi})
print(pd.DataFrame(rows).round(3).to_string(index=False))
print("\nThe 15 aa stratum is untestable (4 positives). Peptides overlap")
print("inside one protein, so these intervals run narrow.")


# ----------------------------------------------------------
# Check 4: within-protein permutation importance
# ----------------------------------------------------------

banner("CHECK 4: WITHIN-PROTEIN PERMUTATION IMPORTANCE")

rng_p = np.random.default_rng(2)
drops = np.zeros((len(FEATS), 0))
fold_drops = []
for tr, te, params in RESULTS[(KIND, MODE)]["folds"]:
    m = fit_model(KIND, params, X_b[tr], y_b[tr],
                  make_weights(g_b[tr], y_b[tr], MODE))
    Xte, yte, gte = X_b[te], y_b[te], g_b[te]
    groups = list(pd.Series(np.arange(len(te))).groupby(gte).indices.values())
    base = macro_auc(yte, m.predict_proba(Xte)[:, 1], gte)
    fold_row = []
    for j in range(len(FEATS)):
        reps = []
        for _ in range(10):
            Xp = Xte.copy()
            for idx in groups:
                Xp[idx, j] = rng_p.permutation(Xp[idx, j])
            reps.append(base - macro_auc(yte, m.predict_proba(Xp)[:, 1], gte))
        fold_row.append(np.mean(reps))
    fold_drops.append(fold_row)

fold_drops = np.array(fold_drops)
imp = pd.DataFrame({
    "feature": FEATS,
    "mean_macro_auc_drop": fold_drops.mean(axis=0),
    "fold_std": fold_drops.std(axis=0),
})
print("B_Cells, held-out folds (shuffle inside each protein)")
print(imp.round(4).to_string(index=False))

mask15 = len_s <= 15
base_s = roc_auc_score(y_s[mask15], final.predict_proba(X_s[mask15])[:, 1])
rows = []
for j, f in enumerate(FEATS):
    reps = []
    for _ in range(200):
        Xp = X_s[mask15].copy()
        Xp[:, j] = rng_p.permutation(Xp[:, j])
        reps.append(base_s - roc_auc_score(
            y_s[mask15], final.predict_proba(Xp)[:, 1]))
    rows.append({"feature": f, "mean_auc_drop": np.mean(reps),
                 "std": np.std(reps)})
print("\nSARS 5-15 aa (SARS is one protein, plain shuffle)")
print(pd.DataFrame(rows).round(4).to_string(index=False))
print("\nImportance describes this model. It does not prove biology.")


# ----------------------------------------------------------
# Check 5: ranked COVID candidates, 5-15 aa only
# ----------------------------------------------------------

banner("CHECK 5: COVID CANDIDATES (5-15 aa, scores only)")

cand = covid[(covid["peptide_length"] >= 5)
             & (covid["peptide_length"] <= 15)].copy()
cand["score"] = final.predict_proba(cand[FEATS].to_numpy())[:, 1]
cand = cand.sort_values("score", ascending=False).reset_index(drop=True)
cand["rank"] = np.arange(1, len(cand) + 1)

print("COVID rows:", len(covid), "kept (5-15 aa):", len(cand))
print("COVID score quantiles:")
print(cand["score"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(3))
print("B_Cells training score quantiles (in-sample):")
print(pd.Series(final.predict_proba(X_b)[:, 1])
      .quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(3))

keep_cols = ["rank", "score", "peptide_seq", "start_position",
             "end_position", "peptide_length",
             "chou_fasman", "emini", "kolaskar_tongaonkar", "parker"]
cand[keep_cols].to_csv(RESULTS_DIR / "COVID_ranked_all.csv", index=False)

chosen = []
for _, row in cand.iterrows():
    if all(
        row["end_position"] < c["start_position"]
        or row["start_position"] > c["end_position"]
        for c in chosen
    ):
        chosen.append(row)
    if len(chosen) == 25:
        break

if chosen:
    top = pd.DataFrame(chosen)[keep_cols]
else:
    top = pd.DataFrame(columns=keep_cols)

top.to_csv(RESULTS_DIR / "COVID_top_nonoverlapping.csv", index=False)

print("\nTop 25 non-overlapping candidates")
print(top.round(3).to_string(index=False))
print("\nThese are rankings, not validated epitopes. No COVID labels exist.")

print("\n" + "=" * 70)
print("FILE 05 COMPLETED")
print("=" * 70)