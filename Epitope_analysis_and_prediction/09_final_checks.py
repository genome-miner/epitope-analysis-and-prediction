# ==========================================================
# File: 09_final_checks.py
# Purpose: Equal-effort tuning, cluster-grouped CV, nulls,
# importance, partial dependence, clean SARS retest, COVID
# ranking, merged results table.
# Input: Data/B_Cells_cleaned.csv, Data/Sars_cleaned.csv,
#        Data/COVID_cleaned.csv
# Features: chou_fasman, log1p(emini), parker
# ==========================================================

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import partial_dependence
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold
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

RUN_SVM = True
SEEDS = [42, 7, 2024]
# Derive the candidate boundary from the supplied construct sequence.
# The complete 1,281-aa sequence remains unchanged in the dataset.
CONSTRUCT_END = None
TEV_SITE = "ENLYFQG"
LINKER_MOTIF = "GGGGS"
FOLDON_MOTIF = "GYIPEAPRDGQAYVRKDGEWVLLSTFL"

LINK_THRESHOLDS = [0.05, 0.20]
PRIMARY_LINK = 0.20
FEATS = ["chou_fasman", "emini_log", "parker"]

GRIDS = {
    "logistic": {"C": [0.01, 0.1, 1.0, 10.0]},
    "svm_rbf": {"C": [1.0, 4.0], "gamma": [0.1, 0.3333]},
    "boosting": {"max_depth": [2, 3], "learning_rate": [0.05, 0.1]},
}
DEFAULTS = {
    "logistic": {"C": 1.0},
    "svm_rbf": {"C": 1.0, "gamma": 0.3333},
    "boosting": {"max_depth": 3, "learning_rate": 0.05},
}
NAMES = ["logistic", "boosting"] + (["svm_rbf"] if RUN_SVM else [])


def banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load(name):
    path = DATA_DIR / name

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path)
    required = {
        "protein_seq", "peptide_seq",
        "start_position", "end_position",
        "chou_fasman", "emini", "parker"}

    if name != "COVID_cleaned.csv":
        required.add("target")

    if name == "B_Cells_cleaned.csv":
        required.add("parent_protein_id")

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name}: missing columns: {sorted(missing)}")

    if df.empty:
        raise ValueError(f"{name}: dataset is empty.")

    for col in ["protein_seq", "peptide_seq"]:
        if df[col].isna().any():
            raise ValueError(f"{name}: missing values in {col}.")
        df[col] = df[col].astype(str).str.strip().str.upper()

        if (df[col] == "").any():
            raise ValueError(f"{name}: empty sequences in {col}.")
    for col in ["start_position", "end_position"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any():
            raise ValueError(f"{name}: invalid values in {col}.")
        if not np.all(df[col] == np.floor(df[col])):
            raise ValueError(f"{name}: non-integer coordinates in {col}.")
        df[col] = df[col].astype(int)

    if (df["start_position"] < 1).any():
        raise ValueError(f"{name}: start positions must be 1-based.")
    if (df["end_position"] < df["start_position"]).any():
        raise ValueError(f"{name}: end position precedes start position.")

    df["peptide_length"] = (df["end_position"] - df["start_position"] + 1)

    if (df["end_position"] > df["protein_seq"].str.len()).any():
        raise ValueError(f"{name}: peptide coordinates exceed protein length.")

    expected = [
        protein[start - 1:end]
        for protein, start, end in zip(
            df["protein_seq"],
            df["start_position"],
            df["end_position"])]

    mismatch = df["peptide_seq"].to_numpy() != np.asarray(expected)

    if mismatch.any():
        raise ValueError(f"{name}: {int(mismatch.sum())} peptide sequences do not " "match their protein coordinates.")

    for col in ["chou_fasman", "emini", "parker"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if not np.isfinite(df[col]).all():
            raise ValueError(f"{name}: invalid or non-finite values in {col}.")
    if (df["emini"] < 0).any():
        raise ValueError(f"{name}: negative Emini values prevent log1p.")
    if "target" in df.columns:
        df["target"] = pd.to_numeric(df["target"], errors="coerce")
        if df["target"].isna().any():
            raise ValueError(f"{name}: missing or invalid target labels.")
        if not set(df["target"].unique()).issubset({0, 1}):
            raise ValueError(f"{name}: target must contain only 0 and 1.")

        df["target"] = df["target"].astype(int)

        # Remove peptide sequences with conflicting target labels.
        n_labels = df.groupby("peptide_seq")["target"].nunique()
        conflicts = n_labels[n_labels > 1].index
        df = df[~df["peptide_seq"].isin(conflicts)].copy()

    coordinate_cols = ["protein_seq", "start_position", "end_position"]

    duplicate_rows = df.duplicated(coordinate_cols, keep=False)
    if duplicate_rows.any() and "target" in df.columns:
        duplicate_labels = (
            df.loc[duplicate_rows]
            .groupby(coordinate_cols)["target"]
            .nunique()
        )

        if (duplicate_labels > 1).any():
            raise ValueError(f"{name}: duplicate protein coordinates have conflicting labels.")

    df = df.drop_duplicates(coordinate_cols).reset_index(drop=True)
    if df.empty:
        raise ValueError(f"{name}: no rows remain after cleaning.")
    df["emini_log"] = np.log1p(df["emini"])
    return df


def make_model(name, params):
    if name == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000,
                               **params))
    if name == "svm_rbf":
        return make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", class_weight="balanced", **params))
    return HistGradientBoostingClassifier(
        max_iter=200, min_samples_leaf=50, l2_regularization=1.0,
        random_state=42, **params)


def fit_model(name, params, X, y):
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) != 2:
        raise ValueError(f"{name}: training data must contain both target classes.")

    if not np.isfinite(X).all():
        raise ValueError(f"{name}: non-finite feature values found.")

    m = make_model(name, params)
    if name == "boosting":
        counts = np.bincount(y, minlength=2)
        if np.any(counts == 0):
            raise ValueError("Boosting training fold is missing a class.")
        class_weights = len(y) / (2.0 * counts)
        m.fit(X, y, sample_weight=class_weights[y])
    else:
        m.fit(X, y)
    return m


def score_of(m, X):
    try:
        return m.predict_proba(X)[:, 1]
    except (AttributeError, NotImplementedError):
        return m.decision_function(X)


def fast_auc(y, s):
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)

    if len(y) != len(s):
        raise ValueError("AUC labels and scores have different lengths.")

    if not np.isfinite(s).all():
        raise ValueError("AUC scores contain non-finite values.")

    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))

    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC requires both positive and negative examples.")

    ranks = rankdata(s)
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)



def group_indices(g):
    return list(pd.Series(np.arange(len(g))).groupby(g).indices.values())

def macro_auc(y, s, g, min_n=10):
    vals = []
    for idx in group_indices(g):
        if len(idx) >= min_n:
            yy = y[idx]
            if 0 < yy.sum() < len(yy):
                vals.append(fast_auc(yy, s[idx]))
    if not vals:
        raise ValueError("No eligible proteins were available for macro AUC calculation.")
    return float(np.mean(vals))


# ----------------------------------------------------------
# Load
# ----------------------------------------------------------

banner("STEP 0: LOAD")
b = load("B_Cells_cleaned.csv")
s = load("Sars_cleaned.csv")
covid = load("COVID_cleaned.csv")

# Verify that the COVID dataset contains one complete reference sequence.
covid_sequences = covid["protein_seq"].unique()

if len(covid_sequences) != 1:
    raise ValueError(
        f"Expected one COVID protein sequence, found {len(covid_sequences)}."
    )

covid_seq = covid_sequences[0]

# Locate the engineered construct elements.
tev_positions = [
    i for i in range(len(covid_seq))
    if covid_seq.startswith(TEV_SITE, i)
]

linker_positions = [
    i for i in range(len(covid_seq))
    if covid_seq.startswith(LINKER_MOTIF, i)
]

foldon_positions = [
    i for i in range(len(covid_seq))
    if covid_seq.startswith(FOLDON_MOTIF, i)
]

if len(tev_positions) != 1:
    raise ValueError(
        f"Expected one TEV site, found {len(tev_positions)}."
    )

if len(linker_positions) != 1:
    raise ValueError(
        f"Expected one linker motif, found {len(linker_positions)}."
    )

if len(foldon_positions) != 1:
    raise ValueError(
        f"Expected one foldon motif, found {len(foldon_positions)}."
    )

tev_start = tev_positions[0] + 1
linker_start = linker_positions[0] + 1
foldon_start = foldon_positions[0] + 1

# The native sequence ends immediately before the TEV cleavage site.
CONSTRUCT_END = tev_start - 1

# The final G of the TEV motif overlaps with the first G
# of the linker motif in the supplied sequence.

tev_end = tev_start + len(TEV_SITE) - 1

if not (
    linker_start == tev_end
    and foldon_start == linker_start + len(LINKER_MOTIF)
):
    raise ValueError("Unexpected arrangement of TEV site, linker and foldon.")

if CONSTRUCT_END <= 0:
    raise ValueError("Invalid COVID candidate boundary.")

print("B_Cells:", len(b), "SARS:", len(s), "COVID:", len(covid))
print("Models:", NAMES)
print("COVID full sequence length:", len(covid_seq))
print("TEV site starts at:", tev_start)
print("Linker starts at:", linker_start)
print("Foldon starts at:", foldon_start)
print("COVID candidate boundary:", CONSTRUCT_END)
print("Terminal His-tag residues:", len(covid_seq) - len(covid_seq.rstrip("H")))

y_b = b["target"].astype(int).to_numpy()
g_exact = b["protein_seq"].to_numpy()
pos_b = b["start_position"].to_numpy()
X_b = b[FEATS].to_numpy()
is_p = (b["parent_protein_id"] == "P59594").to_numpy()

if not is_p.any():
    raise ValueError("P59594 was not found in B_Cells. Cannot perform the planned exclusion.")
if is_p.all():
    raise ValueError("All B_Cells rows belong to P59594. No training data would remain.")
print("P59594 rows excluded from final training:", int(is_p.sum()))
print("Remaining B_Cells training rows:", int((~is_p).sum()))

X_s = s[FEATS].to_numpy()
y_s = s["target"].astype(int).to_numpy()
len_s = s["peptide_length"].to_numpy()

ELIG = [idx for idx in group_indices(g_exact)
        if len(idx) >= 10 and 0 < y_b[idx].sum() < len(idx)]
print("Proteins in macro AUC:", len(ELIG))

rng = np.random.default_rng(0)


def pp_auc(score):
    return np.array([fast_auc(y_b[i], score[i]) for i in ELIG])


def boot_mean(arr, n=2000):
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        raise ValueError("Cannot bootstrap an empty array.")
    if not np.isfinite(arr).all():
        raise ValueError("Bootstrap input contains non-finite values.")
    if n < 1:
        raise ValueError("Number of bootstrap samples must be positive.")
    boots = arr[rng.integers(0, len(arr), size=(n, len(arr)))].mean(axis=1)
    return np.percentile(boots, [2.5, 97.5])


def topk_stats(score, frac=0.10):
    prec, base = [], []
    for idx in ELIG:
        k = max(1, int(np.ceil(frac * len(idx))))
        order = np.argsort(-score[idx], kind="stable")[:k]
        prec.append(y_b[idx][order].mean())
        base.append(y_b[idx].mean())
    return np.array(prec), np.array(base)


# ----------------------------------------------------------
# Protein clusters by shared 5-mers
# ----------------------------------------------------------

banner("CHECK 1: CLUSTER-GROUPED CV (near-identical proteins stay together)")


def kmers(seq, k=5):
    return {seq[i:i + k] for i in range(len(seq) - k + 1)}


def pair_overlaps(seqs):
    sets = [kmers(x) for x in seqs]
    index = defaultdict(list)
    for i, ks in enumerate(sets):
        for km in ks:
            index[km].append(i)
    counts = Counter()
    for members in index.values():
        if len(members) > 1:
            for a in range(len(members)):
                for c in range(a + 1, len(members)):
                    counts[(members[a], members[c])] += 1
    return {pair: c / min(len(sets[pair[0]]), len(sets[pair[1]]))
            for pair, c in counts.items()}


def clusters_at(frac, n, threshold):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (i, j), f in frac.items():
        if f >= threshold:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj
    return np.array([find(i) for i in range(n)])


seqs = list(pd.unique(g_exact))
seq_index = {x: i for i, x in enumerate(seqs)}
count_by_seq = pd.Series(g_exact).value_counts()
rows_per_seq = np.array([count_by_seq[x] for x in seqs])
id_of = b.drop_duplicates("protein_seq").set_index("protein_seq")[
    "parent_protein_id"]
p_seq = b.loc[is_p, "protein_seq"].iloc[0]

frac = pair_overlaps(seqs)
GROUPS = {"exact protein": g_exact}
for thr in LINK_THRESHOLDS:
    lab = clusters_at(frac, len(seqs), thr)
    sizes = pd.Series(lab).value_counts()
    rows_cl = pd.Series(rows_per_seq).groupby(lab).sum()
    p_members = [id_of[seqs[i]] for i in np.where(lab == lab[seq_index[p_seq]])[0]]
    print(f"\nLink threshold {thr:.2f} (share of 5-mers in the smaller protein)")
    print("Proteins:", len(seqs), "clusters:", len(sizes),
          "proteins in clusters of size 2 or more:",
          int(sizes[sizes > 1].sum()))
    print("Largest cluster: proteins", int(sizes.max()),
          "rows", int(rows_cl.max()),
          "share of rows", round(rows_cl.max() / len(b), 3))
    print("Cluster holding P59594:", p_members[:10])
    GROUPS[f"cluster {thr:.2f}"] = np.array(
        [lab[seq_index[x]] for x in g_exact])

PRIMARY = f"cluster {PRIMARY_LINK:.2f}"
G_SPLIT = GROUPS[PRIMARY]
print("\nPrimary grouping for tuning and evaluation:", PRIMARY)


def cv_scores(name, params, X, y, g_split, seed):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    out = np.zeros(len(y))
    for tr, te in cv.split(X, y, g_split):
        m = fit_model(name, params, X[tr], y[tr])
        out[te] = score_of(m, X[te])
    return out


PP_G = {}
for gname, gs in GROUPS.items():
    for name in ["logistic", "boosting"]:
        arrs = [pp_auc(cv_scores(name, DEFAULTS[name], X_b, y_b, gs, sd))
                for sd in SEEDS]
        PP_G[(gname, name)] = np.mean(arrs, axis=0)
        print("done:", gname, name)

rows = []
for (gname, name), per in PP_G.items():
    lo, hi = boot_mean(per)
    row = {"grouping": gname, "model": name, "macro_auc": per.mean(),
           "low": lo, "high": hi}
    if gname != "exact protein":
        d = per - PP_G[("exact protein", name)]
        dl, dh = boot_mean(d)
        row.update({"diff_vs_exact": d.mean(), "diff_low": dl,
                    "diff_high": dh})
    rows.append(row)
print("\nMacro AUC, default settings, mean over 3 seeds")
print(pd.DataFrame(rows).round(3).to_string(index=False))
print("A negative difference means exact-sequence grouping inflated scores.")


# ----------------------------------------------------------
# Equal-effort tuning in nested CV
# ----------------------------------------------------------

banner("CHECK 2: NESTED TUNING, 4 SETTINGS PER MODEL, CLUSTER-GROUPED")


def grid_search(name, X, y, g_split, g_eval, n_splits, seed):
    cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed)

    splits = list(cv.split(X, y, g_split))
    best = None

    for params in ParameterGrid(GRIDS[name]):
        oof = np.full(len(y), np.nan)
        for itr, iva in splits:
            if len(np.unique(y[itr])) != 2:
                raise ValueError(
                    f"{name}: inner training fold is missing a class. "
                    "Review the number and distribution of protein groups.")

            if len(np.unique(y[iva])) != 2:
                raise ValueError(
                    f"{name}: inner validation fold is missing a class.")

            model = fit_model(name, params, X[itr], y[itr])
            oof[iva] = score_of(model, X[iva])

        if not np.isfinite(oof).all():
            raise ValueError(f"{name}: incomplete inner-fold predictions.")
        score = macro_auc(y, oof, g_eval)
        if best is None or score > best[0]:
            best = (score, params)
    if best is None:
        raise ValueError(f"{name}: no valid hyperparameter result.")
    return best


def nested_cv(name, seed=42):
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    tuned = np.zeros(len(y_b))
    default = np.zeros(len(y_b))
    chosen = []
    for tr, te in outer.split(X_b, y_b, G_SPLIT):
        _, params = grid_search(name, X_b[tr], y_b[tr], G_SPLIT[tr],
                                g_exact[tr], 3, seed + 1)
        chosen.append(params)
        tuned[te] = score_of(fit_model(name, params, X_b[tr], y_b[tr]),
                             X_b[te])
        default[te] = score_of(
            fit_model(name, DEFAULTS[name], X_b[tr], y_b[tr]), X_b[te])
    return tuned, default, chosen


TUNED = {}
RES = {}
BOOT = {}
rows = []
for name in NAMES:
    t_sc, d_sc, chosen = nested_cv(name)
    TUNED[name] = (t_sc, d_sc, chosen)
    p_t, p_d = pp_auc(t_sc), pp_auc(d_sc)
    t_lo, t_hi = boot_mean(p_t)
    d_lo, d_hi = boot_mean(p_d)
    diff = p_t - p_d
    df_lo, df_hi = boot_mean(diff)
    prec, base = topk_stats(t_sc)
    gain = prec - base
    g_lo, g_hi = boot_mean(gain)
    BOOT[name] = {"tuned": (p_t.mean(), t_lo, t_hi),
                  "default": (p_d.mean(), d_lo, d_hi)}
    RES[name] = {
        "model": name,
        "macro_auc_tuned": p_t.mean(), "tuned_low": t_lo, "tuned_high": t_hi,
        "macro_auc_default": p_d.mean(),
        "tuned_minus_default": diff.mean(), "diff_low": df_lo,
        "diff_high": df_hi,
        "pooled_auc": roc_auc_score(y_b, t_sc),
        "pr_auc": average_precision_score(y_b, t_sc),
        "top10_gain": gain.mean(), "gain_low": g_lo, "gain_high": g_hi,
    }
    rows.append(RES[name])
    print("done:", name)

print("\nChosen settings per outer fold")
for name in NAMES:
    print(name)
    for p in TUNED[name][2]:
        print("   ", p)
print("\nPR-AUC floor:", round(y_b.mean(), 3))
print(pd.DataFrame(rows).round(3).to_string(index=False))
print("Pooled AUC and PR-AUC mix fold score scales. Macro AUC does not.")

print("\nPaired macro AUC differences between tuned models (by protein)")
rows = []
for i in range(len(NAMES)):
    for j in range(i + 1, len(NAMES)):
        d = pp_auc(TUNED[NAMES[i]][0]) - pp_auc(TUNED[NAMES[j]][0])
        lo, hi = boot_mean(d)
        rows.append({"first": NAMES[i], "second": NAMES[j],
                     "mean_diff": d.mean(), "low": lo, "high": hi,
                     "excludes_zero": bool(lo > 0 or hi < 0)})
print(pd.DataFrame(rows).round(3).to_string(index=False))
print("Three paired tests plus three tuned-minus-default tests above.")
print("No multiple-comparison correction.")


# ----------------------------------------------------------
# Nulls: plain shuffle and circular shift by position
# ----------------------------------------------------------
banner("CHECK 3: RETRAINED WITHIN-PROTEIN NULL TESTS")

N_NULL, NULL_SEED, CV_SEED = 200, 2026, 42

def permute_within_proteins(y, groups, rng, positions=None, circular=False):
    y = np.asarray(y, dtype=int)
    out = y.copy()
    for idx in group_indices(groups):
        if circular:
            if positions is None:
                raise ValueError("Positions are required for circular permutation.")
            order = idx[np.argsort(positions[idx], kind="stable")]
            if len(order) > 1:
                shift = int(rng.integers(1, len(order)))
                out[order] = np.roll(y[order], shift)
        else:
            out[idx] = rng.permutation(y[idx])
    return out

def evaluate_fixed_cv(name, y_eval, seed=CV_SEED):
    y_eval = np.asarray(y_eval, dtype=int)
    if len(np.unique(y_eval)) != 2:
        return None
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.full(len(y_eval), np.nan)
    try:
        splits = list(cv.split(X_b, y_eval, G_SPLIT))
    except ValueError:
        return None
    for train_idx, test_idx in splits:
        if len(np.unique(y_eval[train_idx])) != 2:
            return None
        model = fit_model(name, DEFAULTS[name], X_b[train_idx], y_eval[train_idx])
        oof[test_idx] = score_of(model, X_b[test_idx])
    if not np.isfinite(oof).all():
        return None
    return macro_auc(y_eval, oof, g_exact)

rng_null = np.random.default_rng(NULL_SEED)
null_rows = []

for name in NAMES:
    print(f"\nRunning null tests for: {name}")
    observed = evaluate_fixed_cv(name, y_b, CV_SEED)
    if observed is None:
        raise ValueError(f"{name}: observed grouped CV could not be evaluated.")
    shuffle_null, circular_null = [], []

    for draw in range(N_NULL):
        y_shuffle = permute_within_proteins(y_b, g_exact, rng_null)
        shuffle_score = evaluate_fixed_cv(name, y_shuffle, CV_SEED)
        if shuffle_score is not None:
            shuffle_null.append(shuffle_score)

        y_circular = permute_within_proteins(y_b, g_exact, rng_null, positions=pos_b, circular=True)
        circular_score = evaluate_fixed_cv(name, y_circular, CV_SEED)
        if circular_score is not None:
            circular_null.append(circular_score)

        if (draw + 1) % 20 == 0:
            print(f"  Completed {draw + 1}/{N_NULL} draws", flush=True)

    shuffle_null = np.asarray(shuffle_null, dtype=float)
    circular_null = np.asarray(circular_null, dtype=float)
    min_valid = int(np.ceil(0.90 * N_NULL))

    if len(shuffle_null) < min_valid:
        raise ValueError(f"{name}: only {len(shuffle_null)} valid shuffle draws; at least 90% required.")
    if len(circular_null) < min_valid:
        raise ValueError(f"{name}: only {len(circular_null)} valid circular draws; at least 90% required.")

    p_shuffle = (1 + np.sum(shuffle_null >= observed)) / (1 + len(shuffle_null))
    p_circular = (1 + np.sum(circular_null >= observed)) / (1 + len(circular_null))

    null_rows.append({
        "model": name,
        "observed_default_macro_auc": observed,
        "shuffle_null_mean": shuffle_null.mean(),
        "shuffle_95th": np.percentile(shuffle_null, 95),
        "shuffle_valid_draws": len(shuffle_null),
        "shuffle_p": p_shuffle,
        "circular_null_mean": circular_null.mean(),
        "circular_95th": np.percentile(circular_null, 95),
        "circular_valid_draws": len(circular_null),
        "circular_p": p_circular
    })

    RES[name]["null_default_macro_auc"] = observed
    RES[name]["p_shuffle"] = p_shuffle
    RES[name]["p_circular"] = p_circular

null_df = pd.DataFrame(null_rows)

print("\nRetained within-protein label counts in each null replicate.")
print("Models were retrained for every valid permutation.")
print("All tests use fixed default settings and grouped 5-fold CV.")
print(null_df.round(4).to_string(index=False))
print("\nEmpirical p-value resolution depends on the number of valid null draws.")
print("These null tests evaluate fixed-default models, not nested-tuned models from Check 2.")
print("Circular shifts preserve within-protein label runs but rotate their positions.")

# ----------------------------------------------------------
# Permutation importance inside proteins
# ----------------------------------------------------------

banner("CHECK 4: WITHIN-PROTEIN PERMUTATION IMPORTANCE (default settings)")


def perm_importance(name, reps=10, seed=42):
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    r = np.random.default_rng(3)
    drops = []
    for tr, te in outer.split(X_b, y_b, G_SPLIT):
        m = fit_model(name, DEFAULTS[name], X_b[tr], y_b[tr])
        Xte, yte, gte = X_b[te], y_b[te], g_exact[te]
        groups = group_indices(gte)
        base = macro_auc(yte, score_of(m, Xte), gte)
        row = []
        for j in range(X_b.shape[1]):
            vals = []
            for _ in range(reps):
                Xp = Xte.copy()
                for idx in groups:
                    Xp[idx, j] = r.permutation(Xp[idx, j])
                vals.append(base - macro_auc(yte, score_of(m, Xp), gte))
            row.append(np.mean(vals))
        drops.append(row)
    return np.array(drops)


IMP = {}
rows = []
for name in ["logistic", "boosting"]:
    d = perm_importance(name)
    IMP[name] = d
    for j, f in enumerate(FEATS):
        rows.append({"model": name, "feature": f,
                     "macro_auc_drop": d[:, j].mean(),
                     "fold_std": d[:, j].std()})
print(pd.DataFrame(rows).round(4).to_string(index=False))
print("Shuffling one score creates combinations absent from real peptides.")
print("Correlated scores look more important than they are.")


# ----------------------------------------------------------
# Final models without P59594, SARS retest, partial dependence
# ----------------------------------------------------------

banner("CHECK 5: FINAL MODELS WITHOUT P59594, SARS RETEST")

tr_mask = ~is_p
FINAL = {}
for name in NAMES:
    _, params = grid_search(name, X_b[tr_mask], y_b[tr_mask],
                            G_SPLIT[tr_mask], g_exact[tr_mask], 5, 44)
    FINAL[name] = (params, fit_model(name, params, X_b[tr_mask],
                                     y_b[tr_mask]))
    print(name, "selected:", params)

sc = {name: score_of(FINAL[name][1], X_s) for name in NAMES}
sc["emini_log alone"] = s["emini_log"].to_numpy()

SUBSETS = {"5-15 aa": len_s <= 15, "5-14 aa": len_s <= 14,
           "17 aa": len_s == 17, "all": len_s >= 0}
rng_s = np.random.default_rng(1)
rows = []

for sub, mask in SUBSETS.items():
    idx_all = np.where(mask)[0]
    if len(idx_all) == 0:
        raise ValueError(f"SARS subset '{sub}' contains no rows.")

    if len(np.unique(y_s[idx_all])) != 2:
        raise ValueError(f"SARS subset '{sub}' does not contain both target classes.")
    boots = {k: [] for k in sc}

    for _ in range(1000):
        idx = rng_s.choice(idx_all, size=len(idx_all), replace=True)
        yy = y_s[idx]
        if len(np.unique(yy)) != 2:
            continue
        for k, values in sc.items():
            boots[k].append(fast_auc(yy, values[idx]))

    valid_counts = {k: len(v) for k, v in boots.items()}

    if min(valid_counts.values()) < 200:
        raise ValueError(
            f"Too few valid bootstrap samples for SARS subset '{sub}': "
            f"{valid_counts}")

    # All scorers use the same valid bootstrap resamples.
    lengths = {len(v) for v in boots.values()}

    if len(lengths) != 1:
        raise ValueError("SARS bootstrap samples are not paired.")
    ref = np.asarray(boots["emini_log alone"])
    for k, values in sc.items():
        arr = np.asarray(boots[k])
        lo, hi = np.percentile(arr, [2.5, 97.5])
        row = {
            "subset": sub,
            "rows": int(mask.sum()),
            "positives": int(y_s[mask].sum()),
            "scorer": k,
            "auc": fast_auc(y_s[mask], values[mask]),
            "low": lo,
            "high": hi}

        if k != "emini_log alone":
            differences = arr - ref
            diff_low, diff_high = np.percentile(
                differences, [2.5, 97.5])

            row.update({
                "diff_vs_emini": (
                    fast_auc(y_s[mask], values[mask])
                    - fast_auc(
                        y_s[mask],
                        sc["emini_log alone"][mask])
                ),
                "diff_low": diff_low,
                "diff_high": diff_high
            })
        rows.append(row)

sars_df = pd.DataFrame(rows)
sars_df.round(4).to_csv(RESULTS_DIR / "09_sars_clean.csv", index=False)
print(sars_df.round(3).to_string(index=False))
print("\nSARS is one protein with overlapping peptides, so intervals run")
print("narrow. SARS was not used for tuning. The 15 aa stratum holds")
print("4 positives and stays untestable.")

for name in NAMES:
    for sub, key in [("5-15 aa", "sars_5_15"), ("all", "sars_all")]:
        hit = sars_df[(sars_df["scorer"] == name)
                      & (sars_df["subset"] == sub)].iloc[0]
        RES[name][key + "_auc"] = hit["auc"]
        RES[name][key + "_low"] = hit["low"]
        RES[name][key + "_high"] = hit["high"]

X_tr = X_b[tr_mask][::3]


def pdp_curve(model, X, j):
    res = partial_dependence(model, X, features=[j], kind="average",
                             grid_resolution=30, method="brute")
    grid = res["grid_values"][0] if "grid_values" in res else res["values"][0]
    return grid, res["average"][0]


fig, axes = plt.subplots(1, len(FEATS), figsize=(15, 4.5), sharey=True)
for j, f in enumerate(FEATS):
    for name in ["logistic", "boosting"]:
        grid, avg = pdp_curve(FINAL[name][1], X_tr, j)
        axes[j].plot(grid, avg, label=name)
    axes[j].set_xlabel(f)
    axes[j].set_title(f)
axes[0].set_ylabel("Partial dependence (predicted probability)")
axes[0].legend()
fig.suptitle("Figure 13: Partial Dependence, 3 Features, Trained Without "
             "P59594", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(FIG_DIR / "Fig13_Partial_Dependence_3feat.png", dpi=300,
            bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5.5))
width = 0.36
x0 = np.arange(len(NAMES))
for k, (label, key) in enumerate([("default settings", "default"),
                                  ("tuned (nested)", "tuned")]):
    means = np.array([BOOT[n][key][0] for n in NAMES])
    lows = np.array([BOOT[n][key][1] for n in NAMES])
    highs = np.array([BOOT[n][key][2] for n in NAMES])
    ax.errorbar(x0 + (k - 0.5) * width, means,
                yerr=[means - lows, highs - means],
                fmt="o", capsize=5, markersize=8, label=label)
ax.axhline(0.5, color="k", linestyle="--", label="chance")
ax.set_xticks(x0)
ax.set_xticklabels(NAMES)
ax.set_ylim(0.50, 0.65)
ax.set_ylabel("Macro within-protein AUC")
ax.set_title("Figure 14: Tuned Against Default Settings "
             "(95% protein-bootstrap)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "Fig14_Tuned_vs_Default.png", dpi=300,
            bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5))
w = 0.38
for k, name in enumerate(["logistic", "boosting"]):
    ax.bar(np.arange(len(FEATS)) + (k - 0.5) * w, IMP[name].mean(axis=0),
           width=w, yerr=IMP[name].std(axis=0), capsize=4, label=name,
           edgecolor="black")
ax.set_xticks(np.arange(len(FEATS)))
ax.set_xticklabels(FEATS)
ax.set_ylabel("Drop in macro AUC after shuffling inside proteins")
ax.set_title("Figure 15: Permutation Importance (bars show fold std)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "Fig15_Permutation_Importance.png", dpi=300,
            bbox_inches="tight")
plt.close(fig)
print("\nSaved: Fig13, Fig14, Fig15")


# ----------------------------------------------------------
# COVID ranking
# ----------------------------------------------------------

banner("CHECK 6: COVID RANKING, VERIFIED NATIVE-SEQUENCE BOUNDARY")

cand = covid[
    (covid["peptide_length"] >= 5)
    & (covid["peptide_length"] <= 15)
    & (covid["start_position"] >= 1)
    & (covid["end_position"] <= CONSTRUCT_END)
].copy()

if cand.empty:
    raise ValueError("No COVID candidates remain after sequence filtering.")

print("Full COVID sequence length:", len(covid_seq))
print("Candidate boundary:", CONSTRUCT_END)
print("COVID rows:", len(covid))
print("Candidates retained:", len(cand))
print("Largest candidate end position:", int(cand["end_position"].max()))

# Confirm candidate sequences against the complete reference sequence.
expected_peptides = [
    covid_seq[start - 1:end]
    for start, end in zip(
        cand["start_position"],
        cand["end_position"])]

if not np.array_equal(
    cand["peptide_seq"].to_numpy(),
    np.asarray(expected_peptides)):
    raise ValueError(
        "COVID candidate sequences do not match their reference coordinates."
    )

Xc = cand[FEATS].to_numpy()

if not np.isfinite(Xc).all():
    raise ValueError("COVID candidate features contain invalid values.")
for name in NAMES:
    cand["score_" + name] = score_of(FINAL[name][1], Xc)

    if not np.isfinite(cand["score_" + name]).all():
        raise ValueError(f"Non-finite COVID scores from {name}.")

    cand["pct_" + name] = cand["score_" + name].rank(pct=True)

cand["score"] = cand[[f"pct_{name}" for name in NAMES]].mean(axis=1)
cand["len_pct"] = cand.groupby("peptide_length")["score"].rank(pct=True)
cols = [f"score_{name}" for name in NAMES] + ["emini_log"]
print("\nSpearman between model scores and Emini alone")
print(cand[cols].corr(method="spearman").round(3))
k_top = max(1, int(np.ceil(0.01 * len(cand))))
top_ens = set(cand.nlargest(k_top, "score").index)
top_em = set(cand.nlargest(k_top, "emini_log").index)
print("\nTop 1 percent:", k_top)
print("Overlap between ensemble and Emini alone:", round(len(top_ens & top_em) / k_top, 3))
top1 = cand.loc[list(top_ens)]
mix = pd.DataFrame({
    "share_all": cand["peptide_length"].value_counts(normalize=True),
    "share_top_1pct": top1["peptide_length"].value_counts(normalize=True)
}).fillna(0).sort_index()
print("\nLength distribution: all candidates versus top 1 percent")
print(mix.round(3))
MAX_PER_LENGTH = 3
ranked = cand.sort_values(["len_pct", "score", "start_position", "peptide_seq"], ascending=[False, False, True, True])

chosen = []
per_len = {}

for _, row in ranked.iterrows():
    peptide_len = int(row["peptide_length"])

    if per_len.get(peptide_len, 0) >= MAX_PER_LENGTH:
        continue

    overlaps = any(
        not (
            row["end_position"] < selected["start_position"]
            or row["start_position"] > selected["end_position"]
        )
        for selected in chosen)

    if overlaps:
        continue

    chosen.append(row)
    per_len[peptide_len] = per_len.get(peptide_len, 0) + 1

    if len(chosen) == 25:
        break

if not chosen:
    raise ValueError("No non-overlapping COVID candidates were selected.")

keep = ["len_pct", "score", "peptide_seq",
        "start_position", "end_position", "peptide_length",
        "chou_fasman", "emini", "parker"]

top = pd.DataFrame(chosen)[keep].reset_index(drop=True)
top.to_csv(RESULTS_DIR / "COVID_top_final.csv", index=False)

cand.sort_values(
    ["score", "start_position", "peptide_seq"],
    ascending=[False, True, True]
)[keep + [f"score_{name}" for name in NAMES]].to_csv(RESULTS_DIR / "COVID_ranked_final.csv", index=False)

print("\nSelected non-overlapping candidates:", len(top), "with a maximum of", MAX_PER_LENGTH, "per peptide length.")
print(top.round(3).to_string(index=False))
print("\nCandidate ranking is computational and does not validate epitopes.")
print("The complete supplied sequence remains 1,281 aa.")
print("Engineered C-terminal construct elements are excluded from ranking.")
print("B_Cells training excludes SARS-CoV P59594.")
print("Related proteins may still remain in the training dataset.")

# ----------------------------------------------------------
# Merged results table
# ----------------------------------------------------------

banner("CHECK 7: MERGED RESULTS TABLE")

final_table = pd.DataFrame(list(RES.values()))
final_table.round(4).to_csv(RESULTS_DIR / "09_results_table.csv", index=False)
print(final_table.round(3).T.to_string())
print("\nSaved: Data/09_results_table.csv")

print("\n" + "=" * 70)
print("FILE 09 COMPLETED")
print("=" * 70)