# ==========================================================
# File: 11_final_seeds.py
# Purpose: Stable cluster folds, five outer seeds, model comparison,
# paired tests, SARS retest, and COVID candidate ranking.
# Inputs: Data/B_Cells_cleaned.csv, Data/Sars_cleaned.csv,
#         Data/COVID_cleaned.csv
# ==========================================================

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

pd.set_option("display.width", 230)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 300)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
RESULTS_DIR = BASE_DIR / "Results"
FIG_DIR = BASE_DIR / "Figures"

SEEDS = [42, 7, 2024, 11, 99]
LINK = 0.20
CONSTRUCT_END = 1234
MAX_PER_LENGTH = 3
N_BOOT = 2000
N_SIGNFLIP = 50000
FEATS = ["chou_fasman", "emini_log", "parker"]

DEFAULTS = {
    "logistic": {"C": 1.0},
    "svm_rbf": {"C": 1.0, "gamma": 0.3333},
    "boosting": {"max_depth": 3, "learning_rate": 0.05},
    "forest": {"max_depth": 8, "min_samples_leaf": 20},
    "tree": {"max_depth": 4, "min_samples_leaf": 50},
}
MODEL_NAMES = list(DEFAULTS)
ENSEMBLE_PARTS = ["logistic", "svm_rbf", "boosting", "forest"]
CMP_NAMES = MODEL_NAMES + ["ensemble", "emini_alone"]

rng_boot = np.random.default_rng(2026)
rng_tests = np.random.default_rng(2027)


def banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def require_columns(df, columns, filename):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{filename}: missing required columns: {missing}")


def load(name, require_target):
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path)
    required = [
        "protein_seq", "peptide_seq", "start_position", "end_position",
        "emini", "chou_fasman", "parker"
    ]
    if require_target:
        required += ["target", "parent_protein_id"]
    require_columns(df, required, name)

    if df[required].isna().any().any():
        bad = df[required].columns[df[required].isna().any()].tolist()
        raise ValueError(f"{name}: missing values in required columns: {bad}")

    for col in ["start_position", "end_position"]:
        values = pd.to_numeric(df[col], errors="coerce")
        if values.isna().any() or not np.all(values == np.floor(values)):
            raise ValueError(f"{name}: {col} must contain integer coordinates.")
        df[col] = values.astype(int)

    if (df["start_position"] < 1).any():
        raise ValueError(f"{name}: coordinates must be 1-based.")
    if (df["end_position"] < df["start_position"]).any():
        raise ValueError(f"{name}: end_position precedes start_position.")

    df["peptide_length"] = df["end_position"] - df["start_position"] + 1
    if (df["peptide_length"] != df["peptide_seq"].astype(str).str.len()).any():
        raise ValueError(f"{name}: peptide lengths do not match coordinates.")

    protein_lengths = df["protein_seq"].astype(str).str.len()
    if (df["end_position"].to_numpy() > protein_lengths.to_numpy()).any():
        raise ValueError(f"{name}: peptide coordinates exceed parent protein length.")

    seq_match = []
    for row in df.itertuples(index=False):
        segment = str(row.protein_seq)[row.start_position - 1:row.end_position]
        seq_match.append(segment == str(row.peptide_seq))
    if not all(seq_match):
        raise ValueError(f"{name}: peptide sequences do not match coordinate-derived protein segments.")

    numeric = ["emini", "chou_fasman", "parker"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if not np.isfinite(df[col].to_numpy(dtype=float)).all():
            raise ValueError(f"{name}: {col} contains non-numeric or non-finite values.")
    if (df["emini"] < 0).any():
        raise ValueError(f"{name}: Emini values must be non-negative.")

    if require_target:
        df["target"] = pd.to_numeric(df["target"], errors="coerce")
        if df["target"].isna().any() or not set(df["target"].unique()).issubset({0, 1}):
            raise ValueError(f"{name}: target must contain only 0 and 1.")
        n_labels = df.groupby("peptide_seq")["target"].nunique()
        conflicts = n_labels[n_labels > 1].index
        n_conflicts = int(df["peptide_seq"].isin(conflicts).sum())
        df = df[~df["peptide_seq"].isin(conflicts)].copy()
    else:
        n_conflicts = 0

    before = len(df)
    df = df.drop_duplicates(["protein_seq", "start_position", "end_position"])
    n_duplicates = before - len(df)
    df = df.reset_index(drop=True)
    df["emini_log"] = np.log1p(df["emini"])

    if require_target and set(df["target"].astype(int).unique()) != {0, 1}:
        raise ValueError(f"{name}: both target classes must remain after cleaning.")

    print(f"{name}: retained {len(df)} rows, removed {n_conflicts} conflicting-label rows "
          f"and {n_duplicates} duplicate coordinate rows.")
    return df


def make_model(name, params):
    if name == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, **params)
        )
    if name == "svm_rbf":
        return make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", class_weight="balanced", **params)
        )
    if name == "forest":
        return RandomForestClassifier(
            n_estimators=100, class_weight="balanced", n_jobs=-1,
            random_state=42, **params
        )
    if name == "tree":
        return DecisionTreeClassifier(
            class_weight="balanced", random_state=42, **params
        )
    return HistGradientBoostingClassifier(
        max_iter=200, min_samples_leaf=50, l2_regularization=1.0,
        random_state=42, **params
    )


def fit_model(name, X, y):
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) != 2:
        raise ValueError(f"{name}: training fold does not contain both target classes.")
    model = make_model(name, DEFAULTS[name])
    if name == "boosting":
        counts = np.bincount(y, minlength=2)
        weights = len(y) / (2.0 * counts)
        model.fit(X, y, sample_weight=weights[y])
    else:
        model.fit(X, y)
    return model


def score_of(model, X):
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
    else:
        scores = model.decision_function(X)
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("Model produced non-finite prediction scores.")
    return scores


def fast_auc(y, scores):
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(y) != len(scores) or len(y) == 0:
        raise ValueError("AUC inputs must have equal, non-zero lengths.")
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC requires both positive and negative examples.")
    if not np.isfinite(scores).all():
        raise ValueError("AUC scores contain non-finite values.")
    ranks = rankdata(scores)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def group_indices(groups):
    return list(pd.Series(np.arange(len(groups))).groupby(groups, sort=False).indices.values())


def macro_auc(y, scores, groups, min_n=10):
    values = []
    for idx in group_indices(groups):
        if len(idx) >= min_n and 0 < np.sum(y[idx]) < len(idx):
            values.append(fast_auc(y[idx], scores[idx]))
    if not values:
        raise ValueError("No eligible protein groups for macro AUC.")
    return np.asarray(values, dtype=float)


def boot_ci(values, n=N_BOOT):
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Bootstrap input is empty or non-finite.")
    draws = values[rng_boot.integers(0, len(values), size=(n, len(values)))]
    means = draws.mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def signflip_p(differences, n=N_SIGNFLIP, chunk=1000):
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    if differences.size == 0:
        raise ValueError("Sign-flip test received no valid paired differences.")
    observed = abs(float(differences.mean()))
    extreme = 0
    completed = 0
    while completed < n:
        size = min(chunk, n - completed)
        signs = rng_tests.choice(np.array([-1.0, 1.0]), size=(size, len(differences)))
        null_means = np.abs((signs * differences).mean(axis=1))
        extreme += int(np.sum(null_means >= observed - 1e-15))
        completed += size
    return float((extreme + 1) / (n + 1))


def holm(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    if pvalues.size == 0 or not np.isfinite(pvalues).all():
        raise ValueError("Holm correction requires a non-empty finite p-value array.")
    if ((pvalues < 0) | (pvalues > 1)).any():
        raise ValueError("P-values must lie between 0 and 1.")
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=float)
    running = 0.0
    m = len(pvalues)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def topk_gain(scores, y, eligible_groups, frac=0.10, seed=0):
    random = np.random.default_rng(seed)
    gains = []
    for idx in eligible_groups:
        k = max(1, int(np.ceil(frac * len(idx))))
        tie = random.random(len(idx))
        order = np.lexsort((tie, -scores[idx]))[:k]
        gains.append(y[idx][order].mean() - y[idx].mean())
    return np.asarray(gains, dtype=float)


def kmers(sequence, k=5):
    return {sequence[i:i + k] for i in range(len(sequence) - k + 1)}


def pair_overlaps(sequences):
    sets = [kmers(sequence) for sequence in sequences]
    index = defaultdict(list)
    for i, kmerset in enumerate(sets):
        for kmer in kmerset:
            index[kmer].append(i)
    counts = Counter()
    for members in index.values():
        if len(members) > 1:
            for a in range(len(members)):
                for c in range(a + 1, len(members)):
                    counts[(members[a], members[c])] += 1
    return {
        pair: count / min(len(sets[pair[0]]), len(sets[pair[1]]))
        for pair, count in counts.items()
    }


def build_clusters(protein_sequences, threshold=LINK):
    sequences = list(pd.unique(protein_sequences))
    sequence_index = {sequence: i for i, sequence in enumerate(sequences)}
    overlaps = pair_overlaps(sequences)
    parent = list(range(len(sequences)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (i, j), fraction in overlaps.items():
        if fraction >= threshold:
            root_i, root_j = find(i), find(j)
            if root_i != root_j:
                parent[root_i] = root_j

    roots = np.array([find(i) for i in range(len(sequences))])
    labels = pd.factorize(roots)[0]
    row_ids = pd.Series(protein_sequences).map(sequence_index).to_numpy()
    if pd.isna(row_ids).any():
        raise ValueError("Could not map every protein sequence to a cluster.")
    return labels[row_ids.astype(int)], len(sequences), len(np.unique(labels))


def ensemble_scores(score_dict):
    ranks = [rankdata(score_dict[name]) / len(score_dict[name])
             for name in ENSEMBLE_PARTS]
    return np.mean(ranks, axis=0)


# ----------------------------------------------------------
# Load and validate
# ----------------------------------------------------------

banner("STEP 0: LOAD AND VALIDATE")
b = load("B_Cells_cleaned.csv", require_target=True)
s = load("Sars_cleaned.csv", require_target=True)
covid = load("COVID_cleaned.csv", require_target=False)

y_b = b["target"].astype(int).to_numpy()
g_exact = b["protein_seq"].astype(str).to_numpy()
X_b = b[FEATS].to_numpy(dtype=float)
emini_b = b["emini_log"].to_numpy(dtype=float)
is_p = (b["parent_protein_id"].astype(str).to_numpy() == "P59594")
X_s = s[FEATS].to_numpy(dtype=float)
y_s = s["target"].astype(int).to_numpy()
len_s = s["peptide_length"].to_numpy(dtype=int)

if "parent_protein_id" not in b.columns:
    raise ValueError("B_Cells_cleaned.csv is missing parent_protein_id.")
if not np.isfinite(X_b).all() or not np.isfinite(X_s).all():
    raise ValueError("Model feature matrices contain non-finite values.")

print("B_Cells:", len(b), "SARS:", len(s), "COVID:", len(covid))
print("Seeds:", SEEDS)
print("P59594 B_Cells rows:", int(is_p.sum()))

ELIG = [
    idx for idx in group_indices(g_exact)
    if len(idx) >= 10 and 0 < y_b[idx].sum() < len(idx)
]
print("Proteins eligible for macro AUC:", len(ELIG))
if len(ELIG) == 0:
    raise ValueError("No eligible protein groups for macro AUC.")

# ----------------------------------------------------------
# Stable cluster groups and reproducibility checksum
# ----------------------------------------------------------

banner("CHECK 1: STABLE CLUSTER GROUPING")
G_SPLIT, n_proteins, n_clusters = build_clusters(g_exact, LINK)
print("Unique protein sequences:", n_proteins)
print("Clusters at overlap threshold", LINK, ":", n_clusters)

check_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
fold42 = np.zeros(len(y_b), dtype=np.int64)
for fold, (train_idx, test_idx) in enumerate(check_cv.split(X_b, y_b, G_SPLIT)):
    fold42[test_idx] = fold

fold_checksum = int((fold42 * np.arange(1, len(y_b) + 1)).sum())
cluster_checksum = int((G_SPLIT * np.arange(1, len(y_b) + 1)).sum())
print("Fold checksum (seed 42):", fold_checksum)
print("Cluster checksum:", cluster_checksum)
print("Run this file twice. Equal checksums indicate stable folds for the same input row order.")

# ----------------------------------------------------------
# Five outer seeds, fixed default model settings
# ----------------------------------------------------------

banner("CHECK 2: FIVE OUTER SEEDS, DEFAULT SETTINGS")
PP = {name: [] for name in CMP_NAMES}
GAIN = {name: [] for name in CMP_NAMES}
protein_auc_emini = np.asarray([fast_auc(y_b[idx], emini_b[idx]) for idx in ELIG])

for seed in SEEDS:
    print(f"\nStarting outer seed {seed}", flush=True)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = {name: np.zeros(len(y_b), dtype=float) for name in MODEL_NAMES}
    ensemble_oof = np.zeros(len(y_b), dtype=float)

    for fold, (train_idx, test_idx) in enumerate(cv.split(X_b, y_b, G_SPLIT), start=1):
        fold_ranks = []
        for name in MODEL_NAMES:
            model = fit_model(name, X_b[train_idx], y_b[train_idx])
            scores = score_of(model, X_b[test_idx])
            oof[name][test_idx] = scores
            if name in ENSEMBLE_PARTS:
                fold_ranks.append(rankdata(scores) / len(scores))
        ensemble_oof[test_idx] = np.mean(fold_ranks, axis=0)
        print(f"  seed {seed}: fold {fold}/5 complete", flush=True)

    oof["ensemble"] = ensemble_oof
    for name in MODEL_NAMES + ["ensemble"]:
        PP[name].append(np.asarray([
            fast_auc(y_b[idx], oof[name][idx]) for idx in ELIG
        ]))
        GAIN[name].append(topk_gain(oof[name], y_b, ELIG, seed=seed))

    PP["emini_alone"].append(protein_auc_emini)
    GAIN["emini_alone"].append(topk_gain(emini_b, y_b, ELIG, seed=seed))
    print("Finished seed", seed, {
        name: round(float(np.mean(PP[name][-1])), 4)
        for name in MODEL_NAMES + ["ensemble"]
    }, flush=True)

PP = {name: np.asarray(values, dtype=float) for name, values in PP.items()}
GAIN = {name: np.asarray(values, dtype=float) for name, values in GAIN.items()}
PPM = {name: PP[name].mean(axis=0) for name in CMP_NAMES}

# ----------------------------------------------------------
# Results across five seeds
# ----------------------------------------------------------

banner("CHECK 3: RESULTS OVER FIVE SEEDS")
rows = []
for name in CMP_NAMES:
    seed_means = PP[name].mean(axis=1)
    auc_low, auc_high = boot_ci(PPM[name])
    gain_seed_means = GAIN[name].mean(axis=1)
    gain_low, gain_high = boot_ci(GAIN[name].mean(axis=0))
    rows.append({
        "model": name,
        "macro_auc_mean": seed_means.mean(),
        "seed_std": seed_means.std(ddof=1),
        "seed_min": seed_means.min(),
        "seed_max": seed_means.max(),
        "boot_low": auc_low,
        "boot_high": auc_high,
        "top10_gain": gain_seed_means.mean(),
        "gain_low": gain_low,
        "gain_high": gain_high,
    })

results = pd.DataFrame(rows)
print(results.to_string(index=False, float_format=lambda x: f"{x:.8g}"))
print("Seed standard deviation describes variation across the five outer splits.")
print("Bootstrap intervals resample the 143 eligible exact-protein groups.")
print("Dummy is not included because File 10 already reports the dummy baseline.")
results.to_csv(RESULTS_DIR / "11_results_table.csv", index=False, float_format="%.10g")

print("\nMacro AUC per seed")
print(pd.DataFrame(
    {name: PP[name].mean(axis=1) for name in CMP_NAMES},
    index=SEEDS
).round(4).to_string())

# ----------------------------------------------------------
# Paired protein-level sign-flip tests with Holm correction
# ----------------------------------------------------------

banner("CHECK 4: PAIRED SIGN-FLIP TESTS, HOLM CORRECTED")
pair_rows = []
for i in range(len(CMP_NAMES)):
    for j in range(i + 1, len(CMP_NAMES)):
        first, second = CMP_NAMES[i], CMP_NAMES[j]
        differences = PPM[first] - PPM[second]
        low, high = boot_ci(differences)
        seed_differences = PP[first].mean(axis=1) - PP[second].mean(axis=1)
        pair_rows.append({
            "first": first,
            "second": second,
            "mean_diff": differences.mean(),
            "bootstrap_low": low,
            "bootstrap_high": high,
            "paired_signflip_p": signflip_p(differences),
            "seeds_first_higher": int((seed_differences > 0).sum()),
            "seed_ties": int((seed_differences == 0).sum()),
        })

pairwise = pd.DataFrame(pair_rows)
pairwise["p_holm"] = holm(pairwise["paired_signflip_p"])
pairwise["significant_after_holm"] = pairwise["p_holm"] < 0.05
print(pairwise.to_string(index=False, float_format=lambda x: f"{x:.8g}"))
print(len(pairwise), "paired comparisons. P-values use protein-level sign flips.")
pairwise.to_csv(RESULTS_DIR / "11_pairwise_tests.csv", index=False, float_format="%.10g")

print("\nEach scorer versus chance macro AUC 0.5")
chance_rows = []
for name in CMP_NAMES:
    differences = PPM[name] - 0.5
    low, high = boot_ci(differences)
    chance_rows.append({
        "scorer": name,
        "mean_diff": differences.mean(),
        "bootstrap_low": low,
        "bootstrap_high": high,
        "signflip_p": signflip_p(differences),
    })
chance = pd.DataFrame(chance_rows)
chance["p_holm"] = holm(chance["signflip_p"])
chance["significant_after_holm"] = chance["p_holm"] < 0.05
print(chance.to_string(index=False, float_format=lambda x: f"{x:.8g}"))
chance.to_csv(RESULTS_DIR / "11_chance_tests.csv", index=False, float_format="%.10g")

# ----------------------------------------------------------
# Final models without P59594
# ----------------------------------------------------------

banner("CHECK 5: FINAL MODELS TRAINED WITHOUT P59594")
train_mask = ~is_p
if train_mask.sum() == 0 or len(np.unique(y_b[train_mask])) != 2:
    raise ValueError("Training data without P59594 must contain both target classes.")

final_models = {}
for name in MODEL_NAMES:
    final_models[name] = fit_model(name, X_b[train_mask], y_b[train_mask])
    print("Trained final model:", name, flush=True)

print("Final training rows:", int(train_mask.sum()))
print("Final training positives:", int(y_b[train_mask].sum()))

# ----------------------------------------------------------
# SARS retest, trained without P59594
# ----------------------------------------------------------

banner("CHECK 6: SARS RETEST, TRAINED WITHOUT P59594")
sars_scores = {
    name: score_of(final_models[name], X_s)
    for name in MODEL_NAMES
}
sars_scores["ensemble"] = ensemble_scores(sars_scores)
sars_scores["emini_alone"] = s["emini_log"].to_numpy(dtype=float)

SUBSETS = {
    "5-15 aa": (len_s >= 5) & (len_s <= 15),
    "5-14 aa": (len_s >= 5) & (len_s <= 14),
    "17 aa": len_s == 17,
    "all": np.ones(len_s.shape, dtype=bool),
}

sars_rng = np.random.default_rng(1)
sars_rows = []

for subset_name, mask in SUBSETS.items():
    indices = np.where(mask)[0]
    if len(indices) == 0:
        raise ValueError(f"SARS subset {subset_name} contains no rows.")
    if len(np.unique(y_s[indices])) != 2:
        raise ValueError(f"SARS subset {subset_name} must contain both target classes.")

    bootstrap_indices = []
    for _ in range(N_BOOT):
        sample = sars_rng.choice(indices, size=len(indices), replace=True)
        sample_y = y_s[sample]
        if 0 < sample_y.sum() < len(sample_y):
            bootstrap_indices.append(sample)

    if len(bootstrap_indices) < int(0.90 * N_BOOT):
        raise ValueError(f"SARS subset {subset_name} has too few valid bootstrap samples.")

    bootstrap_auc = {name: [] for name in sars_scores}
    for sample in bootstrap_indices:
        sample_y = y_s[sample]
        for name, scores in sars_scores.items():
            bootstrap_auc[name].append(fast_auc(sample_y, scores[sample]))

    reference = np.asarray(bootstrap_auc["emini_alone"], dtype=float)
    for name, scores in sars_scores.items():
        values = np.asarray(bootstrap_auc[name], dtype=float)
        low, high = np.percentile(values, [2.5, 97.5])
        row = {
            "subset": subset_name,
            "rows": int(mask.sum()),
            "positives": int(y_s[mask].sum()),
            "scorer": name,
            "auc": fast_auc(y_s[mask], scores[mask]),
            "low": low,
            "high": high,
            "valid_paired_bootstraps": len(values),
        }
        if name != "emini_alone":
            differences = values - reference
            diff_low, diff_high = np.percentile(differences, [2.5, 97.5])
            row.update({
                "difference_vs_emini": row["auc"] - fast_auc(
                    y_s[mask], sars_scores["emini_alone"][mask]
                ),
                "difference_low": diff_low,
                "difference_high": diff_high,
            })
        sars_rows.append(row)

sars_results = pd.DataFrame(sars_rows)
print(sars_results.round(4).to_string(index=False))
print("SARS bootstrap intervals use paired peptide resampling.")
print("SARS contains one protein with overlapping peptides. These intervals")
print("describe peptide-level resampling, not independent-protein uncertainty.")
print("The 15-aa peptide length contributes 4 positives to the 5-15 aa subset.")
print("It is excluded from the 5-14 aa subset.")
sars_results.round(4).to_csv(RESULTS_DIR / "11_sars.csv", index=False)

# ----------------------------------------------------------
# COVID ranking, native sequence ends before engineered tail
# ----------------------------------------------------------

banner("CHECK 7: COVID RANKING, END POSITION <= 1234")
covid_candidates = covid[
    (covid["peptide_length"] >= 5)
    & (covid["peptide_length"] <= 15)
    & (covid["end_position"] <= CONSTRUCT_END)
].copy()

if covid_candidates.empty:
    raise ValueError("No COVID candidates remain after length and position filters.")

print("COVID rows:", len(covid))
print("Candidates retained:", len(covid_candidates))
print("Largest candidate end position:", int(covid_candidates["end_position"].max()))

X_covid = covid_candidates[FEATS].to_numpy(dtype=float)
if not np.isfinite(X_covid).all():
    raise ValueError("COVID candidate feature matrix contains non-finite values.")

covid_scores = {
    name: score_of(final_models[name], X_covid)
    for name in MODEL_NAMES
}
for name in MODEL_NAMES:
    covid_candidates["score_" + name] = covid_scores[name]

covid_candidates["score"] = ensemble_scores(covid_scores)
covid_candidates["len_pct"] = covid_candidates.groupby(
    "peptide_length"
)["score"].rank(pct=True)

correlation_columns = ["score_" + name for name in MODEL_NAMES]
correlation_columns += ["score", "emini_log"]
print("\nSpearman correlations between model scores and Emini")
print(covid_candidates[correlation_columns].corr(method="spearman").round(3))

top_count = max(1, int(np.ceil(0.01 * len(covid_candidates))))
top_ensemble = set(covid_candidates.nlargest(top_count, "score").index)
top_emini = set(covid_candidates.nlargest(top_count, "emini_log").index)
overlap = len(top_ensemble & top_emini) / top_count
print("\nTop 1 percent:", top_count)
print("Ensemble/Emini top-1-percent overlap:", round(overlap, 4))

length_mix = pd.DataFrame({
    "share_all": covid_candidates["peptide_length"].value_counts(normalize=True),
    "share_top_1pct": covid_candidates.loc[
        list(top_ensemble), "peptide_length"
    ].value_counts(normalize=True),
}).fillna(0).sort_index()
print("\nLength mix, all candidates versus top 1 percent")
print(length_mix.round(3))

ranked = covid_candidates.sort_values(
    ["len_pct", "score"], ascending=False, kind="stable"
)
chosen = []
per_length = {}

for _, row in ranked.iterrows():
    peptide_length = int(row["peptide_length"])
    if per_length.get(peptide_length, 0) >= MAX_PER_LENGTH:
        continue
    non_overlapping = all(
        row["end_position"] < chosen_row["start_position"]
        or row["start_position"] > chosen_row["end_position"]
        for chosen_row in chosen
    )
    if non_overlapping:
        chosen.append(row)
        per_length[peptide_length] = per_length.get(peptide_length, 0) + 1
    if len(chosen) == 25:
        break

if len(chosen) != 25:
    raise ValueError(
        f"Only {len(chosen)} non-overlapping candidates met the selection rules. "
        "Review the candidate pool and per-length cap before reporting a top 25."
    )

output_columns = [
    "len_pct", "score", "peptide_seq", "start_position", "end_position",
    "peptide_length", "chou_fasman", "emini", "parker"
]
top = pd.DataFrame(chosen)[output_columns]
top.to_csv(RESULTS_DIR / "COVID_top_final_v2.csv", index=False)

ranked_output_columns = output_columns + ["score_" + name for name in MODEL_NAMES]
covid_candidates.sort_values("score", ascending=False, kind="stable")[
    ranked_output_columns
].to_csv(RESULTS_DIR / "COVID_ranked_final_v2.csv", index=False)

print("\nTop 25 non-overlapping candidates, maximum 3 per peptide length")
print(top.round(4).to_string(index=False))
print("COVID rankings are computational prioritizations, not validated epitopes.")
print("COVID has no labels in this dataset, so predictive accuracy is not calculated.")

# ----------------------------------------------------------
# Figures 21 and 22
# ----------------------------------------------------------

banner("FIGURES 21 AND 22")
fig, ax = plt.subplots(figsize=(11, 5.5))
jitter_rng = np.random.default_rng(3)

for position, name in enumerate(CMP_NAMES):
    seed_means = PP[name].mean(axis=1)
    ax.plot(
        position + jitter_rng.uniform(-0.12, 0.12, len(seed_means)),
        seed_means, "o", color="gray", alpha=0.7
    )
    low, high = boot_ci(PPM[name])
    ax.errorbar(
        position, seed_means.mean(),
        yerr=[[seed_means.mean() - low], [high - seed_means.mean()]],
        fmt="s", color="black", capsize=6, markersize=8
    )

ax.axhline(0.5, color="gray", linestyle="--", label="chance")
ax.set_xticks(range(len(CMP_NAMES)))
ax.set_xticklabels(CMP_NAMES, rotation=20)
ax.set_ylim(0.45, 0.70)
ax.set_ylabel("Macro within-protein AUC")
ax.set_title("Figure 21: Five Seeds, Mean and 95% Protein Bootstrap")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "Fig21_Five_Seeds.png", dpi=300, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(length_mix))
ax.bar(
    x - 0.2, length_mix["share_all"], width=0.4,
    label="all candidates", edgecolor="black"
)
ax.bar(
    x + 0.2, length_mix["share_top_1pct"], width=0.4,
    label="top 1 percent", edgecolor="black"
)
ax.set_xticks(x)
ax.set_xticklabels(length_mix.index)
ax.set_xlabel("Peptide length (aa)")
ax.set_ylabel("Share of peptides")
ax.set_title("Figure 22: COVID Length Mix, Ensemble Ranking")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "Fig22_COVID_Length_Mix.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("Saved Figures 21 and 22.")
print("\n" + "=" * 70)
print("FILE 11 COMPLETED SUCCESSFULLY")
print("=" * 70)
