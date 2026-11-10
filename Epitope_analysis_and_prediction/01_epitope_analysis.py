# ==========================================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# ==========================================================

from pathlib import Path
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D


# ==========================================================
# Step 2: Download Dataset (Run Only Once)
# ==========================================================
# 
# path = kagglehub.dataset_download(
#     "futurecorporation/epitope-prediction"
# )
# print("Dataset Path:", path)


# ==========================================================
# Step 3: Directory for saving files
# ==========================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = BASE_DIR / "Results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# Step 4: Load the Datasets
# ==========================================================
bcell = pd.read_csv(DATA_DIR / "input_bcell.csv")
covid = pd.read_csv(DATA_DIR / "input_covid.csv")
sars = pd.read_csv(DATA_DIR / "input_sars.csv")


# Store datasets inside a dictionary
# to avoid repetitive code.

datasets = {
    "B_Cells": bcell,
    "COVID": covid,
    "Sars": sars
}


# ==========================================================
# Step 5: Dataset Exploration
# ==========================================================

print("\n")
print("=" * 80)
print("STEP 4 : DATASET EXPLORATION")
print("=" * 80)

for dataset_name, df in datasets.items():

    print("\n")
    print("=" * 80)
    print(f"Dataset : {dataset_name}")
    print("=" * 80)

    # ------------------------------------------------------
    # Basic Dataset Information
    # ------------------------------------------------------

    print("\n----- Basic Information -----")

    print("Dataset Type :", type(df))
    print("Shape        :", df.shape)
    print("Columns      :", list(df.columns))

    print("\nFirst 10 Rows")
    print(df.head(10))

    print("\n----- Dataset Structure -----")
    df.info()

    print("\n----- Descriptive Statistics -----")
    print(df.describe())


    # ======================================================
    # Step 6 : Data Quality Assessment & Cleaning
    # ======================================================

    print("\n")
    print("-" * 70)
    print("STEP 5 : DATA QUALITY ASSESSMENT & CLEANING")
    print("-" * 70)


    # ------------------------------------------------------
    # Missing Values
    # ------------------------------------------------------

    missing_values = df.isna().sum()

    print("\nMissing Values")
    print(missing_values)


    # ------------------------------------------------------
    # Duplicate Rows
    # ------------------------------------------------------

    duplicate_rows = df.duplicated().sum()

    print("\nDuplicate Rows :", duplicate_rows)
    

    # ------------------------------------------------------
    # Dataset Diversity
    # ------------------------------------------------------

    unique_parent_proteins = df["parent_protein_id"].nunique()
    unique_proteins = df["protein_seq"].nunique()
    unique_peptides = df["peptide_seq"].nunique()

    print("\nDataset Diversity")
    print("Unique Parent Proteins :", unique_parent_proteins)
    print("Unique Protein Sequences :", unique_proteins)
    print("Unique Peptide Sequences :", unique_peptides)


    # ------------------------------------------------------
    # Sequence cleaning is performed before duplicate removal.
    print("\\nSequence Cleaning")

    protein_spaces = df["protein_seq"].astype("string").str.contains(
        r"\s", regex=True, na=False
    ).sum()
    peptide_spaces = df["peptide_seq"].astype("string").str.contains(
        r"\s", regex=True, na=False
    ).sum()

    print("Protein Sequences Containing Whitespace :", int(protein_spaces))
    print("Peptide Sequences Containing Whitespace :", int(peptide_spaces))

    df["protein_seq"] = (
        df["protein_seq"]
        .astype("string")
        .str.replace(r"\s+", "", regex=True)
        .str.upper()
    )
    df["peptide_seq"] = (
        df["peptide_seq"]
        .astype("string")
        .str.replace(r"\s+", "", regex=True)
        .str.upper()
    )

    # Ensure coordinate columns are numeric. Unparseable values become missing
    # and are reported in the validation summary below.
    for column in ["start_position", "end_position"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Remove exact duplicate rows after sequence normalization.
    rows_before_deduplication = len(df)
    df.drop_duplicates(inplace=True)
    removed_duplicates = rows_before_deduplication - len(df)
    print("Duplicate rows removed after normalization :", removed_duplicates)

    # Coordinates are expected to be 1-based and inclusive.
    print("\\nBiological Validation")

    invalid_order = (
        df["start_position"].notna()
        & df["end_position"].notna()
        & (df["end_position"] < df["start_position"])
    ).sum()
    invalid_start = (
        df["start_position"].notna() & (df["start_position"] < 1)
    ).sum()
    invalid_end = (
        df["end_position"].notna() & (df["end_position"] < 1)
    ).sum()
    missing_coordinates = df[
        ["start_position", "end_position"]
    ].isna().any(axis=1).sum()

    non_integer_coordinates = (
        df["start_position"].notna()
        & df["end_position"].notna()
        & (
            (df["start_position"] % 1 != 0)
            | (df["end_position"] % 1 != 0)
        )
    ).sum()

    print("End Position < Start Position :", int(invalid_order))
    print("Start Position < 1            :", int(invalid_start))
    print("End Position < 1              :", int(invalid_end))
    print("Rows with missing coordinates :", int(missing_coordinates))
    print("Rows with non-integer coordinates:", int(non_integer_coordinates))

    valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    invalid_peptide_symbols = df["peptide_seq"].dropna().apply(
        lambda seq: sorted(set(seq) - valid_amino_acids)
    )
    invalid_protein_symbols = df["protein_seq"].dropna().apply(
        lambda seq: sorted(set(seq) - valid_amino_acids)
    )

    print(
        "Peptide sequences with non-standard symbols :",
        int((invalid_peptide_symbols.str.len() > 0).sum())
    )
    print(
        "Protein sequences with non-standard symbols :",
        int((invalid_protein_symbols.str.len() > 0).sum())
    )


    # Target Distribution
    # ------------------------------------------------------

    print("\nTarget Distribution")

    if "target" in df.columns:

        print(df["target"].value_counts())

    else:

        print(f"{dataset_name} does not contain a target column.")


    # ------------------------------------------------------
    # Constant Feature Detection
    # ------------------------------------------------------

    print("\nConstant Features")

    for column in df.columns:

        if df[column].nunique() == 1:

            print(f"{column} is a constant feature.")


    # ------------------------------------------------------
    # Save Clean Dataset
    # ------------------------------------------------------

    cleaned_file = f"{dataset_name}_cleaned.csv"

    df.to_csv(DATA_DIR / cleaned_file, index=False)

    print(f"\n✓ Cleaned dataset saved as: {cleaned_file}")


print("\n")
print("=" * 80)
print("All datasets have been explored, cleaned, validated, and saved.")
print("=" * 80)


# ==========================================================
# Step 7 : Data Transformation
# ==========================================================

print("\n")
print("=" * 80)
print("STEP 6 : DATA TRANSFORMATION")
print("=" * 80)

for dataset_name, df in datasets.items():

    # ------------------------------------------------------
    # Rename Columns
    # ------------------------------------------------------

    df.rename(
        columns={"parent_protein_id": "protein_id"},
        inplace=True
    )


    # ------------------------------------------------------
    # Create Biological Feature
    # ------------------------------------------------------

    # Coordinates are 1-based and inclusive.
    df["peptide_length"] = (
        df["end_position"] - df["start_position"] + 1
    )


    # ------------------------------------------------------
    # Add Dataset Identifier
    # ------------------------------------------------------

    df["dataset"] = dataset_name


print("\nData transformation completed successfully.")


# ==========================================================
# Merge All Datasets
# ==========================================================

merged_file = pd.concat(
    datasets.values(),
    ignore_index=True
)

merged_file.to_csv(RESULTS_DIR / "merged_file.csv", index=False)

print("\nMerged Dataset Summary")
print("Shape :", merged_file.shape)

print("\nColumns")
print(list(merged_file.columns))

print("\nFirst 5 Rows")
print(merged_file.head())

print("\nDataset Structure")
merged_file.info()


# ==========================================================
# Step 8 : Descriptive Statistics & Grouping
# ==========================================================

print("\n")
print("=" * 80)
print("STEP 7 : DESCRIPTIVE STATISTICS & GROUPING")
print("=" * 80)


# ----------------------------------------------------------
# Dataset-wise Descriptive Statistics
# ----------------------------------------------------------

dataset_statistics = merged_file.groupby("dataset").describe()

print("\nDataset-wise Descriptive Statistics")
print(dataset_statistics)


# ----------------------------------------------------------
# Unique Values per Dataset
# ----------------------------------------------------------

unique_feature_counts = merged_file.groupby("dataset").nunique()

print("\nUnique Values per Dataset")
print(unique_feature_counts)


# ----------------------------------------------------------
# Number of Records per Dataset
# ----------------------------------------------------------

print("\nNumber of Peptides in Each Dataset")
print(merged_file["dataset"].value_counts())


# ----------------------------------------------------------
# Average Peptide Length
# ----------------------------------------------------------

mean_peptide_length = (
    merged_file
    .groupby("dataset")["peptide_length"]
    .mean()
)

print("\nAverage Peptide Length")
print(mean_peptide_length)


# ----------------------------------------------------------
# Biochemical Feature Summary
# ----------------------------------------------------------

print("\nBiochemical Feature Summary")

print(
    merged_file[
        [
            "chou_fasman",
            "emini",
            "kolaskar_tongaonkar",
            "parker",
            "isoelectric_point",
            "aromaticity",
            "hydrophobicity",
            "stability",
        ]
    ].agg(["mean", "min", "max"])
)


# ----------------------------------------------------------
# Position Summary
# ----------------------------------------------------------

print("\nPosition Summary")

print(
    merged_file[
        [
            "start_position",
            "end_position",
        ]
    ].agg(["mean", "min", "max"])
)


# ----------------------------------------------------------
# Pivot Table
# ----------------------------------------------------------

mean_length_table = merged_file.pivot_table(
    values="peptide_length",
    index="dataset",
    aggfunc="mean",
)

print("\nPivot Table: Mean Peptide Length")
print(mean_length_table)


print("\n")
print("=" * 80)
print("STEP 7 COMPLETED SUCCESSFULLY")
print("=" * 80)


# ==========================================================
# STEP 9: DATA VISUALIZATION
# ==========================================================
# Purpose:
# Explore peptide characteristics using different visualization
# techniques to better understand the biological properties of
# the merged B-Cell, COVID, and SARS datasets.
# #
# Visualization Types:
#   1. Histogram
#   2. Box Plot
#   3. Bar Plot
#   4. Scatter Plot
# ==========================================================

# Directory for saving images
FIG_DIR = BASE_DIR / "Figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

print(repr(str(FIG_DIR)))
print(FIG_DIR.resolve())
print(FIG_DIR.exists())

DATASET_ORDER = ["B_Cells", "COVID", "Sars"]
DATASET_COLORS = {"B_Cells": "#4C72B0", "COVID": "#55A868", "Sars": "#C44E52"}
TARGET_COLORS = {0: "#4C72B0", 1: "#C44E52"}
TARGET_NAMES = {0: "Non-epitope", 1: "Epitope"}

PEPTIDE_FEATURES = [
    "chou_fasman", "emini", "kolaskar_tongaonkar", "parker", "peptide_length"
]
PROTEIN_FEATURES = [
    "isoelectric_point", "aromaticity", "hydrophobicity", "stability"
]
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# Labeled rows only (B_Cells and Sars). COVID has no target.
labeled = merged_file[merged_file["target"].notna()].copy()
labeled["target"] = labeled["target"].astype(int)
labeled_short = labeled[labeled["peptide_length"] <= 25]


def save_figure(fig, name, rect=None):
    fig.tight_layout(rect=rect)
    fig.savefig(FIG_DIR / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", name)


# ----------------------------------------------------------
# Validation: sequence length must match coordinate length
# ----------------------------------------------------------

print("\n" + "=" * 70)
print("VALIDATION: PEPTIDE LENGTH")
print("=" * 70)

seq_length = merged_file["peptide_seq"].str.len()
mismatch = (seq_length != merged_file["peptide_length"]).sum()
print("Rows where sequence length differs from coordinate length:", int(mismatch))


def extract_parent_segment(row):
    """Extract a 1-based inclusive peptide region from its parent protein."""
    protein = row["protein_seq"]
    start = row["start_position"]
    end = row["end_position"]

    if pd.isna(protein) or pd.isna(start) or pd.isna(end):
        return None
    if start < 1 or end < start or end > len(protein):
        return None

    return protein[int(start) - 1:int(end)]


valid_coordinate_mask = (
    merged_file["start_position"].notna()
    & merged_file["end_position"].notna()
    & (merged_file["start_position"] >= 1)
    & (merged_file["end_position"] >= merged_file["start_position"])
    & (merged_file["start_position"] % 1 == 0)
    & (merged_file["end_position"] % 1 == 0)
    & merged_file["protein_seq"].notna()
    & merged_file["peptide_seq"].notna()
    & (merged_file["end_position"] <= merged_file["protein_seq"].str.len())
)

sequence_check = merged_file.loc[valid_coordinate_mask].copy()
sequence_check["extracted_parent_segment"] = sequence_check.apply(
    extract_parent_segment, axis=1
)
sequence_match = (
    sequence_check["peptide_seq"]
    == sequence_check["extracted_parent_segment"]
)

print(
    "Rows with valid coordinates checked against parent protein:",
    int(valid_coordinate_mask.sum())
)
print(
    "Peptide sequences matching parent-protein segments:",
    int(sequence_match.sum())
)
print(
    "Peptide sequences NOT matching parent-protein segments:",
    int((~sequence_match).sum())
)
print(
    "Rows excluded from segment comparison due to invalid coordinates:",
    int((~valid_coordinate_mask).sum())
)

if (~sequence_match).any():
    print("\nExamples of peptide-to-protein sequence mismatches:")
    print(
        sequence_check.loc[
            ~sequence_match,
            [
                "dataset", "protein_id", "start_position", "end_position",
                "peptide_seq", "extracted_parent_segment"
            ]
        ].head(10).to_string(index=False)
    )

valid_lengths = merged_file["peptide_length"].dropna()
if not valid_lengths.empty:
    longest_index = valid_lengths.idxmax()
    longest = merged_file.loc[
        longest_index,
        ["protein_id", "start_position", "end_position", "dataset", "peptide_length"],
    ]
    print("\nLongest peptide record:")
    print(longest)
else:
    print("\nNo valid peptide lengths are available.")


# ==========================================================
# FIGURE 1: Distributions
# Row 1: peptide length, class balance
# Row 2: protein-level features (one value per protein)
# ==========================================================

print("\nFigure 1: Distributions")

fig, axes = plt.subplots(2, 4, figsize=(20, 9))

# 1a. Peptide length histogram by dataset
ax = axes[0, 0]
bins = np.arange(4.5, 26.5, 1)
for ds in DATASET_ORDER:
    values = merged_file.loc[merged_file["dataset"] == ds, "peptide_length"]
    ax.hist(
        values[values <= 25], bins=bins, density=True, alpha=0.5,
        color=DATASET_COLORS[ds], label=ds, edgecolor="black"
    )
ax.set_title("Peptide Length (up to 25 aa)")
ax.set_xlabel("Peptide length (aa)")
ax.set_xticks(range(5, 26, 2))
ax.set_ylabel("Density")
ax.legend()

# 1b. Peptide length boxplot, log scale keeps the 393 aa peptide visible
ax = axes[0, 1]
length_data = [
    merged_file.loc[merged_file["dataset"] == ds, "peptide_length"]
    for ds in DATASET_ORDER
]
ax.boxplot(length_data)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(DATASET_ORDER)
ax.set_yscale("log")
ax.set_title("Peptide Length by Dataset (log scale)")
ax.set_ylabel("Peptide length (aa)")

# 1c. Number of peptides per dataset
ax = axes[0, 2]
counts = merged_file["dataset"].value_counts().reindex(DATASET_ORDER)
bars = ax.bar(
    counts.index, counts.values,
    color=[DATASET_COLORS[d] for d in counts.index], edgecolor="black"
)
ax.bar_label(bars)
ax.set_ylim(0, counts.max() * 1.12)
ax.set_title("Number of Peptides")
ax.set_ylabel("Peptides")

# 1d. Class balance in labeled data
ax = axes[0, 3]
balance = labeled.groupby(["dataset", "target"]).size().unstack(fill_value=0)
share = balance.div(balance.sum(axis=1), axis=0) * 100
ax.bar(share.index, share[0], color=TARGET_COLORS[0],
       label=TARGET_NAMES[0], edgecolor="black")
ax.bar(share.index, share[1], bottom=share[0], color=TARGET_COLORS[1],
       label=TARGET_NAMES[1], edgecolor="black")
ax.set_title("Class Balance (labeled data)")
ax.set_ylabel("Share of peptides (%)")
ax.set_ylim(0, 125)
ax.legend(loc="upper center", ncol=2)

# 2. Protein-level features, one value per protein
protein_level = merged_file[merged_file["dataset"] == "B_Cells"].drop_duplicates(
    "protein_seq"
)
single_proteins = {
    ds: merged_file[merged_file["dataset"] == ds].iloc[0]
    for ds in ["COVID", "Sars"]
}

for ax, feat in zip(axes[1], PROTEIN_FEATURES):
    ax.hist(
        protein_level[feat], bins=30, color=DATASET_COLORS["B_Cells"],
        edgecolor="black", label=f"B_Cells proteins (n={len(protein_level)})"
    )
    for ds, row in single_proteins.items():
        ax.axvline(
            row[feat], color=DATASET_COLORS[ds], linestyle="--",
            linewidth=2, label=f"{ds} protein"
        )
    ax.set_title(f"{feat} (protein level)")
    ax.set_xlabel(feat)
    ax.set_ylabel("Number of proteins")
    ax.legend(fontsize=8)

fig.suptitle("Figure 1: Length, Class Balance, and Protein-Level Features",
             fontsize=16)
save_figure(fig, "Fig1_Distributions.png", rect=[0, 0, 1, 0.96])


# ==========================================================
# FIGURE 2: Peptide-level features by dataset and target
# ==========================================================

print("\nFigure 2: Peptide-level features by target")

groups = []
for ds in ["B_Cells", "Sars"]:
    for t in [0, 1]:
        subset = labeled[(labeled["dataset"] == ds) & (labeled["target"] == t)]
        groups.append((f"{ds}\n{TARGET_NAMES[t]}", subset, TARGET_COLORS[t]))
groups.append((
    "COVID\nunlabeled",
    merged_file[merged_file["dataset"] == "COVID"],
    "#999999",
))

fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))

for ax, feat in zip(
    axes, ["chou_fasman", "emini", "kolaskar_tongaonkar", "parker"]
):
    bp = ax.boxplot(
        [g[1][feat] for g in groups], patch_artist=True, showfliers=False
    )
    for patch, g in zip(bp["boxes"], groups):
        patch.set_facecolor(g[2])
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels([g[0] for g in groups], fontsize=8)
    ax.set_title(f"{feat} (peptide level)")
    ax.set_ylabel(feat)

fig.suptitle("Figure 2: Peptide-Level Features by Dataset and Target",
             fontsize=16)
save_figure(fig, "Fig2_Peptide_Features_By_Target.png", rect=[0, 0, 1, 0.95])


# ==========================================================
# FIGURE 3: Amino acid composition by target
# Labeled peptides up to 25 aa, so long SARS peptides do not skew counts
# ==========================================================

print("\nFigure 3: Amino acid composition by target")


def composition(sequences):
    counts = Counter("".join(sequences.dropna().astype(str)))
    total = sum(counts[aa] for aa in AMINO_ACIDS)

    if total == 0:
        raise ValueError(
            "No standard amino-acid residues were found for one target group."
        )

    return np.array(
        [counts[aa] / total * 100 for aa in AMINO_ACIDS],
        dtype=float
    )


def smoothed_composition(sequences, pseudocount=0.5):
    counts = Counter("".join(sequences.dropna().astype(str)))
    values = np.array([counts[aa] for aa in AMINO_ACIDS], dtype=float)
    values += pseudocount
    return values / values.sum() * 100


negative_sequences = labeled_short.loc[
    labeled_short["target"] == 0, "peptide_seq"
]
positive_sequences = labeled_short.loc[
    labeled_short["target"] == 1, "peptide_seq"
]

comp_neg = composition(negative_sequences)
comp_pos = composition(positive_sequences)

# Smoothed frequencies keep log2 enrichment finite when an amino acid
# is absent from one of the target groups.
comp_neg_smooth = smoothed_composition(negative_sequences)
comp_pos_smooth = smoothed_composition(positive_sequences)
enrichment = np.log2(comp_pos_smooth / comp_neg_smooth)



x = np.arange(len(AMINO_ACIDS))
fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

axes[0].bar(x - 0.2, comp_neg, width=0.4, color=TARGET_COLORS[0],
            label=TARGET_NAMES[0], edgecolor="black")
axes[0].bar(x + 0.2, comp_pos, width=0.4, color=TARGET_COLORS[1],
            label=TARGET_NAMES[1], edgecolor="black")
axes[0].set_xticks(x)
axes[0].set_xticklabels(AMINO_ACIDS)
axes[0].set_title("Amino Acid Frequency")
axes[0].set_xlabel("Amino acid")
axes[0].set_ylabel("Frequency (%)")
axes[0].legend()

axes[1].bar(
    x, enrichment,
    color=[TARGET_COLORS[1] if v > 0 else TARGET_COLORS[0] for v in enrichment],
    edgecolor="black",
)
axes[1].axhline(0, color="black", linewidth=1)
axes[1].set_xticks(x)
axes[1].set_xticklabels(AMINO_ACIDS)
axes[1].set_title("Enrichment in Epitopes (log2 epitope / non-epitope)")
axes[1].set_xlabel("Amino acid")
axes[1].set_ylabel("log2 enrichment")

fig.suptitle("Figure 3: Amino Acid Composition by Target", fontsize=16)
save_figure(fig, "Fig3_Amino_Acid_Composition.png", rect=[0, 0, 1, 0.95])


# ==========================================================
# FIGURES 4 to 6: Multivariate analysis of peptide-level features
# Fig 4: Spearman correlation heatmaps
# Fig 5: Distributions by target (one figure, five axes)
# Fig 6: Pairwise scatter by target (one figure, ten axes)
# Needs: labeled_short, PEPTIDE_FEATURES, TARGET_COLORS, TARGET_NAMES,
#        FIG_DIR, save_figure, np, plt from earlier in Step 9
# ==========================================================

from itertools import combinations
from matplotlib.lines import Line2D

feats = PEPTIDE_FEATURES
n = len(feats)


# ----------------------------------------------------------
# Correlation method check: Spearman versus Pearson
# ----------------------------------------------------------

print("\n" + "=" * 70)
print("CORRELATION METHOD CHECK")
print("=" * 70)

spearman = labeled_short[feats].corr(method="spearman")
pearson = labeled_short[feats].corr(method="pearson")

print("\nSpearman:")
print(spearman.round(3))
print("\nPearson:")
print(pearson.round(3))
print("\nLargest absolute gap between the two:",
      round((spearman - pearson).abs().to_numpy().max(), 3))
print("Skewness of emini:", round(labeled_short["emini"].skew(), 2))


# ----------------------------------------------------------
# Data prepared for plotting only
# emini is shown as log1p(emini). Spearman is not affected.
# ----------------------------------------------------------

viz = labeled_short.copy()
viz["emini"] = np.log1p(viz["emini"])


def axis_label(feat):
    return "log1p(emini)" if feat == "emini" else feat


LIMITS = {
    f: (viz[f].quantile(0.005), viz[f].quantile(0.995)) for f in feats
}
LIMITS["peptide_length"] = (4.5, 25.5)


# ==========================================================
# FIGURE 4: Spearman correlation heatmaps
# ==========================================================

print("\nFigure 4: Correlation heatmaps")

corr_sets = [
    ("B_Cells", labeled_short[labeled_short["dataset"] == "B_Cells"]),
    ("Sars 5 to 15 aa", labeled_short[
        (labeled_short["dataset"] == "Sars")
        & (labeled_short["peptide_length"] <= 15)
    ]),
    ("Sars up to 25 aa", labeled_short[labeled_short["dataset"] == "Sars"]),
]

fig, axes = plt.subplots(
    1, 4, figsize=(21, 6),
    gridspec_kw={"width_ratios": [1, 1, 1, 0.05]},
)

for ax, (title, df) in zip(axes[:3], corr_sets):
    c = df[feats].corr(method="spearman")
    print(f"\nSpearman, {title} (n={len(df)}):")
    print(c.round(3))

    im = ax.imshow(c.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(feats, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(feats)

    for a in range(n):
        for b in range(n):
            value = c.values[a, b]
            ax.text(
                b, a, f"{value:.2f}", ha="center", va="center",
                fontsize=11,
                color="white" if abs(value) > 0.6 else "black",
            )
    ax.set_title(f"{title} (n={len(df)})")

fig.colorbar(im, cax=axes[3], label="Spearman correlation")
fig.suptitle("Figure 4: Spearman Correlation of Peptide-Level Features",
             fontsize=16)
save_figure(fig, "Fig4_Correlation_Heatmaps.png", rect=[0, 0, 1, 0.95])


# ==========================================================
# FIGURE 5: Distributions by target (five axes, one figure)
# ==========================================================

print("\nFigure 5: Distributions by target")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
flat_axes = axes.ravel()

for ax, feat in zip(flat_axes, feats):
    if feat == "peptide_length":
        bins = np.arange(4.5, 26.5, 1)
    else:
        bins = np.linspace(LIMITS[feat][0], LIMITS[feat][1], 31)

    for t in [0, 1]:
        values = viz.loc[viz["target"] == t, feat]
        ax.hist(
            values, bins=bins, density=True, alpha=0.5,
            color=TARGET_COLORS[t], label=TARGET_NAMES[t],
            edgecolor="black", linewidth=0.3,
        )
    ax.set_title(axis_label(feat))
    ax.set_xlabel(axis_label(feat))
    ax.set_ylabel("Density")
    if feat == "peptide_length":
        ax.set_xticks(range(5, 26, 2))

handles, labels = flat_axes[0].get_legend_handles_labels()
flat_axes[5].axis("off")
flat_axes[5].legend(handles, labels, loc="center", fontsize=14)

fig.suptitle("Figure 5: Peptide-Level Feature Distributions by Target",
             fontsize=16)
save_figure(fig, "Fig5_Distributions_By_Target.png", rect=[0, 0, 1, 0.95])


# ==========================================================
# FIGURE 6: Pairwise scatter by target (ten axes, one figure)
# Length receives a small jitter so the points do not form strips
# ==========================================================

print("\nFigure 6: Pairwise scatter")

sample = viz.sample(n=min(3000, len(viz)), random_state=42).copy()
rng = np.random.default_rng(0)
sample["peptide_length"] = sample["peptide_length"] + rng.uniform(
    -0.3, 0.3, len(sample)
)

sample = sample.sample(frac=1, random_state=1).reset_index(drop=True)
pairs = list(combinations(feats, 2))

fig, axes = plt.subplots(2, 5, figsize=(24, 10))

for ax, (feat_x, feat_y) in zip(axes.ravel(), pairs):
    for t in [0, 1]:
        part = sample[sample["target"] == t]
        ax.scatter(
            part[feat_x],
            part[feat_y],
            s=6,
            alpha=0.3,
            color=TARGET_COLORS[t],
            label=TARGET_NAMES[t]
        )

    ax.set_xlim(LIMITS[feat_x])
    ax.set_ylim(LIMITS[feat_y])
    ax.set_xlabel(axis_label(feat_x))
    ax.set_ylabel(axis_label(feat_y))
    r = spearman.loc[feat_x, feat_y]
    ax.set_title(f"Spearman = {r:.2f}")

legend_handles = [
    Line2D([0], [0], marker="o", linestyle="", markersize=8,
           color=TARGET_COLORS[t], label=TARGET_NAMES[t])
    for t in [0, 1]
]
fig.legend(handles=legend_handles, loc="upper right", fontsize=12)

fig.suptitle("Figure 6: Pairwise Scatter of Peptide-Level Features",
             fontsize=16)
save_figure(fig, "Fig6_Pairwise_Scatter.png", rect=[0, 0, 1, 0.95])

print("\n" + "=" * 70)
print("STEP 9 COMPLETED: figures saved in", FIG_DIR)
print("=" * 70)