# ==========================================================
# File: 02_ml_data_checks.py
# Purpose: Check label conflicts, length leakage, and group
# sizes before building the ML split.
# Input: Data/B_Cells_cleaned.csv, Data/Sars_cleaned.csv
# ==========================================================

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

b = pd.read_csv(BASE_DIR / "Data" / "B_Cells_cleaned.csv")
s = pd.read_csv(BASE_DIR / "Data" / "Sars_cleaned.csv")

b["len"] = b["end_position"] - b["start_position"] + 1
s["len"] = s["end_position"] - s["start_position"] + 1

print("=" * 70)
print("CHECK 1: Label conflicts in B_Cells")
print("=" * 70)
print("Conflicting peptides:",
      b.groupby("peptide_seq")["target"].nunique().gt(1).sum())
print("Rows with repeated peptides:",
      b["peptide_seq"].duplicated(keep=False).sum())

print("\n" + "=" * 70)
print("CHECK 2: Peptide length by target (B_Cells)")
print("=" * 70)
print(b.groupby("target")["len"].describe())

print("\n" + "=" * 70)
print("CHECK 3: Peptide length by target (SARS)")
print("=" * 70)
print(s.groupby("target")["len"].describe())

print("\n" + "=" * 70)
print("CHECK 4: Peptides per protein sequence (B_Cells)")
print("=" * 70)
g = b.groupby("protein_seq").size()
print(g.describe())
print("\nLargest 5 protein groups:")
print(g.sort_values(ascending=False).head(5))

print("\n" + "=" * 70)
print("CHECK 5: Positive rate per protein (B_Cells)")
print("=" * 70)
print(b.groupby("protein_seq")["target"].mean().describe())