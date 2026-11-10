# ==========================================================
# File: 08_check_tail.py
# Purpose: Verify COVID protein sequence tail and construct motifs
# ==========================================================

import pandas as pd
from pathlib import Path

# ----------------------------------------------------------
# 1. Load and validate input
# ----------------------------------------------------------

data_path = Path(__file__).resolve().parent / "Data" / "COVID_cleaned.csv"

if not data_path.exists():
    raise FileNotFoundError(f"Input file not found: {data_path}")

covid = pd.read_csv(data_path)

if covid.empty:
    raise ValueError("COVID_cleaned.csv contains no rows.")

if "protein_seq" not in covid.columns:
    raise ValueError("Required column 'protein_seq' is missing.")

sequences = covid["protein_seq"].dropna().astype(str).str.strip().str.upper()

if sequences.empty:
    raise ValueError("No valid protein sequences were found.")

if (sequences == "").any():
    raise ValueError("Empty protein sequences were found.")

unique_sequences = sequences.unique()

if len(unique_sequences) != 1:
    raise ValueError(f"Expected one unique protein sequence, found {len(unique_sequences)}.")

seq = unique_sequences[0]

# ----------------------------------------------------------
# 2. Report sequence information
# ----------------------------------------------------------

print("=" * 60)
print("COVID PROTEIN SEQUENCE CHECK")
print("=" * 60)

print(f"Total dataset rows: {len(covid)}")
print(f"Unique protein sequences: {len(unique_sequences)}")
print(f"Full sequence length: {len(seq)} amino acids")

print("\nLast 100 amino acids:")
print(seq[-100:])

print("\nLast 50 amino acids:")
print(seq[-50:])

# ----------------------------------------------------------
# 3. Locate construct motifs
# ----------------------------------------------------------

motifs = {"Foldon motif": "GYIPEAPRDGQAYVRKDGEWVLLSTFL",
          "Linker motif": "GGGGS"}

print("\nMotif positions (1-based):")

for name, motif in motifs.items():
    positions = []
    start = 0

    while True:
        index = seq.find(motif, start)

        if index == -1:
            break

        positions.append(index + 1)
        start = index + 1

    if positions:
        print(f"{name}: {positions}")
    else:
        print(f"{name}: NOT FOUND")

# ----------------------------------------------------------
# 4. Check terminal histidine tag
# ----------------------------------------------------------

terminal_his = len(seq) - len(seq.rstrip("H"))

print("\nTerminal sequence check:")
print(f"Terminal consecutive histidines: {terminal_his}")
print(f"Sequence ends with HHHHHHHH: {seq.endswith('HHHHHHHH')}")
print("\nSequence inspection completed.")