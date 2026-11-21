"""Create the fixed train/val/test split and the nested 1%/10%/100% labelled subsets.

Usage:
    python scripts/make_splits.py            # writes splits/ (refuses to overwrite)
    python scripts/make_splits.py --check    # verify committed splits against manifest + data

The split files in splits/ are committed to git. Every group member and every
Colab/Kaggle run must use those files instead of regenerating them.

What it builds (details in docs/DATA.md and eurosat_le/data/splits.py):
    - splits/split.csv: stratified 70/15/15 train/val/test split with SPLIT_SEED = 42, so the
      evaluation sets are identical for every model, input type, label fraction and seed.
    - splits/subsets/*.txt: labelled subsets (1%, 10%, 100% of train) for each seed in SUBSET_SEEDS.
      They are drawn from train only and are nested (1% inside 10% inside 100%), so adding labels never
      removes examples already in use; the unlabelled pool for semi-supervised learning is "train minus
      subset".
    - splits/manifest.json: SHA-256 of every split file plus class counts.

Leakage safeguards: splits.verify() runs on every generation and every --check (no overlap between
splits, subsets inside train, nesting, all classes in the smallest subset), and the manifest hashes
detect any later edit. The RGB and 13-band folders are also checked to contain the same ids, so one
split serves both input types.

Regenerating the split changes which images are in train/val/test and therefore every result, which
is why overwriting needs --force.
"""

# argparse reads the command-line options; json reads and writes the manifest file.
import argparse
import json

# Shared settings: dataset description, raw-data folder, the fixed split seed, the splits folder and subset seeds.
from eurosat_le.constants import DATASETS, RAW_DIR, SPLIT_SEED, SPLITS_DIR, SUBSET_SEEDS
# The module with the split logic (build, save, load, verify).
from eurosat_le.data import splits
# build_index lists every image with its class; check_modalities_match compares the RGB and MS folders.
from eurosat_le.data.index import build_index, check_modalities_match


def main() -> None:
    """Create the train/val/test split and labelled subsets, or (with --check) verify the existing ones.

    Takes no arguments (options come from sys.argv) and returns nothing. Writes into splits/ unless
    --check is given. Stops with an error message if split files exist and --force was not given.
    """
    # Build the argument parser; the module docstring doubles as the --help text.
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Flag: only verify the committed split files, write nothing.
    p.add_argument("--check", action="store_true", help="Only verify existing split files.")
    # Flag: allow overwriting an existing split (this changes every result, so it must be explicit).
    p.add_argument("--force", action="store_true", help="Overwrite existing split files.")
    # Parse sys.argv into the args object.
    args = p.parse_args()

    # The index (and so the split) is built from the RGB ids; the MS folder is optional here and, when
    # present, is only used to check that both versions contain exactly the same ids.
    # Is the 13-band dataset folder present on disk?
    have_ms = (RAW_DIR / DATASETS["ms"]["folder"]).is_dir()
    # List every RGB image as a table with its id, class name and class number.
    index = build_index(RAW_DIR, "rgb")
    # If both versions exist, confirm they hold exactly the same sample ids (so one split fits both).
    if have_ms:
        check_modalities_match(RAW_DIR)
        print("RGB and MS folders contain identical sample ids.")
    else:
        print("MS folder not found; split built from RGB ids (identical by construction of EuroSAT).")

    # --check mode: verify the stored split and stop; nothing is generated or overwritten.
    if args.check:
        # Verify, do not write: hashes vs manifest, ids vs data on disk, then the full leakage checks.
        # Compare every split file with the SHA-256 hash recorded in manifest.json (detects later edits).
        splits.check_integrity(SPLITS_DIR)
        # Load the stored split table (columns id, class, label, split).
        split_df = splits.load_split(SPLITS_DIR)
        # The split must cover exactly the images that are on disk, no more and no fewer.
        assert set(split_df["id"]) == set(index["id"]), "split.csv ids differ from data on disk"
        # Read the manifest, which lists every subset file with its seed and fraction.
        manifest = json.loads((SPLITS_DIR / "manifest.json").read_text())
        subsets = {}  # {subset_seed: {fraction: [ids]}}, the structure splits.verify expects
        # Load every labelled-subset file listed in the manifest into the nested dict above.
        for name, meta in manifest["subsets"].items():
            subsets.setdefault(meta["seed"], {})[meta["fraction"]] = splits.load_subset(meta["fraction"], meta["seed"])
        # Run the leakage checks: no overlap between splits, subsets inside train, nesting, all classes present.
        splits.verify(split_df, subsets)
        print("Splits OK: hashes match, no leakage, subsets nested and class-complete.")
        return

    # Refuse to overwrite an existing split unless --force was given, because that would change all results.
    if (SPLITS_DIR / splits.SPLIT_FILE).exists() and not args.force:
        raise SystemExit("splits/ already exists. Use --check to verify, or --force to regenerate (changes results!).")

    # Build the stratified train/val/test split and all nested subsets in memory (verified, not yet written).
    split_df, subsets = splits.build_all(index, split_seed=SPLIT_SEED, subset_seeds=SUBSET_SEEDS)
    # Write split.csv, the subset files and manifest.json (with hashes) into splits/; keep the manifest for printing.
    manifest = splits.save(split_df, subsets, SPLITS_DIR)
    # Print a short summary: the seed, the number of samples and how many images went to each split.
    print(json.dumps({k: manifest[k] for k in ("split_seed", "n_samples", "split_counts")}, indent=2))
    # Print one line per subset file: its size and the size of its rarest class.
    for name, meta in manifest["subsets"].items():
        print(f"  {name:32s} n={meta['n']:5d}  min/class={min(meta['class_counts'].values())}")


# Run main() only when the file is executed as a script, not when it is imported.
if __name__ == "__main__":
    main()
