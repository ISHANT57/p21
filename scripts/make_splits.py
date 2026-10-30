"""Create the fixed train/val/test split and the nested 1%/10%/100% labelled subsets.

Usage:
    python scripts/make_splits.py            # writes splits/ (refuses to overwrite)
    python scripts/make_splits.py --check    # verify committed splits against manifest + data

The split files in splits/ are committed to git. Every group member and every
Colab/Kaggle run must use those files instead of regenerating them.
"""

import argparse
import json

from eurosat_le.constants import DATASETS, RAW_DIR, SPLIT_SEED, SPLITS_DIR, SUBSET_SEEDS
from eurosat_le.data import splits
from eurosat_le.data.index import build_index, check_modalities_match


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true", help="Only verify existing split files.")
    p.add_argument("--force", action="store_true", help="Overwrite existing split files.")
    args = p.parse_args()

    have_ms = (RAW_DIR / DATASETS["ms"]["folder"]).is_dir()
    index = build_index(RAW_DIR, "rgb")
    if have_ms:
        check_modalities_match(RAW_DIR)
        print("RGB and MS folders contain identical sample ids.")
    else:
        print("MS folder not found; split built from RGB ids (identical by construction of EuroSAT).")

    if args.check:
        splits.check_integrity(SPLITS_DIR)
        split_df = splits.load_split(SPLITS_DIR)
        assert set(split_df["id"]) == set(index["id"]), "split.csv ids differ from data on disk"
        manifest = json.loads((SPLITS_DIR / "manifest.json").read_text())
        subsets = {}
        for name, meta in manifest["subsets"].items():
            subsets.setdefault(meta["seed"], {})[meta["fraction"]] = splits.load_subset(meta["fraction"], meta["seed"])
        splits.verify(split_df, subsets)
        print("Splits OK: hashes match, no leakage, subsets nested and class-complete.")
        return

    if (SPLITS_DIR / splits.SPLIT_FILE).exists() and not args.force:
        raise SystemExit("splits/ already exists. Use --check to verify, or --force to regenerate (changes results!).")

    split_df, subsets = splits.build_all(index, split_seed=SPLIT_SEED, subset_seeds=SUBSET_SEEDS)
    manifest = splits.save(split_df, subsets, SPLITS_DIR)
    print(json.dumps({k: manifest[k] for k in ("split_seed", "n_samples", "split_counts")}, indent=2))
    for name, meta in manifest["subsets"].items():
        print(f"  {name:32s} n={meta['n']:5d}  min/class={min(meta['class_counts'].values())}")


if __name__ == "__main__":
    main()
