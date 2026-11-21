"""Reproducible train/val/test split and nested labelled subsets.

Procedure (documented in docs/DATA.md):
1. Stratified 70/15/15 split of all 27,000 samples with SPLIT_SEED. Test is carved
   out first, then validation from the remainder.
2. For each subset seed, shuffle the training ids of every class once and take the
   first round(fraction * n_class) ids (at least 1). Because every fraction is a
   prefix of the same per-class permutation, subsets are nested: 1% ⊂ 10% ⊂ 100%.
3. Validation and test never change; labelled subsets come only from train.

Pipeline position: scripts/make_splits.py calls build_all() + save() once; the resulting files in
splits/ (split.csv, subsets/*.txt, manifest.json) are committed to git and read by everything
downstream (cache, datasets, training) through load_split / load_subset.

Design decisions:
    - Stratified by class with a fixed seed (SPLIT_SEED), so every class has the same proportion in
      train/val/test and the partition is identical for every model, input type and run.
    - Subsets are nested and drawn per class from one shuffled order per subset seed. Going from 1%
      to 10% to 100% only ever adds examples, which makes label-efficiency curves comparable.
    - Different subset seeds give independent draws, so results can be averaged over seeds.
    - Rounding is done per class, so each subset keeps the class proportions of train.
    - verify() asserts the leakage properties (no overlap between splits, subsets inside train,
      nesting, every class present at the smallest fraction), and manifest.json stores a SHA-256
      of every file so that check_integrity() can detect later edits.
"""

# Allow modern type-hint syntax on older Python versions.
from __future__ import annotations

# hashlib: file fingerprints; json: write the manifest; Path: file paths.
import hashlib
import json
from pathlib import Path

# numpy: seeded random shuffling; pandas: the sample table; sklearn: stratified train/test splitting.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Class list, label fractions, seeds and the default splits folder.
from eurosat_le.constants import CLASSES, LABEL_FRACTIONS, SPLIT_SEED, SPLITS_DIR, SUBSET_SEEDS

# File name of the main split table inside the splits folder.
SPLIT_FILE = "split.csv"
# Sub-folder (inside the splits folder) that holds one text file per labelled subset.
SUBSET_DIR = "subsets"


def make_split(index: pd.DataFrame, seed: int = SPLIT_SEED, val: float = 0.15, test: float = 0.15) -> pd.DataFrame:
    """Return a copy of `index` with a `split` column in {train, val, test}.

    `index` needs a "label" column (the stratification target). The test set is taken first
    (`test` of all samples), then validation is taken from the remainder; because the remainder
    is smaller, the validation fraction is rescaled to val / (1 - test) so that val ends up
    as `val` of the full data (0.15/0.85 for the default 70/15/15). Both calls use the same
    seed, so the result is deterministic.

    Args:
        index: table of all samples (columns id, class, label) from data.index.build_index.
        seed: random seed for the split.
        val: fraction of ALL samples to use for validation.
        test: fraction of ALL samples to use for testing.
    """
    # First cut: carve out the test set. "stratify" keeps the class proportions equal in both parts.
    rest, test_df = train_test_split(index, test_size=test, stratify=index["label"], random_state=seed)
    # Second cut: take validation out of what is left; the fraction is rescaled because the pool is smaller.
    train_df, val_df = train_test_split(
        rest, test_size=val / (1 - test), stratify=rest["label"], random_state=seed
    )
    # Work on a copy so the caller's table is not modified.
    out = index.copy()
    # Start with an empty split name for every row.
    out["split"] = ""
    # Fill in the split name for each group, matching rows by their original table index.
    out.loc[train_df.index, "split"] = "train"
    out.loc[val_df.index, "split"] = "val"
    out.loc[test_df.index, "split"] = "test"
    return out


def make_label_subsets(split_df: pd.DataFrame, fractions=LABEL_FRACTIONS, seed: int = 0) -> dict[float, list[str]]:
    """Return {fraction: sorted list of train ids}, nested and class-stratified.

    Only rows with split == "train" are used, so no validation or test id can enter a subset.
    For each class the train ids are shuffled once with np.random.default_rng(seed) and every
    fraction takes a prefix of that same permutation (at least 1 id per class; a fraction >= 1
    takes everything). Hence the subsets are nested. The returned lists are re-sorted into
    split.csv row order, which makes the written subset files independent of the shuffle order.

    Args:
        split_df: output of make_split (needs columns id, label, split).
        fractions: label fractions to produce, e.g. [0.01, 0.1, 1.0].
        seed: subset seed; different seeds give independent draws.
    """
    # Only training rows may ever become labelled data.
    train = split_df[split_df["split"] == "train"]
    # Seeded random number generator (reproducible shuffles).
    rng = np.random.default_rng(seed)
    # One empty id list per fraction; they are filled class by class below.
    subsets: dict[float, list[str]] = {f: [] for f in fractions}
    # Sorted label order fixes the order in which the RNG is consumed, so results do not depend on
    # the order of rows in the DataFrame.
    for label in sorted(train["label"].unique()):
        # All training ids of this class.
        ids = train.loc[train["label"] == label, "id"].to_numpy()
        # One random ordering of them; every fraction takes a prefix of this same ordering (so subsets nest).
        perm = ids[rng.permutation(len(ids))]
        for f in fractions:
            # Python round() rounds exact halves to the nearest even number; the resulting subset sizes
            # are recorded in manifest.json.
            k = len(ids) if f >= 1.0 else max(1, int(round(f * len(ids))))
            # Add the first k shuffled ids of this class to the subset for fraction f.
            subsets[f] += perm[:k].tolist()
    # Position of each id in split.csv, used to sort every subset back into split.csv order.
    order = {i: n for n, i in enumerate(split_df["id"])}
    return {f: sorted(v, key=order.__getitem__) for f, v in subsets.items()}


def subset_filename(fraction: float, seed: int) -> str:
    """File name of a subset, e.g. fraction=0.01, seed=0 -> "train_1pct_seed0.txt" (%g drops trailing zeros).

    Args:
        fraction: label fraction (0.01, 0.1 or 1.0).
        seed: subset seed.
    """
    # Convert the fraction to a percentage and format it without trailing zeros (1.0 -> "1", 100.0 -> "100").
    return f"train_{fraction * 100:g}pct_seed{seed}.txt"


def verify(split_df: pd.DataFrame, subsets: dict[int, dict[float, list[str]]]) -> None:
    """Assert the split and subsets have no leakage and the expected structure.

    `subsets` is {subset_seed: {fraction: [ids]}}. Checks: ids are unique; train/val/test are
    pairwise disjoint; each subset has no duplicates and lies inside train; for each seed the
    subsets are nested in increasing fraction order; the 100% subset (if present) equals train;
    and the smallest subset contains every class. Raises AssertionError on the first failure.

    Args:
        split_df: the split table (columns id, class, label, split).
        subsets: {subset_seed: {fraction: [ids]}} as returned by make_label_subsets for each seed.
    """
    ids = split_df["id"]
    # Every sample must appear exactly once.
    assert ids.is_unique, "duplicate sample ids"
    # Exactly the three expected split names, nothing else (e.g. no empty names left over).
    assert set(split_df["split"]) == {"train", "val", "test"}
    # Set of ids for each split, so overlaps can be tested with set intersection (&).
    groups = {s: set(ids[split_df["split"] == s]) for s in ("train", "val", "test")}
    # No image may be in two splits, otherwise test results would be contaminated.
    assert not (groups["train"] & groups["val"]), "train/val overlap"
    assert not (groups["train"] & groups["test"]), "train/test overlap"
    assert not (groups["val"] & groups["test"]), "val/test overlap"
    for seed, by_frac in subsets.items():
        # Fractions in increasing order, so each one is compared with the next larger one.
        fracs = sorted(by_frac)
        for f in fracs:
            s = set(by_frac[f])
            # No id repeated inside a subset.
            assert len(s) == len(by_frac[f]), f"duplicates in subset {f}/{seed}"
            # "<=" on sets means "is a subset of": every labelled id must be a training id.
            assert s <= groups["train"], f"subset {f}/{seed} leaks outside train"
        # Check neighbouring pairs (smaller, next bigger): the smaller subset must lie inside the bigger one.
        for small, big in zip(fracs, fracs[1:]):
            assert set(by_frac[small]) <= set(by_frac[big]), f"subset {small} not nested in {big} (seed {seed})"
        if max(fracs) >= 1.0:
            # The full-label subset must be exactly the whole training set.
            assert set(by_frac[max(fracs)]) == groups["train"], f"100% subset != train (seed {seed})"
        # Labels of the ids in the smallest subset (looked up by id in the split table).
        labels = split_df.set_index("id").loc[by_frac[min(fracs)], "label"]
        # Even the 1% subset must contain at least one example of each of the 10 classes.
        assert labels.nunique() == len(CLASSES), f"subset {min(fracs)}/{seed} is missing a class"


def _sha256(path: Path) -> str:
    """SHA-256 hex digest of a whole file (the split files are small).

    Args:
        path: file to fingerprint.
    """
    # Read all bytes at once and hash them; any change to the file changes the digest.
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(split_df: pd.DataFrame, subsets: dict[int, dict[float, list[str]]], out_dir: Path = SPLITS_DIR,
         split_seed: int = SPLIT_SEED) -> dict:
    """Write split.csv, subsets/*.txt and manifest.json; return the manifest.

    split.csv keeps only (id, class, label, split). Each subset file has one id per line. The
    manifest records the seed, split sizes and per-class counts, and a SHA-256 for every file
    written, which check_integrity() later compares against. The manifest is written last so
    its hashes describe the files as they are on disk.

    Args:
        split_df: the split table.
        subsets: {subset_seed: {fraction: [ids]}}.
        out_dir: folder to write into (created if missing).
        split_seed: seed used for the split; recorded in the manifest only.
    """
    # Create the output folder and its "subsets" sub-folder if needed.
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / SUBSET_DIR).mkdir(exist_ok=True)
    split_path = out_dir / SPLIT_FILE
    # Write only the four needed columns; the row order of the table becomes the order of the file.
    split_df[["id", "class", "label", "split"]].to_csv(split_path, index=False)

    # Lookup table id -> class name, used for the per-class counts of each subset.
    label_of = split_df.set_index("id")["class"]
    # Summary record of how everything was produced (written to manifest.json at the end).
    manifest = {
        "split_seed": split_seed,
        "n_samples": len(split_df),
        # Number of samples in each split, in a fixed order.
        "split_counts": split_df["split"].value_counts().reindex(["train", "val", "test"]).to_dict(),
        # For each split, how many samples each class has (shows the stratification worked).
        "split_class_counts": {
            s: split_df[split_df["split"] == s]["class"].value_counts().reindex(CLASSES).to_dict()
            for s in ("train", "val", "test")
        },
        # Fingerprint of every written file (subset files are added in the loop below).
        "files": {SPLIT_FILE: _sha256(split_path)},
        "subsets": {},
    }
    # One text file per (seed, fraction) pair.
    for seed, by_frac in subsets.items():
        for f, ids in by_frac.items():
            # Relative file name such as "subsets/train_1pct_seed0.txt".
            name = f"{SUBSET_DIR}/{subset_filename(f, seed)}"
            # One id per line.
            (out_dir / name).write_text("\n".join(ids) + "\n")
            # Record the file's fingerprint so later edits can be detected.
            manifest["files"][name] = _sha256(out_dir / name)
            # Record how this subset was made and how many samples of each class it holds.
            manifest["subsets"][name] = {
                "fraction": f,
                "seed": seed,
                "n": len(ids),
                "class_counts": label_of.loc[ids].value_counts().reindex(CLASSES).to_dict(),
            }
    # Save the manifest as readable JSON, after all the files it describes are written.
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_split(out_dir: Path = SPLITS_DIR) -> pd.DataFrame:
    """Read split.csv as a DataFrame with columns id, class, label, split. Row order defines cache row order.

    Args:
        out_dir: folder containing split.csv.
    """
    # Parse the CSV into a table.
    return pd.read_csv(out_dir / SPLIT_FILE)


def load_subset(fraction: float, seed: int, out_dir: Path = SPLITS_DIR) -> list[str]:
    """Ids of one labelled subset (a subset of train), read from splits/subsets/ in the stored order.

    Args:
        fraction: label fraction (0.01, 0.1 or 1.0).
        seed: subset seed (0, 1 or 2).
        out_dir: splits folder.
    """
    # Read the text file and split it on whitespace/newlines: one id per entry.
    return (out_dir / SUBSET_DIR / subset_filename(fraction, seed)).read_text().split()


def check_integrity(out_dir: Path = SPLITS_DIR) -> None:
    """Raise if any committed split file differs from the hash in manifest.json.

    Guards against accidental edits or regeneration of the split, which would silently change
    which images are in train/val/test and invalidate comparisons between runs.

    Args:
        out_dir: splits folder containing manifest.json and the files it lists.
    """
    # Load the recorded fingerprints.
    manifest = json.loads((out_dir / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        # Recompute each file's fingerprint now and compare with the recorded one.
        if _sha256(out_dir / name) != digest:
            raise RuntimeError(f"{name} does not match manifest.json; splits were modified")


def build_all(index: pd.DataFrame, split_seed: int = SPLIT_SEED, subset_seeds=SUBSET_SEEDS,
              fractions=LABEL_FRACTIONS):
    """Build the split and all subsets in memory and verify them; nothing is written to disk.

    Returns (split_df, subsets) with subsets = {subset_seed: {fraction: [ids]}}; pass both to save().

    Args:
        index: table of all samples from data.index.build_index.
        split_seed: seed of the train/val/test split.
        subset_seeds: one labelled-subset draw is made per seed in this list.
        fractions: label fractions for each draw.
    """
    # Step 1: stratified train/val/test split.
    split_df = make_split(index, seed=split_seed)
    # Step 2: for every subset seed, build the nested labelled subsets.
    subsets = {s: make_label_subsets(split_df, fractions, seed=s) for s in subset_seeds}
    # Step 3: stop with an AssertionError if anything leaks or is not nested.
    verify(split_df, subsets)
    return split_df, subsets
