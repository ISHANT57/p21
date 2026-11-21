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

# ====================================================================================================
# CONCEPTS USED IN THIS FILE
# - Train / validation / test split: train = data the model learns from; validation = data used to tune and pick the
#   best epoch; test = data used once for the final score. Here 70% / 15% / 15% of the 27,000 images. split.csv stores
#   which split each id is in.
# - Stratified split: each class is divided in the same proportions, so train, val and test all contain about the same
#   share of every class. Done with scikit-learn's train_test_split(stratify=label).
# - Seed / random_state: a number that fixes the random choices. SPLIT_SEED makes the split identical every time; the
#   subset seed (0, 1, 2) makes independent labelled-subset draws.
# - Label fraction / labelled subset: only a fraction (1%, 10%, 100%) of the training images are treated as labelled.
#   This is the label-efficiency experiment.
# - Nested subsets: every smaller subset is contained in the larger one (1% inside 10% inside 100%). It is done by
#   shuffling each class once and taking prefixes of the same shuffled list.
# - Permutation / prefix: a random re-ordering of a list (rng.permutation); a prefix is its first k items.
# - np.random.default_rng: NumPy's modern seeded random generator; the same seed always gives the same permutations.
# - Rounding and 'at least 1': k = round(fraction * class size), but never less than 1, so even 1% keeps every class.
#   Python's round() sends exact halves to the nearest even number.
# - Data leakage: when information from validation/test data gets into training, making scores look better than they
#   are. verify() asserts there is none (no overlapping ids, subsets only from train).
# - Unlabelled pool: the training ids that are NOT in the labelled subset; used later for semi-supervised learning
#   (see data.datasets.unlabelled_ids).
# - split.csv / subsets/*.txt / manifest.json: split.csv holds id, class, label, split for all samples; each subset
#   file lists the labelled ids of one (fraction, seed); manifest.json records seeds, counts and a hash of every file.
# - SHA-256 / checksum / integrity: a fingerprint of a file's bytes. The manifest stores it for each file;
#   check_integrity recomputes it to detect accidental edits or regeneration.
# - DataFrame (pandas) and index alignment: a table of rows and columns; rows are matched by their index, which is how
#   the split names are written back to the right rows (out.loc[...]).
# - Type hints / defaults: dict[float, list[str]] means a dictionary from fraction to list of ids; default arguments
#   come from constants.py.
# ====================================================================================================
#
# FILE-LEVEL Q&A
# Q: What is this file for?
# A: It creates the fixed train/val/test split and the nested labelled subsets, checks them for leakage, writes them
#    to splits/, and reads them back for the rest of the project.
# Q: Where does it fit in the pipeline?
# A: After data.index. scripts/make_splits.py calls build_all() then save() once; the files in splits/ are committed
#    to git and read by data.cache, data.datasets and training through load_split / load_subset.
# Q: What does it read and write?
# A: It writes splits/split.csv, splits/subsets/train_<pct>pct_seed<k>.txt and splits/manifest.json. It reads
#    split.csv and the subset files back, and manifest.json for the integrity check.
# Q: Why is the split fixed and stored in files?
# A: So every model, input type and run is evaluated on exactly the same test images and trained on the same labelled
#    subsets; otherwise results would not be comparable.

# Q: What are SPLIT_FILE and SUBSET_DIR?
# A: Fixed names: 'split.csv' is the split table and 'subsets' is the sub-folder with one text file per labelled
#    subset; shared by save and the loaders so they agree.
# File name of the main split table inside the splits folder.
SPLIT_FILE = "split.csv"
# Sub-folder (inside the splits folder) that holds one text file per labelled subset.
SUBSET_DIR = "subsets"


# Q: What does this function do?
# A: It divides all samples into train, val and test (70/15/15 by default), keeping the class proportions, and returns
#    a copy of the table with a 'split' column.
# Q: Why is the validation fraction val / (1 - test)?
# A: Validation is taken from what is left after the test set is removed (85% of the data). To end up with 15% of ALL
#    samples, 0.15 / 0.85 of the remainder is needed.
# Q: Why use stratify and a seed?
# A: Stratify keeps each class equally represented in every split; the fixed seed makes the split identical on every
#    run and machine.
# Q: What are the inputs and what does it return?
# A: index (DataFrame with id, class, label), seed, val, test. It returns a copy of index with an extra 'split' column
#    and does not change the input table.
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


# Q: What does this function do?
# A: For one subset seed it builds the labelled subsets for every fraction (e.g. 1%, 10%, 100%) of the training set,
#    as lists of ids.
# Q: Why are the subsets nested and taken per class?
# A: Each class's training ids are shuffled once and every fraction takes the first k of that same order, so 1% is
#    inside 10% inside 100%. Doing it per class keeps class proportions, and nesting makes label-efficiency curves
#    comparable because going to more labels only adds examples.
# Q: Why at least one id per class, and why are the lists re-sorted?
# A: max(1, ...) guarantees that even the 1% subset contains every class. Re-sorting into split.csv order makes the
#    written files independent of the shuffle order.
# Q: What are the inputs and what does it return?
# A: split_df (output of make_split), fractions, seed. It returns a dict {fraction: sorted list of train ids}. Only
#    train rows are used, so no val or test id can appear.
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


# Q: What does this function do?
# A: It builds the file name of a subset, e.g. fraction 0.01 and seed 0 give 'train_1pct_seed0.txt'.
# Q: Why is the same function used for writing and reading?
# A: save and load_subset both call it, so the writer and the reader can never disagree about the name.
# Q: What are the inputs and what does it return?
# A: fraction (float) and seed (int). It returns a string; '%g' formatting drops trailing zeros (1.0 gives '100').
def subset_filename(fraction: float, seed: int) -> str:
    """File name of a subset, e.g. fraction=0.01, seed=0 -> "train_1pct_seed0.txt" (%g drops trailing zeros).

    Args:
        fraction: label fraction (0.01, 0.1 or 1.0).
        seed: subset seed.
    """
    # Convert the fraction to a percentage and format it without trailing zeros (1.0 -> "1", 100.0 -> "100").
    return f"train_{fraction * 100:g}pct_seed{seed}.txt"


# Q: What does this function do?
# A: It asserts that the split and the subsets are leak-free and well formed (unique ids, disjoint splits, subsets
#    inside train and nested, 100% subset equals train, every class in the smallest subset).
# Q: Why is this needed? What would go wrong without it?
# A: If a test image were also in train, or a subset contained a non-train image, the reported accuracy would be
#    unfairly high. The assertions stop the pipeline instead of silently producing invalid experiments.
# Q: What are the inputs and what does it return?
# A: split_df and subsets ({subset_seed: {fraction: ids}}). It returns None; it raises AssertionError at the first
#    failed check.
# Q: Why check that the smallest subset contains every class?
# A: A 1% subset that missed a class would make that class impossible to learn and would distort the label-efficiency
#    results.
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


# Q: What does this function do?
# A: It returns the SHA-256 fingerprint (hex string) of a whole file.
# Q: Why read the whole file at once here, unlike in data.download?
# A: The split files are small (a CSV and short text files), so there is no memory problem.
# Q: What are the inputs and what does it return?
# A: path (file). It returns a 64-character hexadecimal string.
def _sha256(path: Path) -> str:
    """SHA-256 hex digest of a whole file (the split files are small).

    Args:
        path: file to fingerprint.
    """
    # Read all bytes at once and hash them; any change to the file changes the digest.
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Q: What does this function do?
# A: It writes split.csv, one text file per (seed, fraction) subset and manifest.json, and returns the manifest dict.
# Q: What is the manifest for and why is it written last?
# A: It records the seed, split sizes, per-class counts and a SHA-256 of every written file, so check_integrity can
#    later detect edits. Written last, its hashes describe the files exactly as they are on disk.
# Q: What are the inputs and what does it return?
# A: split_df, subsets ({subset_seed: {fraction: ids}}), out_dir (default splits/) and split_seed (recorded only). It
#    returns the manifest dictionary.
# Q: Why does split.csv keep only four columns and keep row order?
# A: id, class, label and split are all that is needed, and the row order of split.csv defines the row order of the
#    cache array.
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


# Q: What does this function do?
# A: It reads split.csv back into a DataFrame (id, class, label, split).
# Q: Why does row order matter?
# A: The packed cache array has the same row order as split.csv, so a sample's row here is its row in the array.
# Q: What are the inputs and what does it return?
# A: out_dir (splits folder). It returns a pandas DataFrame.
def load_split(out_dir: Path = SPLITS_DIR) -> pd.DataFrame:
    """Read split.csv as a DataFrame with columns id, class, label, split. Row order defines cache row order.

    Args:
        out_dir: folder containing split.csv.
    """
    # Parse the CSV into a table.
    return pd.read_csv(out_dir / SPLIT_FILE)


# Q: What does this function do?
# A: It reads the id list of one labelled subset (a fraction and a subset seed) from splits/subsets/.
# Q: Why are subsets read from files instead of being re-drawn?
# A: Reading the committed files guarantees everyone uses exactly the same labelled images, even if library versions
#    change.
# Q: What are the inputs and what does it return?
# A: fraction, seed, out_dir. It returns a list of id strings in stored order (the text file is split on whitespace).
def load_subset(fraction: float, seed: int, out_dir: Path = SPLITS_DIR) -> list[str]:
    """Ids of one labelled subset (a subset of train), read from splits/subsets/ in the stored order.

    Args:
        fraction: label fraction (0.01, 0.1 or 1.0).
        seed: subset seed (0, 1 or 2).
        out_dir: splits folder.
    """
    # Read the text file and split it on whitespace/newlines: one id per entry.
    return (out_dir / SUBSET_DIR / subset_filename(fraction, seed)).read_text().split()


# Q: What does this function do?
# A: It recomputes the SHA-256 of every file listed in manifest.json and raises RuntimeError if any differs.
# Q: Why is this needed?
# A: If someone edits or regenerates the split by accident, train/val/test would silently change and results from
#    different runs could no longer be compared.
# Q: What are the inputs and what does it return?
# A: out_dir (splits folder). It returns None; success means no exception. Scripts such as build_cache.py and
#    inspect_dataset.py call it before using the splits.
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


# Q: What does this function do?
# A: It runs the whole procedure in memory: make the split, make the nested subsets for every subset seed, then verify
#    them.
# Q: Why does it not write files?
# A: Building and saving are separate steps, so the result can be checked (or inspected) before anything is written;
#    save() does the writing.
# Q: What are the inputs and what does it return?
# A: index (from build_index), split_seed, subset_seeds, fractions. It returns (split_df, subsets), where subsets =
#    {subset_seed: {fraction: ids}}.
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
