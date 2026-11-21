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

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from eurosat_le.constants import CLASSES, LABEL_FRACTIONS, SPLIT_SEED, SPLITS_DIR, SUBSET_SEEDS

SPLIT_FILE = "split.csv"
SUBSET_DIR = "subsets"


def make_split(index: pd.DataFrame, seed: int = SPLIT_SEED, val: float = 0.15, test: float = 0.15) -> pd.DataFrame:
    """Return a copy of `index` with a `split` column in {train, val, test}.

    `index` needs a "label" column (the stratification target). The test set is taken first
    (`test` of all samples), then validation is taken from the remainder; because the remainder
    is smaller, the validation fraction is rescaled to val / (1 - test) so that val ends up
    as `val` of the full data (0.15/0.85 for the default 70/15/15). Both calls use the same
    seed, so the result is deterministic.
    """
    rest, test_df = train_test_split(index, test_size=test, stratify=index["label"], random_state=seed)
    train_df, val_df = train_test_split(
        rest, test_size=val / (1 - test), stratify=rest["label"], random_state=seed
    )
    out = index.copy()
    out["split"] = ""
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
    """
    train = split_df[split_df["split"] == "train"]
    rng = np.random.default_rng(seed)
    subsets: dict[float, list[str]] = {f: [] for f in fractions}
    # Sorted label order fixes the order in which the RNG is consumed, so results do not depend on
    # the order of rows in the DataFrame.
    for label in sorted(train["label"].unique()):
        ids = train.loc[train["label"] == label, "id"].to_numpy()
        perm = ids[rng.permutation(len(ids))]
        for f in fractions:
            # Python round() rounds exact halves to the nearest even number; the resulting subset sizes
            # are recorded in manifest.json.
            k = len(ids) if f >= 1.0 else max(1, int(round(f * len(ids))))
            subsets[f] += perm[:k].tolist()
    order = {i: n for n, i in enumerate(split_df["id"])}
    return {f: sorted(v, key=order.__getitem__) for f, v in subsets.items()}


def subset_filename(fraction: float, seed: int) -> str:
    """File name of a subset, e.g. fraction=0.01, seed=0 -> "train_1pct_seed0.txt" (%g drops trailing zeros)."""
    return f"train_{fraction * 100:g}pct_seed{seed}.txt"


def verify(split_df: pd.DataFrame, subsets: dict[int, dict[float, list[str]]]) -> None:
    """Assert the split and subsets have no leakage and the expected structure.

    `subsets` is {subset_seed: {fraction: [ids]}}. Checks: ids are unique; train/val/test are
    pairwise disjoint; each subset has no duplicates and lies inside train; for each seed the
    subsets are nested in increasing fraction order; the 100% subset (if present) equals train;
    and the smallest subset contains every class. Raises AssertionError on the first failure.
    """
    ids = split_df["id"]
    assert ids.is_unique, "duplicate sample ids"
    assert set(split_df["split"]) == {"train", "val", "test"}
    groups = {s: set(ids[split_df["split"] == s]) for s in ("train", "val", "test")}
    assert not (groups["train"] & groups["val"]), "train/val overlap"
    assert not (groups["train"] & groups["test"]), "train/test overlap"
    assert not (groups["val"] & groups["test"]), "val/test overlap"
    for seed, by_frac in subsets.items():
        fracs = sorted(by_frac)
        for f in fracs:
            s = set(by_frac[f])
            assert len(s) == len(by_frac[f]), f"duplicates in subset {f}/{seed}"
            assert s <= groups["train"], f"subset {f}/{seed} leaks outside train"
        for small, big in zip(fracs, fracs[1:]):
            assert set(by_frac[small]) <= set(by_frac[big]), f"subset {small} not nested in {big} (seed {seed})"
        if max(fracs) >= 1.0:
            assert set(by_frac[max(fracs)]) == groups["train"], f"100% subset != train (seed {seed})"
        labels = split_df.set_index("id").loc[by_frac[min(fracs)], "label"]
        assert labels.nunique() == len(CLASSES), f"subset {min(fracs)}/{seed} is missing a class"


def _sha256(path: Path) -> str:
    """SHA-256 hex digest of a whole file (the split files are small)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(split_df: pd.DataFrame, subsets: dict[int, dict[float, list[str]]], out_dir: Path = SPLITS_DIR,
         split_seed: int = SPLIT_SEED) -> dict:
    """Write split.csv, subsets/*.txt and manifest.json; return the manifest.

    split.csv keeps only (id, class, label, split). Each subset file has one id per line. The
    manifest records the seed, split sizes and per-class counts, and a SHA-256 for every file
    written, which check_integrity() later compares against. The manifest is written last so
    its hashes describe the files as they are on disk.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / SUBSET_DIR).mkdir(exist_ok=True)
    split_path = out_dir / SPLIT_FILE
    split_df[["id", "class", "label", "split"]].to_csv(split_path, index=False)

    label_of = split_df.set_index("id")["class"]
    manifest = {
        "split_seed": split_seed,
        "n_samples": len(split_df),
        "split_counts": split_df["split"].value_counts().reindex(["train", "val", "test"]).to_dict(),
        "split_class_counts": {
            s: split_df[split_df["split"] == s]["class"].value_counts().reindex(CLASSES).to_dict()
            for s in ("train", "val", "test")
        },
        "files": {SPLIT_FILE: _sha256(split_path)},
        "subsets": {},
    }
    for seed, by_frac in subsets.items():
        for f, ids in by_frac.items():
            name = f"{SUBSET_DIR}/{subset_filename(f, seed)}"
            (out_dir / name).write_text("\n".join(ids) + "\n")
            manifest["files"][name] = _sha256(out_dir / name)
            manifest["subsets"][name] = {
                "fraction": f,
                "seed": seed,
                "n": len(ids),
                "class_counts": label_of.loc[ids].value_counts().reindex(CLASSES).to_dict(),
            }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_split(out_dir: Path = SPLITS_DIR) -> pd.DataFrame:
    """Read split.csv as a DataFrame with columns id, class, label, split. Row order defines cache row order."""
    return pd.read_csv(out_dir / SPLIT_FILE)


def load_subset(fraction: float, seed: int, out_dir: Path = SPLITS_DIR) -> list[str]:
    """Ids of one labelled subset (a subset of train), read from splits/subsets/ in the stored order."""
    return (out_dir / SUBSET_DIR / subset_filename(fraction, seed)).read_text().split()


def check_integrity(out_dir: Path = SPLITS_DIR) -> None:
    """Raise if any committed split file differs from the hash in manifest.json.

    Guards against accidental edits or regeneration of the split, which would silently change
    which images are in train/val/test and invalidate comparisons between runs.
    """
    manifest = json.loads((out_dir / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if _sha256(out_dir / name) != digest:
            raise RuntimeError(f"{name} does not match manifest.json; splits were modified")


def build_all(index: pd.DataFrame, split_seed: int = SPLIT_SEED, subset_seeds=SUBSET_SEEDS,
              fractions=LABEL_FRACTIONS):
    """Build the split and all subsets in memory and verify them; nothing is written to disk.

    Returns (split_df, subsets) with subsets = {subset_seed: {fraction: [ids]}}; pass both to save().
    """
    split_df = make_split(index, seed=split_seed)
    subsets = {s: make_label_subsets(split_df, fractions, seed=s) for s in subset_seeds}
    verify(split_df, subsets)
    return split_df, subsets
