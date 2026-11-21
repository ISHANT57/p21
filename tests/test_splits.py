"""Split logic tests on a synthetic, class-imbalanced index (no dataset needed)."""

# pandas: build the fake image index; pytest: fixtures, approx comparisons and expected errors.
import pandas as pd
import pytest

# CLASSES: the 10 class names. splits: the module that creates, checks, saves and loads the data splits.
from eurosat_le.constants import CLASSES
from eurosat_le.data import splits


@pytest.fixture
def index():
    """Fixture: a fake image index (id, class, label) with 2,700 images and different sizes per class."""
    # Mimic EuroSAT's imbalance: 2000-3000 samples per class, scaled down 10x.
    # Number of images in each of the 10 classes (same order as CLASSES).
    sizes = [300, 300, 300, 250, 250, 200, 250, 300, 250, 300]
    # One row per image: an id such as "Forest/Forest_1", its class name and its numeric label (0-9).
    rows = [(f"{c}/{c}_{i + 1}", c, k) for k, (c, n) in enumerate(zip(CLASSES, sizes)) for i in range(n)]
    return pd.DataFrame(rows, columns=["id", "class", "label"])


def test_split_proportions_and_disjoint(index):
    """The split must be about 70% train, 15% val, 15% test, and the three parts must not share images."""
    df = splits.make_split(index)
    # Share of each split among all images.
    counts = df["split"].value_counts(normalize=True)
    # Each share may be off by at most one percentage point.
    assert counts["train"] == pytest.approx(0.70, abs=0.01)
    assert counts["val"] == pytest.approx(0.15, abs=0.01)
    assert counts["test"] == pytest.approx(0.15, abs=0.01)
    # verify() raises an error if ids are duplicated or if train/val/test overlap (no subsets given here).
    splits.verify(df, {})


def test_split_is_stratified(index):
    """Every split must have almost the same class mix as the whole data set (stratified splitting)."""
    df = splits.make_split(index)
    # Class proportions of the full data set, sorted by label so they can be compared position by position.
    overall = index["label"].value_counts(normalize=True).sort_index()
    for s in ("train", "val", "test"):
        # Class proportions inside this split.
        part = df[df["split"] == s]["label"].value_counts(normalize=True).sort_index()
        # No class share may differ from the overall share by more than one percentage point.
        assert (part - overall).abs().max() < 0.01


def test_split_is_deterministic(index):
    """The same seed must always give the same split, and a different seed must give a different one."""
    # Build the split twice with the same seed 42.
    a, b = splits.make_split(index, seed=42), splits.make_split(index, seed=42)
    # Identical assignment of every image to train/val/test.
    assert a["split"].equals(b["split"])
    # A different seed (7) must not reproduce the same assignment (otherwise the seed would do nothing).
    assert not a["split"].equals(splits.make_split(index, seed=7)["split"])


def test_subsets_nested_stratified_and_train_only(index):
    """Label subsets (1%, 10%, 100%) must be nested, drawn from train only, correctly sized, and differ per seed."""
    # Build the split and the label subsets for two subset seeds (0 and 1); build_all also runs verify().
    split_df, subsets = splits.build_all(index, subset_seeds=[0, 1])
    # Ids of all training images: the only images a subset is allowed to contain.
    train_ids = set(split_df.loc[split_df["split"] == "train", "id"])
    for by_frac in subsets.values():
        # Nesting: every 1% image is also in the 10% subset, which is inside the 100% subset, and 100% = train.
        assert set(by_frac[0.01]) <= set(by_frac[0.10]) <= set(by_frac[1.0]) == train_ids
        # The 10% subset holds about 10% of the training images (5% relative tolerance for rounding per class).
        assert len(by_frac[0.10]) == pytest.approx(0.10 * len(train_ids), rel=0.05)
    # Different subset seeds must pick different 1% subsets, otherwise "several seeds" would mean nothing.
    assert subsets[0][0.01] != subsets[1][0.01], "different seeds should draw different subsets"


def test_tiny_fraction_keeps_every_class(index):
    """Even a very small subset (0.1%) must contain at least one image of every class."""
    split_df = splits.make_split(index)
    # 0.1% of the training images would be only ~2 images; the rule "at least 1 per class" must still hold.
    sub = splits.make_label_subsets(split_df, fractions=[0.001, 1.0], seed=0)
    # Look up the labels of the images in the 0.1% subset and count the different classes.
    labels = split_df.set_index("id").loc[sub[0.001], "label"]
    assert labels.nunique() == len(CLASSES)


def test_save_and_integrity(index, tmp_path):
    """Saved split files must load back unchanged, and the integrity check must catch a manual edit."""
    split_df, subsets = splits.build_all(index, subset_seeds=[0])
    # Write split.csv, the subset files and manifest.json (with file hashes) to a temporary folder.
    splits.save(split_df, subsets, tmp_path)
    # Right after saving, every file matches its recorded hash, so this must not raise.
    splits.check_integrity(tmp_path)
    # Loading a subset from disk gives back exactly the ids that were saved.
    assert splits.load_subset(0.01, 0, tmp_path) == subsets[0][0.01]
    # Tamper with one subset file; its hash no longer matches the manifest.
    (tmp_path / "subsets" / splits.subset_filename(0.01, 0)).write_text("tampered\n")
    # The integrity check must now raise, protecting against accidental changes of the official splits.
    with pytest.raises(RuntimeError):
        splits.check_integrity(tmp_path)
