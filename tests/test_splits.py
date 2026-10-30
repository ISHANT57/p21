"""Split logic tests on a synthetic, class-imbalanced index (no dataset needed)."""

import pandas as pd
import pytest

from eurosat_le.constants import CLASSES
from eurosat_le.data import splits


@pytest.fixture
def index():
    # Mimic EuroSAT's imbalance: 2000-3000 samples per class, scaled down 10x.
    sizes = [300, 300, 300, 250, 250, 200, 250, 300, 250, 300]
    rows = [(f"{c}/{c}_{i + 1}", c, k) for k, (c, n) in enumerate(zip(CLASSES, sizes)) for i in range(n)]
    return pd.DataFrame(rows, columns=["id", "class", "label"])


def test_split_proportions_and_disjoint(index):
    df = splits.make_split(index)
    counts = df["split"].value_counts(normalize=True)
    assert counts["train"] == pytest.approx(0.70, abs=0.01)
    assert counts["val"] == pytest.approx(0.15, abs=0.01)
    assert counts["test"] == pytest.approx(0.15, abs=0.01)
    splits.verify(df, {})


def test_split_is_stratified(index):
    df = splits.make_split(index)
    overall = index["label"].value_counts(normalize=True).sort_index()
    for s in ("train", "val", "test"):
        part = df[df["split"] == s]["label"].value_counts(normalize=True).sort_index()
        assert (part - overall).abs().max() < 0.01


def test_split_is_deterministic(index):
    a, b = splits.make_split(index, seed=42), splits.make_split(index, seed=42)
    assert a["split"].equals(b["split"])
    assert not a["split"].equals(splits.make_split(index, seed=7)["split"])


def test_subsets_nested_stratified_and_train_only(index):
    split_df, subsets = splits.build_all(index, subset_seeds=[0, 1])
    train_ids = set(split_df.loc[split_df["split"] == "train", "id"])
    for by_frac in subsets.values():
        assert set(by_frac[0.01]) <= set(by_frac[0.10]) <= set(by_frac[1.0]) == train_ids
        assert len(by_frac[0.10]) == pytest.approx(0.10 * len(train_ids), rel=0.05)
    assert subsets[0][0.01] != subsets[1][0.01], "different seeds should draw different subsets"


def test_tiny_fraction_keeps_every_class(index):
    split_df = splits.make_split(index)
    sub = splits.make_label_subsets(split_df, fractions=[0.001, 1.0], seed=0)
    labels = split_df.set_index("id").loc[sub[0.001], "label"]
    assert labels.nunique() == len(CLASSES)


def test_save_and_integrity(index, tmp_path):
    split_df, subsets = splits.build_all(index, subset_seeds=[0])
    splits.save(split_df, subsets, tmp_path)
    splits.check_integrity(tmp_path)
    assert splits.load_subset(0.01, 0, tmp_path) == subsets[0][0.01]
    (tmp_path / "subsets" / splits.subset_filename(0.01, 0)).write_text("tampered\n")
    with pytest.raises(RuntimeError):
        splits.check_integrity(tmp_path)
