"""Split logic tests on a synthetic, class-imbalanced index (no dataset needed)."""

# pandas: build the fake image index; pytest: fixtures, approx comparisons and expected errors.
import pandas as pd
import pytest

# CLASSES: the 10 class names. splits: the module that creates, checks, saves and loads the data splits.
from eurosat_le.constants import CLASSES
from eurosat_le.data import splits


# ====================================================================================================
# CONCEPTS USED IN THIS FILE (study notes)
# ====================================================================================================
# - unit test: a small automatic check of one function on tiny made-up input. Every test here is a unit test of
#     data/splits.py.
# - regression test: a test that remembers a promise (e.g. 'splits are reproducible') so a later change that breaks
#     it is noticed immediately.
# - assert: a statement that must be true; if false, pytest marks the test as FAILED.
# - fixture (@pytest.fixture): a function that builds shared test input. Tests list its name as an argument and
#     pytest passes the result in. Here `index` builds the fake image table (fresh for each test).
# - tmp_path: a built-in pytest fixture: a fresh empty temporary folder per test. test_save_and_integrity writes the
#     split files there instead of into the real splits/ folder.
# - synthetic (fake) index: a pandas DataFrame with columns id, class, label for 2,700 pretend images with unequal
#     class sizes (200-300 per class), a 10x smaller copy of EuroSAT's class imbalance. No images exist.
# - pandas DataFrame / value_counts(normalize=True): a table; value_counts(normalize=True) gives the SHARE of each
#     value (e.g. 0.70 of the rows are 'train').
# - train / val / test split (70/15/15): train = images the model learns from, val = used to choose settings and
#     stop training, test = used once for the final score.
# - stratified split: the split is done class by class so every part keeps almost the same class mix as the whole
#     data (scikit-learn train_test_split(..., stratify=label)). Without it a rare class could vanish from val.
# - pytest.approx: compares floats with a tolerance (abs=0.01 means within one percentage point; rel=0.05 means
#     within 5% of the value).
# - data leakage / disjoint sets: an image appearing in two splits (e.g. train and test) would inflate test scores.
#     'Disjoint' means no shared ids; splits.verify() asserts this.
# - determinism / seed: a seed starts the random generator so the same 'random' choice repeats. Same seed -> same
#     split; different seed -> different split.
# - nested subsets: the labelled subsets 1% <= 10% <= 100% are built from one shuffled order per class, so each
#     smaller subset is contained in the bigger one; going to more labels only ADDS images.
# - labelled subset / label fraction: only a fraction (1%, 10%, 100%) of the train images is treated as labelled.
#     Subsets come from train only, never from val or test.
# - subset seed: each subset seed gives an independent random draw, so results can be averaged over seeds.
# - manifest.json and SHA-256 hash: a SHA-256 hash is a fingerprint of a file's bytes; any change gives a different
#     hash. save() records the hash of every written file in manifest.json.
# - integrity check: check_integrity() recomputes the hashes and raises RuntimeError if a split file was edited,
#     protecting the official splits from accidental changes.
# - pytest.raises: passes only if the code inside raises the named error.

# ====================================================================================================
# Q&A ABOUT THIS FILE (viva practice)
# ====================================================================================================
# Q: Which part of the project does this file protect?
# A: eurosat_le/data/splits.py: the train/val/test split and the nested labelled subsets. Every experiment depends
#    on these, so leakage or non-reproducibility here would make all results unreliable.
# Q: How do you run it?
# A: pytest tests/test_splits.py from the repo root (with eurosat_le importable).
# Q: Why does it need no dataset?
# A: The `index` fixture builds a fake table of ids, class names and labels. The split code only needs that table,
#    not pixels. The saving test writes to tmp_path, so the committed splits/ folder is never touched.

# Q: What does the `index` fixture do?
# A: Returns a DataFrame with 2,700 rows (id like 'Forest/Forest_1', class name, label 0-9). Class sizes are [300,
#    300, 300, 250, 250, 200, 250, 300, 250, 300], mimicking EuroSAT's imbalance scaled down 10x.
# Q: Why is it enough?
# A: Splitting only needs the label column and unique ids. Unequal class sizes make the stratification test
#    meaningful: with equal sizes a lazy split would also pass.
@pytest.fixture
def index():
    """Fixture: a fake image index (id, class, label) with 2,700 images and different sizes per class."""
    # Mimic EuroSAT's imbalance: 2000-3000 samples per class, scaled down 10x.
    # Number of images in each of the 10 classes (same order as CLASSES).
    sizes = [300, 300, 300, 250, 250, 200, 250, 300, 250, 300]
    # One row per image: an id such as "Forest/Forest_1", its class name and its numeric label (0-9).
    rows = [(f"{c}/{c}_{i + 1}", c, k) for k, (c, n) in enumerate(zip(CLASSES, sizes)) for i in range(n)]
    return pd.DataFrame(rows, columns=["id", "class", "label"])


# Q: What does this test check?
# A: make_split gives about 70% train, 15% val and 15% test (each within 0.01), and splits.verify(df, {}) passes.
# Q: What bug would make it fail?
# A: Wrong fractions (e.g. forgetting that validation is taken from the remainder and must be rescaled to
#    0.15/0.85), duplicated ids, a split name left empty, or any image in two splits (verify raises AssertionError).
# Q: Which functions does it exercise?
# A: splits.make_split (two stratified train_test_split calls) and splits.verify (with an empty subsets dict, so
#    only the split checks run).
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


# Q: What does this test check?
# A: In train, val and test, every class's share differs from its share in the whole index by less than 0.01.
# Q: Why does it matter?
# A: Otherwise val/test could over- or under-represent some land-cover classes, and macro-F1 would not be comparable
#    between splits.
# Q: How are the shares compared?
# A: value_counts(normalize=True).sort_index() on the label column gives a vector of 10 shares for the whole index
#    and for each split; the maximum absolute difference must be below 0.01.
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


# Q: What does this test check?
# A: The same seed (42) gives an identical split twice, and a different seed (7) gives a different one.
# Q: Why does it matter?
# A: Every model must be evaluated on exactly the same test images; the split must be reproducible. The second
#    assert proves the seed really influences the result (the function is not ignoring it).
def test_split_is_deterministic(index):
    """The same seed must always give the same split, and a different seed must give a different one."""
    # Build the split twice with the same seed 42.
    a, b = splits.make_split(index, seed=42), splits.make_split(index, seed=42)
    # Identical assignment of every image to train/val/test.
    assert a["split"].equals(b["split"])
    # A different seed (7) must not reproduce the same assignment (otherwise the seed would do nothing).
    assert not a["split"].equals(splits.make_split(index, seed=7)["split"])


# Q: What does this test check?
# A: For subset seeds 0 and 1: the 1%, 10% and 100% subsets are nested, the 100% subset equals the train ids, the
#    10% subset has about 10% of train (within 5%), and seeds 0 and 1 give different 1% subsets.
# Q: What bug would make it fail?
# A: A subset containing a val or test image (leakage), subsets not nested (a 1% image missing from the 10% set),
#    wrong sizes, or the seed having no effect.
# Q: Which function does it exercise?
# A: splits.build_all, which calls make_split, make_label_subsets for each seed, and verify (so verify's own
#    assertions also run here). The chained comparison uses `<=` on sets, meaning 'is a subset of'.
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


# Q: What does this test check?
# A: A 0.1% subset still contains at least one image of each of the 10 classes.
# Q: Why does it matter?
# A: 0.1% of the fake index's ~1,890 train images is only about 2 images, and per class it would round to 0. The
#    code takes max(1, round(fraction * n_class)) per class, so no class disappears; a missing class would make
#    training and F1 meaningless.
# Q: How is it checked?
# A: Labels of the subset ids are looked up in the split table and labels.nunique() must equal len(CLASSES) = 10.
def test_tiny_fraction_keeps_every_class(index):
    """Even a very small subset (0.1%) must contain at least one image of every class."""
    split_df = splits.make_split(index)
    # 0.1% of the training images would be only ~2 images; the rule "at least 1 per class" must still hold.
    sub = splits.make_label_subsets(split_df, fractions=[0.001, 1.0], seed=0)
    # Look up the labels of the images in the 0.1% subset and count the different classes.
    labels = split_df.set_index("id").loc[sub[0.001], "label"]
    assert labels.nunique() == len(CLASSES)


# Q: What does this test check?
# A: save() writes split.csv, subset files and manifest.json; check_integrity() then passes; load_subset reads back
#    exactly the saved ids; after one subset file is overwritten with 'tampered', check_integrity raises
#    RuntimeError.
# Q: Why does it matter?
# A: The official splits are committed to git. If someone regenerated or edited them by accident, all result
#    comparisons would silently become invalid; the SHA-256 hashes in manifest.json catch this.
# Q: Why tmp_path?
# A: The files are written into a temporary folder so the real splits/ folder is not changed or damaged by the
#    tampering step.
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
