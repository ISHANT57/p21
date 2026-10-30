# Data and splitting procedure

## Source

EuroSAT (Helber et al., IEEE JSTARS 2019, arXiv:1709.00029). It has 27,000 Sentinel-2 patches of 64×64 pixels in 10 land-use classes.

| Version | Content | Format | Download |
|---|---|---|---|
| RGB | B04, B03, B02 rendered as 8-bit colour | JPEG, 64×64×3 `uint8` | ~95 MB |
| MS | all 13 Sentinel-2 bands | GeoTIFF, 64×64×13 `uint16` | ~2.1 GB zip, ~2.9 GB extracted |

`scripts/download_data.py` downloads both versions from one of two mirrors and verifies each archive against a fixed hash before extracting it:

- Zenodo record 7711810, the authors' release (MD5).
- The torchgeo Hugging Face mirror at a pinned commit (SHA-256).

The URLs and hashes are in `src/eurosat_le/constants.py`. After extraction, the class folders are at:

```
data/raw/EuroSAT_RGB/<Class>/<Class>_<n>.jpg
data/raw/EuroSAT_MS/<Class>/<Class>_<n>.tif
```

The MS files store **B8A last**: `B01 B02 B03 B04 B05 B06 B07 B08 B09 B10 B11 B12 B8A`. The GeoTIFFs carry no band names, so this order was checked against the data:

- Channels 3, 2 and 1 correlate above 0.99 with the R, G and B channels of the matching JPEG.
- Channel 9 has a train mean of about 12 DN and a maximum of 183. Only the cirrus band B10 is this dark.
- Channels 10 and 11 have means of about 1,820 and 1,120, which match SWIR bands B11 and B12.
- Channel 12 has a mean of about 2,600. It follows B07 and B08, as near-infrared B8A should.

Many code bases assume the standard Sentinel-2 order, with B8A after B08. With that order, B8A, B09, B10, B11 and B12 would all get the wrong name.

## Sample ids

Each sample has an id that does not depend on the input type, for example `AnnualCrop/AnnualCrop_1`. The RGB and MS versions contain exactly the same ids; `make_splits.py` checks this. As a result, one split file serves both inputs, and an RGB model and a 13-band model are always trained and tested on the same scenes.

## Split procedure (`scripts/make_splits.py`)

1. **Train/val/test (70/15/15).** The split is stratified by class with `SPLIT_SEED = 42`, using `sklearn.model_selection.train_test_split`. The 15% test set is taken first, then the validation set is taken from the remaining 85% (val fraction 0.15/0.85). This produces 18,899 / 4,051 / 4,050 images.
2. **Labelled subsets (1%, 10%, 100% of train).** These are drawn for each subset seed in `SUBSET_SEEDS = [0, 1, 2]`. Within each class, the training ids are shuffled once with `numpy.random.default_rng(seed)`. The subset then takes the first `round(fraction × n_class)` ids, with at least 1 per class.
   - Every fraction is a prefix of the same shuffled order, so the subsets are **nested**: 1% ⊂ 10% ⊂ 100%. This makes the label-efficiency curves easier to compare, because adding labels never swaps out examples already in use.
   - Each subset keeps the class proportions of the training split. Rounding is done per class, so the 1% subset has 191 images (14 to 21 per class) and the 10% subset has 1,890.
   - The three seeds give independent draws, so results can be reported as mean ± std over seeds.
3. **Fixed evaluation sets.** The validation and test sets are the same for every model, input type, label fraction and seed.
4. **Unlabelled pool for semi-supervised learning.** For fraction *f*, the unlabelled pool is the training ids that are *not* in the *f* subset. Their labels are never read during semi-supervised training. Validation and test images are never used as unlabelled data.

## Leakage safeguards

- `splits.verify()` asserts the following: the train, val and test sets do not overlap; every subset is contained in train; the subsets are nested; and every class appears in the 1% subset. It runs every time the splits are generated or checked.
- `splits/manifest.json` stores the SHA-256 hash of every split file, along with class counts. `python scripts/make_splits.py --check` and every training run check these hashes, and stop if any split file has been edited.
- `scripts/inspect_dataset.py` hashes the decoded pixels of every RGB image and reports exact duplicates, including duplicates that fall in different splits.
- **Normalisation statistics** (per-channel mean and std, saved in `splits/norm_stats.json`) are computed **from the training split only**.
  - Known limitation: these are statistics of the full training split, so they include images that are "unlabelled" in the 1% and 10% settings. The statistics never use labels, so this is not label leakage. They also never use validation or test images.

## Preprocessing and loading (Phase 2)

- **Cache.** `scripts/build_cache.py` packs every image into `data/cache/rgb.npy` (`uint8`) or `data/cache/ms.npy` (`uint16`, raw DN), with shape (27000, C, 64, 64) and rows in the order of `split.csv`. A sidecar JSON file stores the hash of `split.csv`. A cache built for a different split is rejected.
- **Normalisation.** RGB is divided by 255; 13-band data stays in raw DN. Each channel is then standardised with the train-only mean and std from `splits/norm_stats.json`. On the full train split this gives mean 0 and std 1 (within 0.001) for every band.
- **Augmentation** (`src/eurosat_le/data/transforms.py`). Every operation works the same for 3 or 13 channels, so RGB and 13-band runs share one policy.
  - *weak*: random flip and 90° rotation (overhead images have no natural "up"), plus a random resized crop (scale 0.75–1.0) back to 64 px.
  - *strong*: *weak*, plus random brightness/contrast (one gain and offset shared by all bands), Gaussian noise and blur (each with probability 0.5), plus cutout (filled with 0, which is the band mean). This is used only for the unlabelled data in semi-supervised training.
  - Evaluation and pseudo-label prediction use no augmentation.
- **Resolution.** The data pipeline always outputs 64×64. Models that expect 224×224 (the ImageNet models and ViT-Small) upsample on the GPU inside the model.
- **Unlabelled pool.** `unlabelled_ids(f, seed)` returns the train ids that are not in the labelled subset. This dataset never returns labels.

## Known dataset caveats (for the report)

- EuroSAT patches come from a limited set of Sentinel-2 tiles, so nearby patches can be very similar. A random split, which is the protocol used in the original paper and followed here, can therefore overestimate how well a model generalises to new regions. A spatially disjoint split would be stricter but would no longer match the original benchmark.
- RGB JPEGs are an 8-bit, compressed rendering of B04/B03/B02. The MS files hold the raw 16-bit values. So part of any RGB-vs-MS difference may come from quantisation and compression, and not only from the extra bands.
- B10 (cirrus, channel 9) carries very little surface information: its train mean is 12 DN and its std is 4.8. It is kept so that all 13 bands are used, as the brief asks. Per-band standardisation scales it up, so an ablation that drops it is worthwhile.
