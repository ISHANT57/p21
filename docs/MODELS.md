# Models and training

## Architectures

| Name | Backbone | Pretraining | Input size | Params (RGB / 13-band) |
|---|---|---|---|---|
| `scratch_cnn` | Small ResNet-style CNN: 4 stages × 2 BasicBlocks, widths 32-64-128-256, no stride in the stem | none | 64 px | 2.80M / 2.80M |
| `resnet50` | timm `resnet50.a1_in1k` | ImageNet-1k | 224 px | 23.5M / 23.6M |
| `vit_small` | timm `vit_small_patch16_224.augreg_in21k_ft_in1k` | ImageNet-21k, then 1k | 224 px | 21.7M / 22.7M |
| `vit_scratch` *(optional)* | timm ViT-Tiny, patch 8 (64 tokens) | none | 64 px | 5.4M / 5.5M |

Every model sits inside `Classifier`, which upsamples the 64×64 input to the backbone's input size on the GPU (bilinear) before the forward pass. The data pipeline itself always stays at 64 px.

## Using pretrained RGB weights with 13 bands

ImageNet models expect 3 input channels. For 13-band input, the first layer is replaced: `conv1` in ResNet-50, or the patch-embedding projection in the ViT. `adapt_input_conv` (in `src/eurosat_le/models/__init__.py`) builds the new filters from the pretrained RGB filters `W_R, W_G, W_B`.

**Default: `init_mode: mean`**
- The R, G and B filters go to the matching Sentinel-2 bands B04, B03 and B02 (file channels 3, 2, 1).
- Each of the other 10 bands gets the mean filter `(W_R + W_G + W_B) / 3`.
- All 13 filters are then multiplied by 3/13. Every band is standardised to unit variance, so this keeps the summed first-layer response at about the size the pretrained network saw with 3 channels.

**Alternative: `init_mode: zero`** *(ablation)*
- RGB filters are copied unchanged; the other 10 bands start at zero.
- At the start of training, the 13-band network behaves exactly like the RGB network on bands B04/B03/B02. It has to learn from scratch how to use the extra bands.

**Limitations to state in the report:**
- The extra bands start with filters that respond to *visible* texture and brightness. Nothing in ImageNet pretraining knows what near-infrared, red-edge or SWIR mean physically, so the spectral information is learned only during fine-tuning. With 1% of labels (191 images) there is little data for that.
- The mean init also scales down the RGB filters by 3/13, so the initial RGB features are weaker than in the RGB model. The `zero` init avoids this but starts with the extra bands ignored.
- ImageNet models were trained with ImageNet colour normalisation. We use per-band EuroSAT train statistics for both RGB and 13-band input, so the two input types are processed the same way. Fine-tuning absorbs the difference, but the very first forward pass differs slightly from the original pretraining.
- A model pretrained on 13-band Sentinel-2 data (e.g. SSL4EO-S12) would avoid these problems. It is outside the project brief and is mentioned as future work.

## Training (`src/eurosat_le/train/engine.py`)

- **Optimiser.** AdamW. No weight decay on biases, normalisation layers, position embeddings or the class token.
- **Schedule.** Linear warm-up, then cosine decay to 0, stepped every batch.
- **Loss.** Cross-entropy (label smoothing is configurable, default 0). Mixed precision (fp16 autocast with GradScaler) on CUDA.
- **Model selection.** The model is evaluated on the validation set after every epoch. The weights with the best **validation macro-F1** are kept, and training stops early after `patience` epochs without improvement. The test set is predicted only once, with the selected weights.
- **Seeds.** The `seed` of a run fixes both the labelled-subset draw (`splits/subsets/*_seed<s>.txt`) and the initialisation, shuffling and augmentation.
- **Outputs** go to `results/runs/<model>_<input>_<pct>pct_seed<s>/`:
  - `config.yaml`, `env.json` (versions, device, git commit), `history.csv`
  - `preds_{val,test}.npz` (logits, labels, ids), so Phase 6 can recompute every metric without retraining
  - `metrics.json`, written last; a finished run is skipped when re-launched
  - For runs with 1% or 10% labels, `preds_unlabelled.npz` holds the trained model's predictions on the unlabelled training pool. These are the teacher pseudo-labels for Phase 5, so no checkpoint has to be stored.

Starting hyper-parameters are in `configs/*.yaml`. In Phase 4 they are tuned lightly on the validation set only, then frozen.
