# Semi-supervised learning: pseudo-labelling (Phase 5)

Semi-supervised results are reported **separately** from the supervised grid. Each student is compared with its own supervised teacher: same model, input type, label fraction and seed.

## Method (self-training with confidence-filtered pseudo-labels)

1. **Teacher.** The finished supervised run, for example `resnet50_ms_1pct_seed0`. At the end of training it saves logits for the *unlabelled pool* (`preds_unlabelled.npz`). The pool is the train images that are not in its labelled subset: 18,708 images at 1%, 17,009 at 10%. These logits come from the weights selected on validation macro-F1.
2. **Threshold τ (validation only).** The teacher's validation predictions are sorted by softmax confidence. For τ ∈ {0.50, 0.51, …, 0.99}, we measure the fraction of validation predictions with confidence ≥ τ that are correct (their *precision*). τ is the smallest value with precision ≥ 0.95, which keeps as many pseudo-labels as possible at a known quality. If no τ reaches 0.95, the τ with the highest precision is used. The full table is saved as `pseudo_label_thresholds.csv`.
3. **Selection.** Every pool image with confidence ≥ τ gets the teacher's predicted class as its pseudo-label.
   - With `--balance` *(optional)*, each predicted class keeps at most its labelled-set share of the confident images, most confident first. Classes with fewer confident images keep all of theirs. This stops a few easy classes from dominating the pseudo-labelled set.
4. **Student.** A fresh model with the same architecture, starting weights and hyper-parameters as the teacher. It trains on the labelled images plus the pseudo-labelled images, both with the weak augmentation.
   - The budget rule applies to the combined set, so a student usually trains for more steps than its teacher. Training time is reported next to the gains.
   - Model selection and early stopping use the true validation labels, the same as for supervised runs.
5. **More rounds** *(optional)*. A student can teach the next student: `--teacher <run>_pl` creates `<run>_pl2`.

## Leakage safeguards

- Pseudo-labels come only from the unlabelled pool. The code checks that the pool does not overlap the labelled subset; neither ever overlaps val or test.
- τ is chosen on the validation set only. The test set is predicted once, by the selected student weights.
- The true labels of pool images are read **only** after training, by `quality()`. The result goes to `pseudo_label_quality.json` (accuracy overall and per predicted class) for the report's analysis. Nothing in training or selection reads that file.

## Files in a student run (`results/runs/<teacher>_pl/`)

| File | Content |
|---|---|
| `pseudo_labels.csv` | id, pseudo-label, confidence: the exact training targets used |
| `pseudo_label_thresholds.csv` | validation precision and coverage for each τ |
| `pseudo_label_selection.json` | the chosen τ, its validation precision and coverage, counts per class |
| `pseudo_label_quality.json` | *analysis only:* pseudo-label accuracy against the hidden labels |
| `config.yaml`, `metrics.json`, `preds_*.npz`, ... | same as a supervised run (`teacher` and `pseudo_labels` set) |

## Commands

```bash
python scripts/run_grid.py --plan pl --dry-run          # 24 students; [w] = teacher not finished yet
python scripts/run_grid.py --plan pl --max-hours 11
python scripts/pseudo_label.py --teacher resnet50_ms_1pct_seed0 --balance --set name_suffix=plb
python scripts/collect_results.py                       # -> results/summary/semi_supervised.csv
```

## Known limitations (for the report)

- Confirmation bias: the student learns the teacher's mistakes. Measuring pseudo-label accuracy shows how large that effect could be.
- Teacher confidence may be poorly calibrated at 1% labels. Choosing τ on validation corrects for this. But the teacher's weights were also selected on that validation set, so its validation precision can be slightly optimistic. `pseudo_label_quality.json` shows the actual pool accuracy.
- Students train on more images and for more steps than their teachers, so part of any gain may come from the extra compute.
