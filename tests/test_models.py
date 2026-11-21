"""Model construction tests (no pretrained downloads)."""

# pytest: parametrise the tests and check for expected errors; torch: tensors and optimisers.
import pytest
import torch

# MS_RGB_IDX: positions of the red, green and blue bands (B04, B03, B02) inside the 13-band input.
from eurosat_le.constants import MS_RGB_IDX
# ARCHS: all model names; build_model: create a model; adapt_input_conv: widen first-layer weights to 13 bands.
from eurosat_le.models import ARCHS, adapt_input_conv, build_model
# load_config/parse_set: build a training config with overrides; cosine_with_warmup: the learning-rate schedule.
from eurosat_le.train.config import load_config, parse_set
from eurosat_le.train.engine import cosine_with_warmup


# ====================================================================================================
# CONCEPTS USED IN THIS FILE (study notes)
# ====================================================================================================
# - unit test: a check of one function or class on small made-up input. All tests here are unit tests of the model
#     builders, the training config, the learning-rate schedule and the experiment grid.
# - assert: a statement that must be true; if false, pytest marks the test as FAILED.
# - pytest.mark.parametrize: runs one test for several values. test_forward_shape stacks two decorators: every
#     architecture in ARCHS (4) x c in [3, 13] = 8 cases; test_budget_rule_min_steps has 3 cases.
# - pytest.raises: passes only if the code inside raises the named error (ValueError for a config typo).
# - synthetic / fake data: torch.randn tensors stand in for images and for first-layer weights. Only shapes and
#     arithmetic are tested, never accuracy.
# - architecture / ARCHS / build_model: ARCHS lists the models: scratch_cnn (own small CNN), resnet50, vit_small
#     (both from timm, ImageNet-pretrained in real runs) and vit_scratch. build_model(name, in_chans,
#     pretrained=False) creates one wrapped in Classifier.
# - pretrained=False: random weights, so no download of ImageNet weights is needed. Fine for tests that check shapes
#     and structure.
# - forward pass, logits, eval(), torch.no_grad(): forward pass = feeding a batch through the model. The output is
#     raw class scores (logits), shape (N, 10). eval() switches off training-only behaviour (dropout, batch-norm
#     updates); no_grad() skips gradient bookkeeping to save memory.
# - batch shape (N, C, H, W): N images, C channels (3 RGB or 13 multispectral), height and width (64 x 64).
# - multispectral bands B04, B03, B02 and MS_RGB_IDX: in the 13-band Sentinel-2 file the red, green and blue bands
#     are B04, B03, B02 at positions 3, 2, 1 (MS_RGB_IDX = [3, 2, 1]).
# - adapt_input_conv: widens a pretrained 3-channel first-layer filter bank to 13 channels. mode 'mean': RGB filters
#     go to B04/B03/B02, the other 10 bands get the mean RGB filter, everything is scaled by 3/13. mode 'zero': RGB
#     filters copied, other bands start at 0, so the network initially behaves like the RGB model.
# - convolution (conv2d) and filters: the first layer slides small filters over the image; a filter has one weight
#     per input channel. With zero weights, extra bands contribute nothing.
# - torch.allclose vs torch.equal: allclose = equal up to a tiny tolerance (floats); equal = exactly identical.
# - upsampling to 224: the pretrained ResNet/ViT expect 224 x 224 images; the Classifier wrapper enlarges the 64 x
#     64 batch (bilinear) inside the model, so the data pipeline stays at 64.
# - forward hook: a function attached to a layer that PyTorch calls whenever the layer runs. Here it records the
#     shape of the data entering conv1.
# - config (TrainConfig), load_config, parse_set, YAML: TrainConfig is a dataclass with all settings of a run.
#     load_config merges defaults, a YAML file and keyword overrides; parse_set turns 'lr=3e-4' strings into typed
#     values; unknown keys raise ValueError so typos are caught.
# - run_name: built from the settings: <model>_<modality>_<pct>pct_seed<seed>[_<suffix>], e.g.
#     resnet50_ms_1pct_seed2 (0.01 -> '1pct').
# - learning rate, warm-up, cosine decay: the learning rate is the step size of the optimiser. cosine_with_warmup
#     raises it linearly during the first steps (warm-up), then lowers it smoothly towards 0 along a cosine curve.
# - optimiser and scheduler (SGD, LambdaLR): the optimiser updates weights; the scheduler changes its learning rate
#     each step. The test uses a dummy SGD with one parameter to observe only the learning rates.
# - epoch / step / batch / training budget: a step = one batch update; an epoch ends with one validation check.
#     min_steps guarantees a minimum number of steps in total even when 1% labels give only a few batches per pass.
# - steps per epoch (resolve_steps_per_epoch): natural = n_labelled // batch_size; if epochs x natural < min_steps
#     the epoch is stretched to ceil(min_steps / epochs) steps (sampling with replacement), else None.
# - experiment grid (main_grid, tune_grid): main grid = 3 models x 2 input types x 3 label fractions x 3 seeds = 54
#     runs; tuning grid = 3 learning rates per model on one cell (RGB, 10% labels, seed 0), validation only.
# - import inside a test: grid builders are imported inside test_grids because only that test needs them.

# ====================================================================================================
# Q&A ABOUT THIS FILE (viva practice)
# ====================================================================================================
# Q: Which part of the project does this file protect?
# A: Model construction and training set-up: eurosat_le/models (build_model, adapt_input_conv, the 224 px
#    upsampling), train/config.py (config overrides, run names, step budget), train/engine.py (cosine_with_warmup)
#    and train/grid.py (the 54-run and tuning grids).
# Q: How do you run it?
# A: pytest tests/test_models.py from the repo root (needs torch and timm installed).
# Q: Why are no dataset and no pretrained download needed?
# A: Models are built with pretrained=False and fed random tensors; config and grid tests are pure Python objects.
#    The grid test only reads the committed configs/<model>.yaml files.

# Q: What does test_forward_shape check?
# A: For every architecture in ARCHS and for 3 and 13 input channels, a batch of 2 random 64 x 64 images gives an
#    output of shape (2, 10): ten class scores per image. 8 cases.
# Q: What bug would make it fail?
# A: A first layer that does not accept 13 channels, a wrong resizing to the backbone's input size, a wrong number
#    of output classes, or a broken model constructor.
# Q: How is the input built?
# A: build_model(name, c, pretrained=False).eval() with random weights, and torch.randn(2, c, 64, 64) under
#    torch.no_grad(). Values do not matter, only that the shapes work.
# Every architecture, with both RGB (3 channels) and multispectral (13 channels) input.
@pytest.mark.parametrize("name", list(ARCHS))
@pytest.mark.parametrize("c", [3, 13])
def test_forward_shape(name, c):
    """Every model must accept a batch of 64x64 images with 3 or 13 channels and output 10 class scores each."""
    # Random weights (pretrained=False, so nothing is downloaded); eval() switches off training behaviour.
    model = build_model(name, c, pretrained=False).eval()
    # no_grad: no gradients needed, which saves memory. A batch of 2 fake images must give a (2, 10) output.
    with torch.no_grad():
        assert model(torch.randn(2, c, 64, 64)).shape == (2, 10)


# Q: What does this test check?
# A: In 'mean' mode the widened filter bank has the R, G, B filters at bands B04, B03, B02 (positions 3, 2, 1), each
#    multiplied by 3/13, and the other 10 bands equal the mean of the three RGB filters times 3/13.
# Q: Why does it matter?
# A: The pretrained network knows red, green and blue. They must land on the right Sentinel-2 bands, and the 3/13
#    scaling keeps the total first-layer response about as large as with 3 channels.
# Q: How is the input built?
# A: Fake first-layer weights torch.randn(8, 3, 7, 7) (8 filters, 3 channels, 7x7 kernel) and adapt_input_conv(w,
#    13, mode='mean'); the 7x7 size mimics ResNet-50's conv1.
def test_adapt_mean_maps_rgb_filters_to_b04_b03_b02():
    """In "mean" mode the red/green/blue filters go to bands B04/B03/B02 and all other bands get their mean."""
    # Fake first-layer weights: 8 filters, 3 input channels (R, G, B), 7x7 kernel.
    w = torch.randn(8, 3, 7, 7)
    new = adapt_input_conv(w, 13, mode="mean")
    # All weights are scaled by 3/13 so the summed response keeps its size with 13 inputs instead of 3.
    s = 3 / 13
    # Each RGB filter must sit in its own band position, scaled by s.
    for rgb_channel, band in enumerate(MS_RGB_IDX):
        assert torch.allclose(new[:, band], w[:, rgb_channel] * s)
    # The other 10 bands must all equal the average of the R, G and B filters, scaled by s.
    others = [b for b in range(13) if b not in MS_RGB_IDX]
    assert torch.allclose(new[:, others], (w.mean(1, keepdim=True) * s).expand(-1, 10, -1, -1))


# Q: What does this test check?
# A: In 'zero' mode, convolving a 13-band image with the widened filters gives the same result as convolving only
#    its R, G, B bands with the original filters.
# Q: Why does it matter?
# A: It proves the ablation starts exactly as the RGB model: extra bands have zero weights and add nothing until
#    they are learned.
# Q: How is it checked?
# A: x13 = random 13-band image; x3 = x13[:, MS_RGB_IDX] (its R, G, B bands); torch.nn.functional.conv2d on both
#    must agree (atol 1e-5).
def test_adapt_zero_reproduces_rgb_model_on_rgb_bands():
    """In "zero" mode the widened layer must behave exactly like the original RGB layer on the RGB bands."""
    # Fake RGB filters (8 filters, 3x3), widened to 13 channels with zeros for the extra bands.
    w = torch.randn(8, 3, 3, 3)
    new = adapt_input_conv(w, 13, mode="zero")
    # A fake 13-band image, and the 3-band image made from only its R, G, B bands.
    x13 = torch.randn(1, 13, 16, 16)
    x3 = x13[:, MS_RGB_IDX]
    # Convolving the 13-band image with the new filters must equal convolving the RGB image with the old ones,
    # because the extra bands are multiplied by zero weights.
    assert torch.allclose(torch.nn.functional.conv2d(x13, new), torch.nn.functional.conv2d(x3, w), atol=1e-5)


# Q: What does this test check?
# A: adapt_input_conv(w, 3) returns weights exactly equal to the input (torch.equal).
# Q: Why does it matter?
# A: RGB models must keep their pretrained first layer untouched. The function returns a clone, so the original is
#    not modified either.
def test_adapt_rgb_is_identity():
    """Asking for 3 channels must return the weights unchanged (nothing to adapt)."""
    w = torch.randn(4, 3, 3, 3)
    # torch.equal needs exactly the same values, so the weights are copied, not altered.
    assert torch.equal(adapt_input_conv(w, 3), w)


# Q: What does this test check?
# A: A 64 x 64 image fed to the ResNet-50 model reaches its first convolution with width 224.
# Q: Why does it matter?
# A: The pretrained ResNet was trained at 224 px. Classifier.forward enlarges the batch on the fly (bilinear
#    interpolation) so the data pipeline can stay at 64 px; if this step were missing the pretrained filters would
#    see images of the wrong size.
# Q: How is it observed?
# A: A forward hook on model.backbone.conv1 stores the shape of its input in a dict; then one fake image is run
#    through model.eval().
def test_classifier_upsamples_to_224():
    """The ResNet must enlarge the 64x64 input to 224x224 before its first layer (the size it was pretrained on)."""
    # Dictionary the hook below fills with the shape of the data it sees.
    seen = {}
    model = build_model("resnet50", 3, pretrained=False)
    # A forward hook is a function called each time the first conv layer runs; it records the input's shape.
    model.backbone.conv1.register_forward_hook(lambda m, i, o: seen.update(shape=i[0].shape))
    # Run one fake 64x64 image through the model so the hook fires.
    model.eval()(torch.randn(1, 3, 64, 64))
    # The last dimension (width) seen by the first layer must be 224, proving the upsampling happened.
    assert seen["shape"][-1] == 224


# Q: What does this test check?
# A: Overrides 'lr=3e-4' and 'amp=false' (from parse_set) are applied with the right types (float 0.0003, boolean
#    False); the run name is 'resnet50_ms_1pct_seed2'; an unknown key raises ValueError.
# Q: Why does it matter?
# A: YAML 1.1 reads 3e-4 as a string, so the config code converts numeric fields; wrong types or silently ignored
#    typos would run experiments with the wrong settings. Run names identify result folders, so they must be exact.
# Q: Which functions does it exercise?
# A: train/config.py: load_config (with _coerce), parse_set and TrainConfig.__post_init__ (default run_name).
def test_config_overrides_and_run_name():
    """Config values can be overridden from the command line, the run name is built from the settings, typos fail."""
    # No config file (None); keyword settings plus "--set"-style overrides ("lr=3e-4", "amp=false") parsed to
    # real number and boolean types.
    cfg = load_config(None, model="resnet50", modality="ms", fraction=0.01, seed=2, **parse_set(["lr=3e-4", "amp=false"]))
    # Overrides were applied with the right types, and 0.01 labels became "1pct" in the run name.
    assert cfg.lr == 3e-4 and cfg.amp is False and cfg.run_name == "resnet50_ms_1pct_seed2"
    # An unknown setting name must raise an error instead of being silently ignored (catches typos).
    with pytest.raises(ValueError):
        load_config(None, not_a_field=1)


# Q: What does this test check?
# A: With 100 steps and 10 warm-up steps: the first learning rate is 0.1 (a tenth of the base), step 10 (index 9)
#    reaches 1.0, the last is below 0.01, and after warm-up the rate never increases.
# Q: Why does it matter?
# A: Warm-up protects pretrained weights from large early updates; the cosine decay helps training settle. A wrong
#    schedule would change every training result.
# Q: How is it observed?
# A: A dummy SGD optimiser (one parameter, base lr 1.0) is stepped 100 times with the scheduler and its learning
#    rate is recorded before every step.
def test_cosine_with_warmup_schedule():
    """The learning rate must rise linearly during warm-up, peak at the base rate, then decay smoothly towards 0."""
    # A dummy optimiser with base learning rate 1.0 (a single dummy parameter); the scheduler scales this rate.
    opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
    # 100 steps in total, the first 10 of them warm-up.
    sched = cosine_with_warmup(opt, total_steps=100, warmup_steps=10)
    # Record the learning rate used at each of the 100 steps.
    lrs = []
    for _ in range(100):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    # The first step has a small non-zero rate (1/10 of the base); the 10th step (index 9) reaches the full rate.
    assert lrs[0] == pytest.approx(0.1) and lrs[9] == pytest.approx(1.0)
    # By the end the rate is nearly 0, and after warm-up it never goes up again (a - b >= 0 for every pair of steps).
    assert lrs[-1] < 0.01 and all(a >= b for a, b in zip(lrs[10:], lrs[11:]))


# Q: What does test_budget_rule_min_steps check?
# A: With 30 epochs, batch size 64 and min_steps 1500: 191 labelled images (about the 1% subset) and 1890 (about
#    10%) give 50 steps per epoch; 18,899 (the full train set) gives None. Without min_steps the answer is None even
#    for 191 images.
# Q: Why does it matter?
# A: With few labels one pass over the data is only a few batches, so models would get very few updates. The budget
#    rule stretches each epoch (1500 / 30 = 50 steps) so all label fractions get at least the minimum number of
#    updates; with 100% labels one normal pass already exceeds it.
# Q: Show the arithmetic.
# A: 191 // 64 = 2 steps per pass, 30 x 2 = 60 < 1500, so ceil(1500 / 30) = 50. 1890 // 64 = 29, 30 x 29 = 870 <
#    1500, so 50. 18,899 // 64 = 295, 30 x 295 = 8,850 >= 1500, so None.
# (number of labelled images, expected steps per epoch). Small label sets need a fixed step count so that
# training still gets at least min_steps updates; a large set already has enough steps (None = normal pass).
@pytest.mark.parametrize("n_lab,expected", [(191, 50), (1890, 50), (18899, None)])
def test_budget_rule_min_steps(n_lab, expected):
    """The step-budget rule must force a fixed number of steps per epoch for small label sets only."""
    # 30 epochs, batch size 64, and at least 1500 steps in total.
    cfg = load_config(None, epochs=30, batch_size=64, min_steps=1500)
    # 1500 / 30 epochs = 50 steps per epoch when the natural pass is too short, else no change (None).
    assert cfg.resolve_steps_per_epoch(n_lab) == expected
    # Without min_steps there is no minimum, so even the smallest set trains for a normal pass (None).
    assert load_config(None, epochs=30, batch_size=64).resolve_steps_per_epoch(191) is None


# Q: What does this test check?
# A: main_grid() has 54 configs with 54 different run names and its first 18 entries all use seed 0; tune_grid() has
#    9 configs, none of which evaluates the test set, all RGB with 10% labels.
# Q: Why does it matter?
# A: 54 = 3 models x 2 inputs x 3 fractions x 3 seeds; duplicate names would make runs overwrite each other. Seed-
#    first order means an interrupted session still has every cell for the early seeds. The tuning runs must never
#    touch the test set (no tuning on test data = no leakage).
# Q: Which functions does it exercise?
# A: train/grid.py: main_grid and tune_grid, which call load_config on configs/<model>.yaml.
def test_grids():
    """The main experiment grid must have 54 unique runs ordered seed-first, and the tuning grid 9 val-only runs."""
    # Imported here because only this test needs the grid builders.
    from eurosat_le.train.grid import main_grid, tune_grid

    grid = main_grid()
    names = [c.run_name for c in grid]
    # 54 runs, and 54 different names (no run listed twice).
    assert len(grid) == 54 and len(set(names)) == 54
    # Ordering is seed-first: the first 18 runs (one per model/input/fraction cell) all use seed 0.
    assert all(c.seed == 0 for c in grid[:18]), "seed 0 of every cell should come first"
    # The learning-rate sweep: 9 runs, none of which may touch the test set, all on RGB with 10% labels.
    tune = tune_grid()
    assert len(tune) == 9 and not any(c.evaluate_test for c in tune)
    assert all(c.fraction == 0.1 and c.modality == "rgb" for c in tune)
