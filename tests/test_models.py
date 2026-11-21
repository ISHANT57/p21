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


def test_adapt_rgb_is_identity():
    """Asking for 3 channels must return the weights unchanged (nothing to adapt)."""
    w = torch.randn(4, 3, 3, 3)
    # torch.equal needs exactly the same values, so the weights are copied, not altered.
    assert torch.equal(adapt_input_conv(w, 3), w)


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
