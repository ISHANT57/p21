"""Model construction tests (no pretrained downloads)."""

import pytest
import torch

from eurosat_le.constants import MS_RGB_IDX
from eurosat_le.models import ARCHS, adapt_input_conv, build_model
from eurosat_le.train.config import load_config, parse_set
from eurosat_le.train.engine import cosine_with_warmup


@pytest.mark.parametrize("name", list(ARCHS))
@pytest.mark.parametrize("c", [3, 13])
def test_forward_shape(name, c):
    model = build_model(name, c, pretrained=False).eval()
    with torch.no_grad():
        assert model(torch.randn(2, c, 64, 64)).shape == (2, 10)


def test_adapt_mean_maps_rgb_filters_to_b04_b03_b02():
    w = torch.randn(8, 3, 7, 7)
    new = adapt_input_conv(w, 13, mode="mean")
    s = 3 / 13
    for rgb_channel, band in enumerate(MS_RGB_IDX):
        assert torch.allclose(new[:, band], w[:, rgb_channel] * s)
    others = [b for b in range(13) if b not in MS_RGB_IDX]
    assert torch.allclose(new[:, others], (w.mean(1, keepdim=True) * s).expand(-1, 10, -1, -1))


def test_adapt_zero_reproduces_rgb_model_on_rgb_bands():
    w = torch.randn(8, 3, 3, 3)
    new = adapt_input_conv(w, 13, mode="zero")
    x13 = torch.randn(1, 13, 16, 16)
    x3 = x13[:, MS_RGB_IDX]
    assert torch.allclose(torch.nn.functional.conv2d(x13, new), torch.nn.functional.conv2d(x3, w), atol=1e-5)


def test_adapt_rgb_is_identity():
    w = torch.randn(4, 3, 3, 3)
    assert torch.equal(adapt_input_conv(w, 3), w)


def test_classifier_upsamples_to_224():
    seen = {}
    model = build_model("resnet50", 3, pretrained=False)
    model.backbone.conv1.register_forward_hook(lambda m, i, o: seen.update(shape=i[0].shape))
    model.eval()(torch.randn(1, 3, 64, 64))
    assert seen["shape"][-1] == 224


def test_config_overrides_and_run_name():
    cfg = load_config(None, model="resnet50", modality="ms", fraction=0.01, seed=2, **parse_set(["lr=3e-4", "amp=false"]))
    assert cfg.lr == 3e-4 and cfg.amp is False and cfg.run_name == "resnet50_ms_1pct_seed2"
    with pytest.raises(ValueError):
        load_config(None, not_a_field=1)


def test_cosine_with_warmup_schedule():
    opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
    sched = cosine_with_warmup(opt, total_steps=100, warmup_steps=10)
    lrs = []
    for _ in range(100):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] == pytest.approx(0.1) and lrs[9] == pytest.approx(1.0)
    assert lrs[-1] < 0.01 and all(a >= b for a, b in zip(lrs[10:], lrs[11:]))
