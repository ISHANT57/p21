"""Tests for scripts/crosscheck_pipelines.py (comparison with a teammate's independent pipeline)."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("crosscheck_pipelines", REPO_ROOT / "scripts" / "crosscheck_pipelines.py")
cc = importlib.util.module_from_spec(spec)
sys.modules["crosscheck_pipelines"] = cc
spec.loader.exec_module(cc)


def _other_summary() -> pd.DataFrame:
    """Other pipeline: mean accuracy of its three RGB models at 1%, 10%, 100% (made-up numbers)."""
    rows = []
    for model, init, base in (("cnn", "scratch", 76.0), ("resnet50", "imagenet", 88.0), ("vit_s", "imagenet", 90.0)):
        for f, add in ((0.01, 0.0), (0.1, 5.0), (1.0, 8.0)):
            rows.append(dict(model=model, init=init, frac=f, mean_acc=base + add, std_acc=0.5, n=3))
    return pd.DataFrame(rows)


def _ours() -> pd.DataFrame:
    """Our runs: the same nine RGB cells (accuracy 1 point higher than the other pipeline) plus ResNet-50 13-band runs."""
    rows = []
    for model, base in (("scratch_cnn", 77.0), ("resnet50", 89.0), ("vit_small", 91.0)):
        for f, add in ((0.01, 0.0), (0.1, 5.0), (1.0, 8.0)):
            rows.append(dict(variant="supervised", model=model, modality="rgb", fraction=f, test_acc=(base + add) / 100))
    for f, add in ((0.01, 0.0), (0.1, 5.0), (1.0, 8.0)):  # ResNet-50 on 13 bands: 2 points above its RGB run
        rows.append(dict(variant="supervised", model="resnet50", modality="ms", fraction=f, test_acc=(91.0 + add) / 100))
    return pd.DataFrame(rows)


def test_crosscheck_table_has_nine_rows_and_signed_differences():
    t = cc.crosscheck_table(_other_summary(), _ours())
    assert len(t) == 9 and list(t.columns) == ["Model", "Labels", "Other pipeline", "Ours", "Difference"]
    # every cell was built so that ours is one point above the other pipeline
    assert set(t["Difference"]) == {"+1.0"}
    assert t.iloc[0]["Other pipeline"] == "76.0" and t.iloc[0]["Ours"] == "77.0"


def test_band_control_separates_input_source_from_extra_bands():
    control = pd.DataFrame([dict(input=i, frac=f, mean_acc=a, std_acc=0.3, n=3)
                            for f, (tif, a13) in {0.01: (89.0, 88.5), 0.1: (96.0, 96.0), 1.0: (99.0, 99.5)}.items()
                            for i, a in (("rgb_ms", tif), ("all13", a13))])
    t = cc.band_control_table(control, _other_summary(), _ours())
    r = t.iloc[0]  # 1% labels: JPEG 88.0, GeoTIFF RGB 89.0, 13 bands 88.5
    assert r["GeoTIFF RGB $-$ JPEG RGB"] == "+1.0"   # the input source alone
    assert r["13 bands $-$ GeoTIFF RGB"] == "-0.5"   # the extra bands on top of it
    assert r["13 bands $-$ JPEG (other)"] == "+0.5"
    assert r["13 bands $-$ JPEG (ours)"] == "+2.0"
    assert t.iloc[1]["13 bands $-$ GeoTIFF RGB"] == "+0.0"  # a zero difference never prints as -0.0


def test_diff_never_prints_negative_zero():
    assert cc._diff(-0.04) == "+0.0" and cc._diff(0.96) == "+1.0" and cc._diff(-0.26) == "-0.3"


def test_band_ablation_table_has_one_row_per_input_and_one_column_per_fraction():
    # fake summary of the other pipeline's ablation: every input at every fraction, accuracy = 90 + row index
    keys = list(cc.ABLATION_ROWS)
    ablation = pd.DataFrame([dict(input=k, frac=f, mean_acc=90.0 + i, std_acc=0.5, n=3)
                             for i, k in enumerate(keys) for f in cc.FRACTIONS])
    t = cc.band_ablation_table(ablation)
    assert len(t) == len(keys) and list(t.columns) == ["Input bands", "1%", "10%", "100%"]
    assert t.iloc[0]["1%"] == "90.0 ± 0.5" and t.iloc[-1]["100%"] == "94.0 ± 0.5"


def test_generated_latex_escapes_every_percent_sign_exactly_once(tmp_path):
    """An already escaped percent sign that is escaped again ends the table row and comments out the rest of the line."""
    import subprocess
    # Rebuild the three tables from the saved summaries and our logged runs (committed files), into a temporary folder.
    res = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "crosscheck_pipelines.py"), "--tables", str(tmp_path)],
                         capture_output=True, text=True, env={**__import__("os").environ, "PYTHONPATH": str(REPO_ROOT / "src")})
    assert res.returncode == 0, res.stderr
    for name in ("crosscheck", "band_control", "band_ablation"):
        text = (tmp_path / f"{name}.tex").read_text(encoding="utf-8")
        # no doubled backslash in front of a percent sign, and no bare percent sign at all
        assert "\\\\%" not in text, f"{name}.tex has a doubled backslash before %"
        bare = [i for i, ch in enumerate(text) if ch == "%" and (i == 0 or text[i - 1] != "\\")]
        assert not bare, f"{name}.tex has a bare %"
