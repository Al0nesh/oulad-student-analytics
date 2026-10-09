import pandas as pd
import pytest

from oulad_analytics import __version__
from oulad_analytics.cli import main


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_features_command(synthetic_dir, tmp_path):
    out = tmp_path / "f.csv"
    assert (
        main(["features", "--data-dir", str(synthetic_dir), "--cutoff", "35", "--out", str(out)])
        == 0
    )
    df = pd.read_csv(out)
    assert "clicks_w5" in df.columns and "clicks_w6" not in df.columns


def test_train_command_writes_outputs(synthetic_dir, tmp_path):
    assert main(["train", "--data-dir", str(synthetic_dir), "--out-dir", str(tmp_path)]) == 0
    for name in (
        "metrics.csv",
        "feature_importance.csv",
        "summary.md",
        "weekly_clicks.png",
        "feature_importance.png",
        "model_comparison.png",
    ):
        assert (tmp_path / name).stat().st_size > 0
    assert "| model |" in (tmp_path / "summary.md").read_text(encoding="utf-8")


def test_synthetic_and_demo_commands(tmp_path):
    assert main(["synthetic", "--out", str(tmp_path / "data"), "--n-students", "50"]) == 0
    assert (tmp_path / "data" / "studentVle.csv").exists()
    assert main(["demo", "--out-dir", str(tmp_path / "demo"), "--n-students", "200"]) == 0
    assert "synthetic" in (tmp_path / "demo" / "summary.md").read_text(encoding="utf-8")


def test_bad_data_dir_returns_error_code(tmp_path, capsys):
    assert main(["train", "--data-dir", str(tmp_path / "missing")]) == 2
    assert "error" in capsys.readouterr().err
