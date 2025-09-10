import os
import yaml
import glob
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def test_config_loads():
    cfg_path = os.path.join(ROOT, "config.yaml")
    assert os.path.exists(cfg_path), "config.yaml is missing"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    assert "paths" in cfg and "results_dir" in cfg["paths"]

def test_directories_present():
    for d in ["src", "data", "results", "outputs"]:
        assert os.path.isdir(os.path.join(ROOT, d)), f"Missing folder: {d}"

@pytest.mark.xfail(strict=False, reason="Enable when demo script is added")
def test_demo_creates_output(tmp_path):
    """
    Optional: mark as xfail until you add a demo script that writes to results/.
    When ready, set strict=True and remove xfail.
    """
    # Example contract you can implement later:
    # python scripts/run_demo.py --out results/demo.png
    matching = glob.glob(os.path.join(ROOT, "results", "*.png"))
    assert len(matching) > 0, "Expected at least one result image in results/"

