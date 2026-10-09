"""Tests for scripts/run_part1.py: flags and the command list (nothing is run).

Runs under pytest or directly: python tests/test_run_part1.py
"""

import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_part1 as rp  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "pipeline" / "config.yaml"


def test_cli_flags():
    assert rp.cli_flags({"bleed_max_box_edge": 0.3, "wrong_region_fix": False, "fin_threshold": None,
                         "bleed_fill": [0.2, 0.9], "x": True}) == [
        "--bleed-max-box-edge", "0.3", "--no-wrong-region-fix", "--bleed-fill", "0.2", "0.9", "--x"]


def test_commands_follow_the_config():
    cfg = yaml.safe_load(CONFIG.read_text())
    with tempfile.TemporaryDirectory() as tmp:
        cfg["processed_dir"] = tmp
        cmds = rp.commands(cfg, rp.STEPS)
        assert [c[0] for c in cmds] == ["labels", "segment full_body", "segment Röntgen", "merge full_body",
                                        "merge Röntgen", "split", "manifest"]
        argv = {step: a for step, a, _ in cmds}
        assert "--bleed-max-box-edge" not in argv["segment full_body"] and "--resume" in argv["segment full_body"]
        assert argv["segment Röntgen"][-2:] == ["--bleed-max-box-edge", "0.3"]
        assert "--qa-rule" not in argv["merge full_body"] and argv["merge Röntgen"][-2:] == ["--qa-rule", "B"]
        assert str(Path(tmp) / "labels_thesis.csv") in argv["split"] and str(Path(tmp) / "split" / "thesis_seed0") in argv["manifest"]
        # an existing curated dir is not merged again
        (Path(tmp) / "segmented" / "thesis" / "Röntgen").mkdir(parents=True)
        skips = {step: skip for step, _, skip in rp.commands(cfg, ["merge"])}
        assert skips["merge full_body"] is None and "exists" in skips["merge Röntgen"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
