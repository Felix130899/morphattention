"""Unit tests for scripts/flag_bleed.py. Runs under pytest or directly:
python tests/test_flag_bleed.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import flag_bleed as fb  # noqa: E402


def rec(name, areas, border, status="ok"):
    return {"file_name": name, "status": status, "width": 100, "height": 100, "qa": {"flags": []},
            "instances": [{"area": a, "touches_border": b} for a, b in zip(areas, border)]}


def test_rule_needs_high_coverage_and_border():
    assert fb.is_suspect(rec("a", [8000], [True]), 0.7)
    assert not fb.is_suspect(rec("b", [8000], [False]), 0.7)       # big but not at the border
    assert not fb.is_suspect(rec("c", [5000], [True]), 0.7)        # at the border but small
    assert fb.is_suspect(rec("d", [4000, 3500], [False, True]), 0.7)  # coverage sums instances


def test_flag_run_sorts_and_skips_failed_images():
    with tempfile.TemporaryDirectory() as d:
        lines = [rec("low", [7500], [True]), rec("high", [9500], [True]), rec("small", [100], [True]),
                 {"file_name": "skip", "status": "unreadable"}]
        (Path(d) / "annotations.jsonl").write_text("\n".join(json.dumps(l) for l in lines) + "\n")
        assert [s[0] for s in fb.flag_run(Path(d), 0.7)] == ["high", "low"]


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in list(globals().items()) if name.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"  ok  {name}")
    print(f"All {len(tests)} flag_bleed tests passed.")
