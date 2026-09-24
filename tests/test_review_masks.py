"""Unit tests for the decision bookkeeping in scripts/review_masks.py.

No server is started. Runs under pytest or directly:
python tests/test_review_masks.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import review_masks as rm  # noqa: E402


def test_last_decision_wins_and_lists_follow():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        decisions = {}
        rm.record_decision(d, decisions, "a.jpg", "fix")
        rm.record_decision(d, decisions, "b.jpg", "drop")
        rm.record_decision(d, decisions, "a.jpg", "ok")
        rm.record_decision(d, decisions, "b.jpg", None)
        rm.record_decision(d, decisions, "c.jpg", "fix")
        assert decisions == {"a.jpg": "ok", "c.jpg": "fix"}
        assert rm.load_decisions(d) == decisions  # restart sees the same state
        assert (d / "fix.txt").read_text() == "c.jpg\n"
        assert (d / "drop.txt").read_text() == ""


def test_truncated_last_line_is_ignored():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        rm.record_decision(d, {}, "a.jpg", "fix")
        with open(d / "decisions.jsonl", "a") as f:
            f.write('{"file_name": "b.jpg", "verd')
        assert rm.load_decisions(d) == {"a.jpg": "fix"}


def test_unknown_verdict_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            rm.record_decision(Path(tmp), {}, "a.jpg", "maybe")
        except ValueError:
            return
        raise AssertionError("expected ValueError")


def test_load_items_skips_skipped_and_missing_overlays():
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp)
        (run / "overlays").mkdir()
        (run / "run_config.json").write_text(json.dumps({"settings": {"raw_dir": "/raw"}}))
        recs = [
            {"file_name": "b.jpg", "status": "ok", "qa": {"flags": []}, "instances": [{}]},
            {"file_name": "sub/a.png", "status": "ok", "qa": {"flags": ["tiny"]}, "instances": [{}, {}]},
            {"file_name": "c.jpg", "status": "skipped", "reason": "no detections"},
            {"file_name": "d.jpg", "status": "ok", "qa": {"flags": []}, "instances": [{}]},  # no overlay
        ]
        (run / "annotations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs) + '{"trunc')
        for stem in ("a", "b"):
            (run / "overlays" / f"{stem}.jpg").write_bytes(b"")
        items = rm.load_items(run)
        assert [it["file_name"] for it in items] == ["b.jpg", "sub/a.png"]
        assert items[1]["overlay"] == run / "overlays" / "a.jpg"
        assert items[1]["raw"] == Path("/raw/sub/a.png")
        assert items[1]["n_instances"] == 2
        assert [it["file_name"] for it in rm.load_items(run, flagged_only=True)] == ["sub/a.png"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
