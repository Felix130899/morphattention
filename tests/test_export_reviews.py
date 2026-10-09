"""Tests for scripts/export_reviews.py on tiny fake review dirs (made-up names).

Runs under pytest or directly: python tests/test_export_reviews.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import export_reviews as er  # noqa: E402


def setup(tmp, names):
    raw = tmp / "raw"
    raw.mkdir()
    for k, n in enumerate(names):
        (raw / n).write_bytes(bytes([k]))
    run = tmp / "run"
    run.mkdir()
    (run / "run_config.json").write_text(json.dumps({"settings": {"raw_dir": str(raw)}}))
    return raw, run


def test_random_sample_export_last_verdict_wins_and_order_kept():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        raw, run = setup(tmp, ["A_b_NMW11111_WEB.jpg", "A_b_NMW22222_WEB.jpg"])
        rev = run / "review_random_seed5"
        rev.mkdir()
        (rev / "sample.json").write_text(json.dumps({"run_dir": str(run), "seed": 5, "population": 2, "n": 2,
                                                     "sample": ["A_b_NMW22222_WEB.jpg", "A_b_NMW11111_WEB.jpg"]}))
        lines = [{"file_name": "A_b_NMW11111_WEB.jpg", "verdict": "fix", "categories": ["bleed"]},
                 {"file_name": "A_b_NMW22222_WEB.jpg", "verdict": "ok", "categories": []},
                 {"file_name": "A_b_NMW11111_WEB.jpg", "verdict": "ok", "categories": []},
                 {"file_name": "Gone_NMW33333_WEB.jpg", "verdict": "drop", "categories": []}]
        (rev / "decisions.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
        header, rows = er.export(rev, er.raw_dir_of(rev, {"": str(run)}), {})
        assert [(r["position"], r["verdict"]) for r in rows] == [(0, "ok"), (1, "ok")]
        assert rows[0]["sha256"] == er.file_sha256(raw / "A_b_NMW22222_WEB.jpg", {})
        assert "seed: 5" in header and "raw_file_missing: 1" in header and "judged: 2" in header
        out = tmp / "out.tsv"
        er.write_tsv(out, header, rows)
        text = out.read_text()
        assert "NMW" not in text and "position\tsha256\tverdict\tcategories\n" in text


def test_compare_export_keeps_runs_and_groups():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        raw, run = setup(tmp, ["A_b_NMW11111_WEB.jpg"])
        rev = tmp / "review_compare"
        rev.mkdir()
        name = "A_b_NMW11111_WEB.jpg"
        (rev / "order.json").write_text(json.dumps({"seed": 0, "runs": {"X": str(run), "Y": str(run)},
                                                    "items": [["Y", name], ["X", name]]}))
        (rev / "groups.csv").write_text(f"file_name,group\n{name},fin_cut\n")
        lines = [{"item": 0, "run": "Y", "file_name": name, "verdict": "fix", "categories": ["fin_cut", "bleed"]},
                 {"item": 1, "run": "X", "file_name": name, "verdict": "ok", "categories": []}]
        (rev / "decisions.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
        header, rows = er.export(rev, er.raw_dir_of(rev, {"X": str(run)}), {})
        assert [(r["position"], r["run"], r["verdict"], r["categories"], r["group"]) for r in rows] == [
            (0, "Y", "fix", "fin_cut,bleed", "fin_cut"), (1, "X", "ok", "", "fin_cut")]
        assert "mode: compare" in header


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
