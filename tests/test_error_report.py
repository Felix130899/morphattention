"""Unit tests for scripts/error_report.py.

Runs under pytest or directly: python tests/test_error_report.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import error_report as er  # noqa: E402
import review_masks as rm  # noqa: E402


def make_domain(tmp, domain, verdicts, settings=None):
    """A fake run dir <tmp>/<domain> with a fully judged random sample; returns the review dir."""
    run = tmp / domain
    (run / "overlays").mkdir(parents=True)
    names = [f"f{k}.jpg" for k in range(len(verdicts))]
    (run / "run_config.json").write_text(json.dumps({"settings": {"raw_dir": "raw", **(settings or {})}}))
    recs = [{"file_name": n, "status": "ok", "qa": {"flags": []}, "instances": [{}]} for n in names]
    (run / "annotations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    for n in names:
        (run / "overlays" / f"{Path(n).stem}.jpg").write_bytes(b"\xff\xd8overlay-bytes")
    review = run / "review_random_seed0"
    review.mkdir()
    items = rm.draw_sample(review, rm.load_items(run), len(names), 0, run)
    decisions = {}
    for item, (verdict, cats) in zip(items, verdicts):
        rm.record_decision(review, decisions, item["file_name"], verdict, cats)
    return review


def test_acceptance_limit_is_inclusive_and_exact():
    assert er.accepted(30, 300, 0.10)
    assert er.accepted(21, 300, 0.10)
    assert not er.accepted(31, 300, 0.10)
    assert er.accepted(1, 10, 0.1)  # 0.1 is not exact in floating point
    assert not er.accepted(0, 0, 0.10)


def test_two_proportion_p():
    assert abs(er.two_proportion_p(21, 300, 30, 300) - 0.19) < 0.01
    assert er.two_proportion_p(5, 100, 5, 100) == 1.0
    assert er.two_proportion_p(0, 100, 0, 100) == 1.0
    assert er.two_proportion_p(0, 100, 30, 100) < 1e-6


def test_nice_max():
    for v, want in [(0.139 * 1.05, (0.15, 0.05)), (0.067, (0.08, 0.02)), (0.05, (0.05, 0.01))]:
        top, step = er.nice_max(v)
        assert abs(top - want[0]) < 1e-9 and step == want[1], (v, top, step)


def test_domain_stats_and_page():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        photos = make_domain(tmp, "full_body", [("ok", [])] * 9 + [("fix", ["bleed", "fin_cut"])],
                             {"mask_threshold": -1.0, "dino_box_threshold": 0.35})
        xrays = make_domain(tmp, "Röntgen", [("ok", [])] * 7 + [("fix", ["missed_fish"])] * 2
                            + [("drop", ["wrong_object"])], {"mask_threshold": -1.0, "dino_box_threshold": 0.3})
        a, b = (er.domain_stats(d, 0.10) for d in (photos, xrays))
        assert (a["label"], a["bad"], a["judged"], a["accepted"]) == ("Photos", 1, 10, True)
        assert (b["label"], b["bad"], b["drop"], b["accepted"]) == ("X-rays", 3, 1, False)
        assert a["cats"]["bleed"]["k"] == 1 and a["cats"]["fin_cut"]["k"] == 1 and b["cats"]["wrong_object"]["k"] == 1
        assert a["est"][0] == 1 and a["complete"]

        page = er.render([a, b], 0.10, "2026-01-01", tmp / "out.html")
        assert "Photos: accepted" in page and "X-rays: not accepted" in page
        assert "0.35 (Photos) / 0.3 (X-rays)" in page  # per-domain setting values differ
        assert "f0.jpg" not in page  # no file names (they carry catalog numbers)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
