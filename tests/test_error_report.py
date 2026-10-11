"""Unit tests for scripts/error_report.py.

Runs under pytest or directly: python tests/test_error_report.py
"""

import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import error_report as er  # noqa: E402
import review_masks as rm  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
THESIS = rm.load_review_config(REPO / "pipeline" / "config_thesis_nhm.yaml")
NEUTRAL = rm.load_review_config(REPO / "pipeline" / "config.yaml")
CATS = [{"name": c["name"], "text": c["text"]} for c in THESIS["categories"]]  # what the test reviews judge with


def make_domain(tmp, domain, verdicts, settings=None, categories=CATS, reviewer="Neo"):
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
    rm.start_review(review, categories, reviewer, "test.yaml")
    items = rm.draw_sample(review, rm.load_items(run), len(names), 0, run)
    decisions = {}
    for item, (verdict, cats) in zip(items, verdicts):
        rm.record_decision(review, decisions, item["file_name"], verdict, cats,
                           allowed=[c["name"] for c in categories])
    return review


def h2s(page):
    return tuple(re.findall(r"<h2>(.*?)</h2>", page))


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
        a, b = (er.domain_stats(d, 0.10, THESIS["labels"]) for d in (photos, xrays))
        assert (a["label"], a["bad"], a["judged"], a["accepted"]) == ("Photos", 1, 10, True)
        assert (b["label"], b["bad"], b["drop"], b["accepted"]) == ("X-rays", 3, 1, False)
        assert a["cats"]["bleed"]["k"] == 1 and a["cats"]["fin_cut"]["k"] == 1 and b["cats"]["wrong_object"]["k"] == 1
        assert a["est"][0] == 1 and a["complete"]

        page = er.render([a, b], THESIS, 0.10, "2026-01-01", "cmd")
        assert "Photos: accepted." in page and "X-rays: not accepted." in page
        assert "0.35 (Photos) / 0.3 (X-rays)" in page  # per-domain setting values differ
        assert "f0.jpg" not in page  # no file names (they carry catalog numbers)
        assert h2s(page) == er.SECTIONS and "two-proportion z-test" in page
        assert "One reviewer (Neo)." in page and "NHM Wien" in page and "fin_cut is an upper bound" in page
        one = er.render([a], THESIS, 0.10, "2026-01-01", "cmd")  # same sections with one domain
        assert h2s(one) == er.SECTIONS and "Not applicable: needs exactly 2 domains, this report has 1." in one


def test_own_categories_with_the_neutral_config():
    own = [{"name": "missing_tail", "text": "tail cut off"}, {"name": "bleed", "text": "background in the mask"}]
    with tempfile.TemporaryDirectory() as tmp:
        review = make_domain(Path(tmp), "photos", [("ok", [])] * 8 + [("fix", ["missing_tail"])] * 2,
                             {"bleed_outside_frac": 0}, categories=own, reviewer="Ana")
        s = er.domain_stats(review, NEUTRAL["max_rate"], NEUTRAL["labels"])
        assert s["label"] == "Photos" and s["cats"]["missing_tail"]["k"] == 2 and s["rate"] == 0.2
        page = er.render([s], NEUTRAL, NEUTRAL["max_rate"], "2026-01-01", "cmd")
        assert h2s(page) == er.SECTIONS
        assert "tail cut off" in page and "no advice in the config" in page  # own category, no fixes
        assert "<code>bleed_outside_frac</code></td><td class='left'>0</td>" in page  # neutral advice + value used
        assert "One reviewer (Ana)." in page and "<footer>" not in page
        for thesis_word in ("Neo", "NHM", "fin_cut", "missed_fish", "X-rays"):
            assert thesis_word not in page, thesis_word
        assert "missed_object" not in page  # only the review's own categories are shown



def test_run_settings_of_merged_runs():
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp)
        cfg = {"merged": {}, "base_config": {"settings": {"a": 1, "b": 1}}, "patch_config": {"settings": {"b": 2}}}
        (run / "run_config.json").write_text(json.dumps(cfg))
        assert er.run_settings(run) == {"a": 1, "b": 2}
        cfg["patch_config"] = None  # merge_runs.py drop-only mode
        (run / "run_config.json").write_text(json.dumps(cfg))
        assert er.run_settings(run) == {"a": 1, "b": 1}

if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
