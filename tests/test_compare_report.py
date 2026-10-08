"""Unit tests for scripts/compare_report.py.

Runs under pytest or directly: python tests/test_compare_report.py
"""

import json
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import compare_report as cr  # noqa: E402
from test_error_report import make_domain  # noqa: E402

# (B verdict, B categories, D verdict, D categories) per image
PAIRS = [("fix", ["missed_fish"], "ok", []),
         ("fix", ["bleed"], "ok", []),
         ("drop", ["wrong_object"], "fix", ["bleed"]),
         ("fix", ["bleed"], "fix", ["bleed"]),
         ("ok", [], "ok", []),
         ("ok", [], "fix", ["fin_cut"])]


def make_compare(tmp, domain="full_body", pairs=PAIRS, changed=None):
    """Fake run dirs B/<domain>, D/<domain> and a judged compare review dir."""
    names = [f"Fish_NMW{k}_WEB.jpg" for k in range(len(pairs))]
    runs = {r: tmp / r / domain for r in ("B", "D")}
    for run in runs.values():
        run.mkdir(parents=True)
    review = tmp / f"review_compare_BD_{domain}"
    review.mkdir()
    items = [[r, n] for n in names for r in ("D", "B")]
    (review / "order.json").write_text(json.dumps({"seed": 0, "runs": {r: str(p) for r, p in runs.items()},
                                                   "items": items}))
    lines = []
    for k, (n, (bv, bc, dv, dc)) in enumerate(zip(names, pairs)):
        lines.append({"item": 2 * k + 1, "run": "B", "file_name": n, "verdict": bv, "categories": bc})
        lines.append({"item": 2 * k, "run": "D", "file_name": n, "verdict": dv, "categories": dc})
    (review / "decisions.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    if changed:
        (runs["D"] / "changed_vs_B.json").write_text(json.dumps({"changed": changed[0], "compared": changed[1]}))
    return review


def test_mcnemar_exact_p():
    assert cr.mcnemar_exact_p(0, 0) == 1.0
    assert abs(cr.mcnemar_exact_p(6, 0) - 2 / 64) < 1e-12  # 2 * 0.5^6
    assert cr.mcnemar_exact_p(3, 3) == 1.0
    assert cr.fmt_p(1e-9) == "p < 0.001" and cr.fmt_p(0.0312) == "p = 0.031"


def test_domain_stats_outcomes_rates_and_categories():
    with tempfile.TemporaryDirectory() as tmp:
        s = cr.domain_stats(make_compare(Path(tmp)), "B", "D")
        assert (s["key"], s["label"], s["n"]) == ("full_body", "Photos", 6)
        assert s["outcome"] == {"fixed": 2, "still": 2, "both_ok": 1, "broken": 1}
        assert s["rates"]["B"]["k"] == 4 and s["rates"]["D"]["k"] == 3  # drop counts as wrong
        assert s["cats"]["bleed"] == {"before": 2, "after": 2, "gone": 1, "kept": 1, "new": 1}
        assert s["cats"]["missed_fish"]["gone"] == 1 and s["cats"]["fin_cut"]["new"] == 1
        assert "changed" not in s  # no changed_vs_B.json
        try:
            cr.domain_stats(make_compare(Path(tmp) / "x"), "B", "E")
        except SystemExit:
            return
        raise AssertionError("unknown run accepted")


def test_scaling_and_expected_rate():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        review = make_compare(tmp, changed=(60, 1000))
        baseline = make_domain(tmp / "base", "full_body", [("fix", ["bleed"])] * 10 + [("ok", [])] * 90)
        s = cr.domain_stats(review, "B", "D", baseline)
        assert s["est_fixed"][0] == 20  # 2 of 6 fixed x 60 changed
        assert abs(s["baseline"]["rate"] - 0.10) < 1e-12
        # 10 % - 6 % changed x (2 fixed - 1 broken) / 6
        assert abs(s["baseline"]["expected"] - (0.10 - 0.06 * 1 / 6)) < 1e-12


def test_page_is_valid_offline_and_has_no_file_names():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        stats = [cr.domain_stats(make_compare(tmp / d, d, changed=(60, 1000)), "B", "D") for d in ("full_body", "Röntgen")]
        page = cr.render(stats, "B", "D", "2026-10-05", "python scripts/compare_report.py ...")
        for svg in re.findall(r"<svg.*?</svg>", page, re.S):
            ET.fromstring(svg)  # well-formed, e.g. "p < 0.001" escaped
        assert "NMW" not in page and "_WEB" not in page
        assert not re.findall(r"https?://", page)
        assert "Photos" in page and "X-rays" in page and "NHM Wien" in page
        assert "merged" in page  # named as empty in both runs


def test_random_mode_needs_the_exact_sample_and_words_rates_for_the_whole_set():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        review = make_compare(tmp, "Röntgen", changed=(60, 1000))
        names = [f"Fish_NMW{k}_WEB.jpg" for k in range(len(PAIRS))]
        sample = tmp / "sample.json"
        sample.write_text(json.dumps({"seed": 2, "population": 4431, "sample": names}))
        s = cr.domain_stats(review, "B", "D", sample=sample)
        assert (s["population"], s["sample_seed"]) == (4431, 2)
        page = cr.render([s], "B", "D", "2026-10-08", "cmd", mode="random", max_rate=0.5)
        assert "random images of all 4,431" in page and "error rate (whole set)" in page
        assert "random sample of the whole set" in page and "changed</strong> images only" not in page
        assert "What this means for the whole set" not in page  # no scaling of changed images
        assert "Acceptance rule (error rate ≤ 50 %)" in page
        assert page.count("✓ accepted") == 1 and page.count("✗ not accepted") == 1  # D 3/6 = 50 %, B 4/6
        sample.write_text(json.dumps({"seed": 2, "population": 4431, "sample": names[:-1]}))
        try:
            cr.domain_stats(review, "B", "D", sample=sample)
        except SystemExit:
            return
        raise AssertionError("review that is not the sample was accepted")


def test_list_mode_and_groups():
    with tempfile.TemporaryDirectory() as tmp:
        review = make_compare(Path(tmp), "Röntgen", changed=(60, 1000))
        fin = {"Fish_NMW0_WEB.jpg", "Fish_NMW1_WEB.jpg", "Fish_NMW2_WEB.jpg"}
        ok = {f"Fish_NMW{k}_WEB.jpg" for k in (3, 4, 5)}
        a = cr.domain_stats(review, "B", "D", group=("fin_cut", fin))
        b = cr.domain_stats(review, "B", "D", group=("ok", ok))
        assert (a["label"], a["n"], b["label"], b["n"]) == ("X-rays · fin_cut", 3, "X-rays · ok", 3)
        assert a["outcome"] == {"fixed": 2, "still": 1, "both_ok": 0, "broken": 0}
        assert b["outcome"] == {"fixed": 0, "still": 1, "both_ok": 1, "broken": 1}
        none = cr.domain_stats(review, "B", "D", group=("other", {"Fish_NMW99_WEB.jpg"}))
        assert none["n"] == 0 and "changed" not in none  # a group nobody judged yet does not crash
        page = cr.render([a, b], "B", "D", "2026-10-08", "cmd", mode="list", note="Dev list from seed 1.")
        assert "chosen list, not a random sample" in page and "Dev list from seed 1." in page
        assert "listed images" in page and "What this means for the whole set" not in page
        assert "random changed images" not in page


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in list(globals().items()) if name.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"  ok  {name}")
    print(f"All {len(tests)} compare_report tests passed.")
