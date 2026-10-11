"""Tests for scripts/run_part1.py and its two configs: flags, the command list,
the input check and the decision templates (nothing is run).

Runs under pytest or directly: python tests/test_run_part1.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "scripts" / d) for d in ("", "pipeline", "review")]

import run_part1 as rp  # noqa: E402

NEUTRAL = REPO / "pipeline" / "config.yaml"
THESIS = REPO / "pipeline" / "config_thesis_nhm.yaml"
TEMPLATE = REPO / "pipeline" / "decisions_template"


def pairs(argv):
    return list(zip(argv, argv[1:]))


def segment_args(cfg, domain):
    import segment_fish as sf  # torch etc.; only needed here
    return vars(sf.build_parser().parse_args(rp.cli_flags(rp.segment_options(cfg, domain))))


def test_cli_flags():
    assert rp.cli_flags({"bleed_max_box_edge": 0.3, "wrong_region_fix": False, "fin_threshold": None,
                         "bleed_fill": [0.2, 0.9], "x": True}) == [
        "--bleed-max-box-edge", "0.3", "--no-wrong-region-fix", "--bleed-fill", "0.2", "0.9", "--x"]


def test_commands_follow_the_thesis_config():
    cfg = yaml.safe_load(THESIS.read_text())
    with tempfile.TemporaryDirectory() as tmp:
        cfg["processed_dir"] = tmp
        cmds = rp.commands(cfg, rp.STEPS)
        assert [c[0] for c in cmds] == ["labels", "segment full_body", "segment Röntgen", "merge full_body",
                                        "merge Röntgen", "split", "manifest"]
        argv = {step: a for step, a, _ in cmds}
        assert ("--bleed-max-box-edge", "0.1") in pairs(argv["segment full_body"]) and "--resume" in argv["segment full_body"]
        assert ("--bleed-max-box-edge", "0.3") in pairs(argv["segment Röntgen"])
        assert ("--bleed-max-box-edge", "0.1") not in pairs(argv["segment Röntgen"])
        assert "--qa-rule" not in argv["merge full_body"] and argv["merge Röntgen"][-2:] == ["--qa-rule", "B"]
        assert ("--class", "genus") in pairs(argv["split"])
        assert str(Path(tmp) / "labels_thesis.csv") in argv["split"] and str(Path(tmp) / "split" / "thesis_seed0") in argv["manifest"]
        # an existing curated dir is not merged again
        (Path(tmp) / "segmented" / "thesis" / "Röntgen").mkdir(parents=True)
        skips = {step: skip for step, _, skip in rp.commands(cfg, ["merge"])}
        assert skips["merge full_body"] is None and "exists" in skips["merge Röntgen"]


def test_thesis_config_is_setting_d():
    """Photos get exactly segment_fish.py's defaults (setting D, so --resume on D_full_2026-10-05 is
    accepted); X-rays differ only in --bleed-max-box-edge."""
    import segment_fish as sf
    defaults = vars(sf.build_parser().parse_args([]))
    cfg = yaml.safe_load(THESIS.read_text())
    assert segment_args(cfg, "full_body") == defaults
    xray = segment_args(cfg, "Röntgen")
    assert {k for k in defaults if xray[k] != defaults[k]} == {"bleed_max_box_edge"} and xray["bleed_max_box_edge"] == 0.3


def test_neutral_config_has_every_fix_off():
    neutral, thesis = yaml.safe_load(NEUTRAL.read_text()), yaml.safe_load(THESIS.read_text())
    assert set(thesis["segment"]) <= set(neutral["segment"])  # every thesis setting is explained there
    assert all(not d.get("qa_rule") for d in neutral["domains"].values())
    assert neutral["decisions_dir"] != thesis["decisions_dir"]
    for domain in neutral["domains"]:
        a = segment_args(neutral, domain)
        assert a["mask_threshold"] == 0.0 and a["wrong_region_fix"] is False and a["fin_threshold"] is None
        assert a["min_component_frac"] == a["bleed_outside_frac"] == a["keep_near_frac"] == 0
        assert a["fill_holes_frac"] == a["margin_frac"] == 0


def test_missing_inputs_point_at_the_template():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cfg = yaml.safe_load(NEUTRAL.read_text())
        cfg["raw_dir"], cfg["decisions_dir"] = str(tmp / "raw"), str(tmp / "dec")
        cfg["labels"]["path"] = str(tmp / "raw" / "labels.csv")
        missing = rp.missing_inputs(cfg, rp.STEPS)
        # raw_dir, the 2 csv-route sheets, the labels CSV, raw_dir/photos, mask_drops_photos.tsv
        assert len(missing) == 1 + 2 + 1 + 2 and all("template" in m for m in missing if "missing" in m)
        assert [m for m in missing if "your labels CSV" in m]
        assert rp.missing_inputs(cfg, ["split", "manifest"]) == []
        (tmp / "raw" / "photos").mkdir(parents=True)
        (tmp / "raw" / "labels.csv").write_text("file_name,specimen\n")
        shutil.copytree(TEMPLATE, tmp / "dec")
        assert [m for m in rp.missing_inputs(cfg, rp.STEPS) if "mask_drops_photos" in m]
        (tmp / "dec" / "mask_drops_DOMAIN.tsv").rename(tmp / "dec" / "mask_drops_photos.tsv")
        assert rp.missing_inputs(cfg, rp.STEPS) == []
        cfg["labels"]["source"] = "spreadsheet"
        assert [m for m in rp.missing_inputs(cfg, ["labels"]) if "labels.source" in m]


def test_labels_route_follows_the_config():
    neutral, thesis = yaml.safe_load(NEUTRAL.read_text()), yaml.safe_load(THESIS.read_text())
    a = rp.commands(neutral, ["labels"])[0][1]
    assert Path(a[1]).name == "import_labels.py" and ("--csv", neutral["labels"]["path"]) in pairs(a)
    assert ("--specimen", "specimen") in pairs(a) and ("--decisions", "pipeline/decisions_my_dataset/labels") in pairs(a)
    b = rp.commands(thesis, ["labels"])[0][1]
    assert Path(b[1]).name == "extract_labels.py" and "--csv" not in b and ("--decisions", "pipeline/decisions/labels") in pairs(b)


def test_template_files_mean_no_decisions():
    import extract_labels as el
    import merge_runs as mr
    assert mr.read_sha_drop_list((TEMPLATE / "mask_drops_DOMAIN.tsv").read_text()) == {}
    for sheet in rp.LABEL_SHEETS["nhm_filenames"]:  # same header as the thesis sheets
        assert (TEMPLATE / "labels" / sheet).read_text().splitlines() == \
            (REPO / "pipeline" / "decisions" / "labels" / sheet).read_text().splitlines()[:1]
    for sheet in rp.LABEL_SHEETS["csv"]:  # header only
        assert len((TEMPLATE / "labels" / sheet).read_text().splitlines()) == 1
    rows = [{"file_name": f"full_body/Salmo_trutta_NMW1234{i}.jpg", "photo_type": "full_body",
             **el.parse_filename(f"Salmo_trutta_NMW1234{i}")} for i in range(2)]
    done = el.apply_decisions(rows, TEMPLATE / "labels", {r["file_name"]: f"sha{i}" for i, r in enumerate(rows)})
    assert not any(done.values()) and all(r["label_drop"] == "" and r["label_change"] == "" for r in rows)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
