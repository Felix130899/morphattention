"""Unit tests for scripts/split_specimens.py and scripts/build_manifest.py.

All catalog numbers here are made up. Runs under pytest or directly:
python tests/test_split_manifest.py
"""

import collections
import csv
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_manifest as bm  # noqa: E402
import split_specimens as ss  # noqa: E402


def nums(name):
    return ss.catalog_numbers(name)[0]


def test_catalog_number_parsing():
    assert nums("Salmo_trutta_NMW12345_left_WEB.jpg") == {12345}
    assert nums("Salmo_trutta_NMW_12345_left_WEB.jpg") == {12345}
    assert nums("Salmo_trutta_MNW12345_WEB.jpg") == {12345}
    assert nums("A_b_NMW12345-47_syntypes_WEB.jpg") == {12345, 12346, 12347}
    assert nums("A_b_NMW12345-12347_WEB.jpg") == {12345, 12346, 12347}
    assert nums("A_b_NMW12345-NMW12347_WEB.jpg") == {12345, 12346, 12347}
    assert nums("A_b_NMW12340-42,48-50_WEB.jpg") == {12340, 12341, 12342, 12348, 12349, 12350}
    assert nums("A_b_NMW12345,12399_WEB.jpg") == {12345, 12399}
    # underscore tails are indices inside the lot, parentheses are ignored
    assert nums("A_b_NMW12345_1-3_RW99_WEB.jpg") == {12345}
    assert nums("A_b_NMW12345_4_WEB.jpg") == {12345}
    assert nums("A_b_NMW12345(678)_WEB.jpg") == {12345}
    # "-" tails giving no forward range are sub-indices, the rest of the chain too
    assert ss.catalog_numbers("A_b_NMW54321-2222-3_RW1_WEB.jpg") == ({54321}, ["-2222-3"])
    assert ss.catalog_numbers("A_b_NMW4321-10_WEB.jpg") == ({4321}, ["-10"])
    assert nums("A_b_NMW12345-99999_WEB.jpg") == {12345}  # longer than MAX_RANGE
    assert nums("Unknown_fish_WEB.jpg") == set()


def test_genus_of():
    assert ss.genus_of("Salmo trutta") == "Salmo"
    assert ss.genus_of("Loricaria(Rineloricaria) lima") == "Loricaria"
    assert ss.genus_of("Barbus sp.") == "Barbus"


def test_overlapping_numbers_form_one_group():
    groups = ss.build_groups({"series": {10, 11, 12}, "single": {11}, "other": {50}, "later": {5}})
    assert groups["series"] == groups["single"] != groups["other"]
    assert groups["later"] == "S00001" and groups["series"] == "S00002" and groups["other"] == "S00003"
    linked = ss.build_groups({"a": {10}, "copy_of_a": {99}, "b": {50}}, same_bytes=[["a", "copy_of_a"]])
    assert linked["a"] == linked["copy_of_a"] != linked["b"]


def make_groups(spec):
    """spec: {genus: [domains string per group]}, e.g. {"A": ["fr", "f"]} (f = full_body, r = Röntgen)."""
    groups, k = {}, 0
    for genus, ds in spec.items():
        for d in ds:
            k += 1
            groups[f"S{k:05d}"] = {"genus": genus, "domains": {{"f": "full_body", "r": "Röntgen"}[c] for c in d}}
    return groups


def test_assign_splits_targets_and_train_only():
    domains = ["full_body", "Röntgen"]
    groups = make_groups({"Big": ["fr"] * 20, "Five": ["f"] * 5, "Mixed": ["fr"] * 3 + ["f"] * 3, "Rare": ["fr"] * 4})
    split, evaluated = ss.assign_splits(groups, domains, 0, 0.15, 0.15, 5)
    by = collections.defaultdict(collections.Counter)
    for sid, s in split.items():
        by[groups[sid]["genus"]][s] += 1
    assert by["Big"] == {"train": 14, "val": 3, "test": 3}
    assert by["Five"] == {"train": 3, "val": 1, "test": 1}
    assert by["Rare"] == {"train_only": 4}
    assert evaluated == {"full_body": {"Big", "Five", "Mixed"}, "Röntgen": {"Big"}}
    # Mixed: 6 photo specimens (evaluated), 3 X-ray (not) -> 1 val + 1 test for the photos
    assert by["Mixed"]["test"] == 1 and by["Mixed"]["val"] == 1
    # deterministic, and another seed gives another assignment
    assert ss.assign_splits(groups, domains, 0, 0.15, 0.15, 5)[0] == split
    assert ss.assign_splits(groups, domains, 1, 0.15, 0.15, 5)[0] != split


def test_adding_a_genus_leaves_others_unchanged():
    domains = ["full_body", "Röntgen"]
    base = make_groups({"Alpha": ["f"] * 8})
    more = {**base, **{f"S9{k:04d}": {"genus": "Beta", "domains": {"full_body"}} for k in range(8)}}
    a = ss.assign_splits(base, domains, 0, 0.15, 0.15, 5)[0]
    b = ss.assign_splits(more, domains, 0, 0.15, 0.15, 5)[0]
    assert all(b[s] == a[s] for s in a)


def write_run(run, raw, records, masks):
    (run / "masks").mkdir(parents=True)
    (run / "run_config.json").write_text(json.dumps({"settings": {"raw_dir": str(raw)}}))
    (run / "annotations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    for m in masks:
        (run / "masks" / f"{Path(m).stem}.png").write_bytes(b"png")


def test_manifest_rows():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        raw = tmp / "raw" / "full_body"
        raw.mkdir(parents=True)
        names = ["A_b_NMW11111_WEB.jpg", "A_b_NMW22222_WEB.jpg", "A_b_NMW33333_WEB.jpg", "A_b_WEB.jpg"]
        for k, n in enumerate(names):
            (raw / n).write_bytes(bytes([k]))
        write_run(tmp / "seg" / "full_body", raw, [
            {"file_name": names[0], "status": "ok", "instances": [{}, {}]},
            {"file_name": names[1], "status": "dropped", "reason": "after fallback: bleed"},
            {"file_name": names[2], "status": "skipped", "instances": []},
            {"file_name": names[3], "status": "ok", "instances": [{}]}], [names[0], names[3]])
        (tmp / "seg" / "full_body" / "skipped.txt").write_text(f"{names[2]}\tno_detection\n")
        labels = [{"file_name": f"full_body/{n}", "photo_type": "full_body", "species": "A b",
                   "catalog_number": "", "needs_review": "False"} for n in names]
        split_rows = {f"full_body/{names[0]}": {"specimen": "S00001", "split": "test", "genus_evaluated": "True"}}
        cache = {}
        rows = bm.build_rows(labels, tmp / "seg", split_rows, {"full_body": raw}, cache)
        got = [(r["status"], r["reason"], r["split"], r["specimen"], bool(r["mask_path"]), r["n_instances"])
               for r in rows]
        assert got == [("usable", "", "test", "S00001", True, 2),
                       ("dropped", "after fallback: bleed", "none", "", False, ""),
                       ("skipped", "no_detection", "none", "", False, ""),
                       ("dropped", "no catalog number", "none", "", False, "")]
        assert len({r["sha256"] for r in rows}) == 4 and len(cache) == 4
        # the cache is used when size and mtime are unchanged
        cache[str(raw / names[0])][2] = "cached"
        assert bm.file_sha256(raw / names[0], cache) == "cached"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
