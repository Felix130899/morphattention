"""Unit tests for the label decisions in scripts/extract_labels.py.

All catalog numbers here are made up. Runs under pytest or directly:
python tests/test_extract_labels.py
"""

import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import extract_labels as el  # noqa: E402


def row(file_name):
    stem = Path(file_name).stem
    return {"file_name": file_name, "photo_type": file_name.split("/")[0], **el.parse_filename(stem)}


def write_tsv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def make_sheets(d, conflicts, needs_review=()):
    write_tsv(d / "genus_spelling.tsv", ["from_genus", "to_genus", "images_from", "images_to", "confidence",
                                         "note", "decision"],
              [("Tetrodon", "Tetraodon", 1, 0, "high", "", "accept"),
               ("Hypsolepis", "Hypsilepis", 1, 0, "low", "", "")])
    write_tsv(d / "catalog_conflicts.tsv", ["names (images)", "catalogs", "images", "proposal", "confidence",
                                            "note", "decision"], conflicts)
    write_tsv(d / "needs_review.tsv", ["sha256", "reason", "note", "decision"],
              needs_review)


def test_conflict_names():
    assert el.conflict_names("Rutilus (from S plotizza) (1) | Scardinius plotizza (2)") == (
        "Rutilus (from S plotizza)", "Scardinius plotizza")


def test_apply_decisions():
    names = [
        "full_body/Tetrodon_fluviatilis_NMW11111_WEB.jpg",       # respelled
        "full_body/Hypsolepis_sp_NMW11112_WEB.jpg",              # spelling rejected
        "full_body/Gobius_batrachocephalus_NMW22222_WEB.jpg",    # renamed (conflict catalog)
        "Röntgen/Mesogobius_batrachocephalus_NMW22222_WEB.jpg",
        "full_body/Gobius_batrachocephalus_NMW22299_WEB.jpg",    # renamed elsewhere too
        "full_body/Coregonus_danneri_NMW33333_WEB.jpg",
        "full_body/Salmo_NMW33333_WEB.jpg",                      # dropped in this catalog
        "full_body/Salmo_NMW33344_WEB.jpg",                      # kept: another catalog
        "Röntgen/Acanthobrama_marmid_MNW44444_RW1_WEB.jpg",      # catalog typo
        "full_body/Squalius_albus_MW55555_WEB.jpg",              # catalog typo, number from the name
        "Röntgen/Cottus_gobio_NRM(Stockholm)66666_RW2_WEB.jpg",  # own group
        "full_body/Acipenser_sp._NoNumber_WEB.jpg",              # dropped
        "Röntgen/Barbus_dobsoni_NMW77777_RW3_WEB.jpg",           # duplicate, both names final -> dropped
        "Röntgen/Barbus_Barbodes_dobsoni_NMW77777_RW3_WEB.jpg",  # ... alphabetically first -> kept
        "Röntgen/Gobius_batrachocephalus_NMW88888_RW4_WEB.jpg",  # duplicate, renamed -> dropped
        "Röntgen/Mesogobius_batrachocephalus_NMW88888_RW4_WEB.jpg",  # ... name is final -> kept
        "full_body/Alburnus_mento_NMW99999_WEB.jpg",             # renamed in this catalog only
        "full_body/Aspius_aspius_NMW99999_WEB.jpg",
        "full_body/Alburnus_mento_NMW99988_WEB.jpg",             # kept
        "Röntgen/Tetragonopterus_rutilus_56565_1-5_RW5_WEB.jpg", # number without prefix
    ]
    rows = [row(n) for n in names]
    sha = {n: f"h{k}" for k, n in enumerate(names)}
    sha[names[13]] = sha[names[12]]
    sha[names[15]] = sha[names[14]]
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        make_sheets(d, [
            ("Gobius batrachocephalus (1) | Mesogobius batrachocephalus (1)", 1, 2, "", "", "",
             "rename Gobius batrachocephalus -> Mesogobius batrachocephalus"),
            ("Coregonus danneri (1) | Salmo (1)", 1, 2, "", "", "", "drop Salmo"),
            ("Alburnus mento (1) | Aspius aspius (1)", 1, 2, "", "", "",
             "rename in catalog Alburnus mento -> Aspius aspius"),
        ], [
            (sha[names[8]], "catalog_prefix_typo", "", "set catalog NMW<number>"),
            (sha[names[9]], "no_catalog_anchor", "", "set catalog NMW<number>"),
            (sha[names[10]], "no_catalog_anchor", "", "keep, own specimen group NRM<number>"),
            (sha[names[11]], "no_catalog_anchor", "no catalog number", "drop"),
            (sha[names[19]], "no_catalog_anchor", "", "set catalog NMW<number>"),
        ])
        # the NMW88888 pair is a second Gobius/Mesogobius catalog: the sheet must count it
        try:
            el.apply_decisions([dict(r) for r in rows], d, sha)
            raise AssertionError("count mismatch not caught")
        except SystemExit as e:
            assert "labels give 2 / 4" in str(e)
        make_sheets(d, [
            ("Gobius batrachocephalus (3) | Mesogobius batrachocephalus (2)", 2, 4, "", "", "",
             "rename Gobius batrachocephalus -> Mesogobius batrachocephalus"),
            ("Coregonus danneri (1) | Salmo (1)", 1, 2, "", "", "", "drop Salmo"),
            ("Alburnus mento (1) | Aspius aspius (1)", 1, 2, "", "", "",
             "rename in catalog Alburnus mento -> Aspius aspius"),
        ], [
            (sha[names[8]], "catalog_prefix_typo", "", "set catalog NMW<number>"),
            (sha[names[9]], "no_catalog_anchor", "", "set catalog NMW<number>"),
            (sha[names[10]], "no_catalog_anchor", "", "keep, own specimen group NRM<number>"),
            (sha[names[11]], "no_catalog_anchor", "no catalog number", "drop"),
            (sha[names[19]], "no_catalog_anchor", "", "set catalog NMW<number>"),
        ])
        done = el.apply_decisions(rows, d, sha)
    by = {r["file_name"]: r for r in rows}
    assert by[names[0]]["species"] == "Tetraodon fluviatilis"
    assert by[names[1]]["species"] == "Hypsolepis sp"
    assert by[names[2]]["species"] == by[names[4]]["species"] == "Mesogobius batrachocephalus"
    assert by[names[6]]["label_drop"] and not by[names[7]]["label_drop"] and not by[names[5]]["label_drop"]
    assert by[names[8]]["catalog_number"] == "NMW44444" and by[names[9]]["catalog_number"] == "NMW55555"
    assert by[names[10]]["catalog_number"] == "NRM66666" and by[names[19]]["catalog_number"] == "NMW56565"
    assert by[names[11]]["label_drop"] == "needs_review: no catalog number"
    assert not any(by[n]["needs_review"] for n in names[8:12])
    # duplicates: both carry their final name -> alphabetically first; renamed copy loses to the named one
    assert not by[names[13]]["label_drop"] and by[names[12]]["label_drop"]
    assert by[names[14]]["label_drop"] and not by[names[15]]["label_drop"]
    assert by[names[16]]["species"] == "Aspius aspius" and by[names[18]]["species"] == "Alburnus mento"
    assert by[names[16]]["label_change"] == "rename in catalog Alburnus mento -> Aspius aspius"
    assert done["duplicate_sets"] == 2 and done["dropped_duplicate"] == 2 and done["renamed"] == 4
    assert by[names[0]]["label_change"] == "genus Tetrodon -> Tetraodon"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
