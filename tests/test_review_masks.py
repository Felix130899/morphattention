"""Unit tests for the decision bookkeeping and blind compare mode in
scripts/review_masks.py.

One test starts the HTTP handler on a random 127.0.0.1 port to check that
nothing run-specific is served in compare mode. Runs under pytest or
directly: python tests/test_review_masks.py
"""

import csv
import json
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import review_masks as rm  # noqa: E402

# The category set these tests judge with (any config's set works the same way).
CATS = [{"name": n, "text": f"{n} text"} for n in ("fin_cut", "bleed", "merged", "partial", "missed_fish", "wrong_object")]
NAMES = [c["name"] for c in CATS]


def record(*args, **kw):
    return rm.record_decision(*args, allowed=NAMES, **kw)


def act(*args):
    return rm.apply_action(*args, categories=NAMES)


def started(review_dir):
    """A review dir with its review.json, as review_masks.py main() starts it."""
    review_dir.mkdir(parents=True, exist_ok=True)
    rm.start_review(review_dir, CATS, "tester", "test-config.yaml")
    return review_dir


def make_run(run, raw_dir, names, flags=None):
    """A fake segment_fish.py run dir with an overlay for every name."""
    (run / "overlays").mkdir(parents=True)
    (run / "run_config.json").write_text(json.dumps({"settings": {"raw_dir": str(raw_dir)}}))
    flags = flags or {}
    recs = [{"file_name": n, "status": "ok", "qa": {"flags": flags.get(n, [])}, "instances": [{}]} for n in names]
    (run / "annotations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    for n in names:
        (run / "overlays" / f"{Path(n).stem}.jpg").write_bytes(b"\xff\xd8overlay-bytes")
    return run


def serve(handler):
    """Start the handler on a random 127.0.0.1 port in a background thread."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def http_fetch(server, path, body=None):
    """(status, headers, body) of one GET, or POST when ``body`` is given."""
    req = urllib.request.Request(f"http://127.0.0.1:{server.server_address[1]}{path}",
                                 data=body and json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, str(r.headers).encode(), r.read()
    except urllib.error.HTTPError as e:
        return e.code, str(e.headers).encode(), e.read()


def test_last_decision_wins_and_lists_follow():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        decisions = {}
        record(d, decisions, "a.jpg", "fix", ["bleed"])
        record(d, decisions, "b.jpg", "drop")
        record(d, decisions, "a.jpg", "ok")
        record(d, decisions, "b.jpg", None)
        record(d, decisions, "c.jpg", "fix", ["merged", "fin_cut"])
        assert decisions == {"a.jpg": {"verdict": "ok", "categories": []},
                             "c.jpg": {"verdict": "fix", "categories": ["fin_cut", "merged"]}}
        assert rm.load_decisions(d) == decisions  # restart sees the same state
        assert (d / "fix.txt").read_text() == "c.jpg\n"
        assert (d / "drop.txt").read_text() == ""
        with open(d / "categories.csv") as f:
            assert list(csv.reader(f)) == [["file_name", "verdict", "categories"],
                                           ["a.jpg", "ok", ""], ["c.jpg", "fix", "fin_cut;merged"]]
        last = json.loads((d / "decisions.jsonl").read_text().splitlines()[-1])
        assert last["categories"] == ["fin_cut", "merged"] and "time" in last


def test_truncated_last_line_is_ignored():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        record(d, {}, "a.jpg", "fix")
        with open(d / "decisions.jsonl", "a") as f:
            f.write('{"file_name": "b.jpg", "verd')
        assert rm.load_decisions(d) == {"a.jpg": {"verdict": "fix", "categories": []}}


def test_old_lines_without_categories_load():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "decisions.jsonl").write_text(
            '{"file_name": "a.jpg", "verdict": "fix", "time": "2026-09-30T10:00:00"}\n'
            '{"file_name": "b.jpg", "verdict": "drop", "time": "2026-09-30T10:00:01"}\n'
            '{"file_name": "c.jpg", "verdict": "ok", "time": "2026-09-30T10:00:02"}\n'
            '{"file_name": "c.jpg", "verdict": null, "time": "2026-09-30T10:00:03"}\n')
        decisions = rm.load_decisions(d)
        assert decisions == {"a.jpg": {"verdict": "fix", "categories": []},
                             "b.jpg": {"verdict": "drop", "categories": []}}
        record(d, decisions, "a.jpg", "fix", ["partial"])  # old and new lines mix
        assert rm.load_decisions(d)["a.jpg"] == {"verdict": "fix", "categories": ["partial"]}


def test_invalid_decisions_are_refused():
    with tempfile.TemporaryDirectory() as tmp:
        for verdict, cats in [("maybe", []), ("fix", ["blurry"]), ("ok", ["bleed"]), (None, ["bleed"])]:
            try:
                record(Path(tmp), {}, "a.jpg", verdict, cats)
            except ValueError:
                continue
            raise AssertionError(f"expected ValueError for {verdict!r} {cats!r}")
        assert not (Path(tmp) / "decisions.jsonl").exists()


def test_toggle_categories_and_verdict_derivation():
    assert rm.derive_verdict([]) == "ok"
    assert rm.derive_verdict(["bleed"]) == "fix"
    assert rm.derive_verdict([], drop=True) == rm.derive_verdict(["bleed"], drop=True) == "drop"
    d = act(None, "toggle", "merged")  # unjudged + category -> fix
    assert d == {"verdict": "fix", "categories": ["merged"]}
    d = act(d, "toggle", "fin_cut")  # several allowed, kept in key order
    assert d == {"verdict": "fix", "categories": ["fin_cut", "merged"]}
    d = act(act(d, "toggle", "merged"), "toggle", "fin_cut")
    assert d == {"verdict": "ok", "categories": []}  # all toggled off -> ok
    assert act(None, "fix") == {"verdict": "fix", "categories": []}  # f without category
    drop = act(act(None, "toggle", "partial"), "drop")
    assert drop == {"verdict": "drop", "categories": ["partial"]}  # drop keeps categories
    assert act(drop, "toggle", "bleed")["verdict"] == "drop"
    assert act(act(drop, "toggle", "partial"), "toggle", "bleed")["verdict"] == "drop"
    assert act(drop, "fix") == {"verdict": "fix", "categories": ["partial"]}
    assert act(drop, "ok") == {"verdict": "ok", "categories": []}
    assert act(drop, "clear") is None
    for bad in [("toggle", "blurry"), ("maybe", None)]:
        try:
            act(None, *bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")


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


def test_image_list_filtering():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "list.txt").write_text("# dev set\n\nsub/a.png\n  c.jpg  \nb.jpg\nnot_there.jpg\r\nb.jpg\n# x.jpg\n")
        assert rm.read_image_list(tmp / "list.txt") == ["sub/a.png", "c.jpg", "b.jpg", "not_there.jpg"]
        run = make_run(tmp / "run", "/raw", ["sub/a.png", "b.jpg", "c.jpg", "d.jpg", "sub/e.png"])
        wanted = set(rm.read_image_list(tmp / "list.txt"))
        assert [it["file_name"] for it in rm.load_items(run, wanted=wanted)] == ["b.jpg", "c.jpg", "sub/a.png"]
        # overlay names (<stem>.jpg) work too
        assert [it["file_name"] for it in rm.load_items(run, wanted={"e.jpg"})] == ["sub/e.png"]


def test_compare_items_with_and_without_image_list():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        runs = [("A", make_run(tmp / "A", tmp / "raw", ["1.jpg", "2.jpg", "3.jpg"])),
                ("B", make_run(tmp / "B", tmp / "raw", ["2.jpg", "3.jpg", "4.jpg"]))]
        keys = lambda items: [(it["run"], it["file_name"]) for it in items]  # noqa: E731
        # no list: only images every run has
        assert keys(rm.build_compare_items(runs)) == [("A", "2.jpg"), ("B", "2.jpg"), ("A", "3.jpg"), ("B", "3.jpg")]
        # list: every run that has an overlay for a listed image
        assert keys(rm.build_compare_items(runs, wanted={"1.jpg", "4.jpg", "5.jpg"})) == [("A", "1.jpg"), ("B", "4.jpg")]
        it = rm.build_compare_items(runs, wanted={"1.jpg"})[0]
        assert it["overlay"] == tmp / "A" / "overlays" / "1.jpg" and it["raw"] == tmp / "raw" / "1.jpg"


def test_compare_order_is_seeded_persisted_and_resumed():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        names = [f"{k:03d}.jpg" for k in range(30)]
        runs = [(n, make_run(tmp / n, tmp / "raw", names)) for n in ("A", "B", "C")]
        items = rm.build_compare_items(runs)
        keys = lambda items: [(it["run"], it["file_name"]) for it in items]  # noqa: E731
        r1, r2, r3 = tmp / "r1", tmp / "r2", tmp / "r3"
        for r in (r1, r2, r3):
            r.mkdir()
        first = keys(rm.order_items(r1, items, 0, runs))
        assert sorted(first) == sorted(keys(items)) and first != keys(items)  # shuffled, nothing lost
        assert keys(rm.order_items(r2, items, 0, runs)) == first  # same seed -> same order
        assert keys(rm.order_items(r3, items, 1, runs)) != first  # other seed -> other order
        saved = json.loads((r1 / "order.json").read_text())
        assert saved["seed"] == 0 and saved["runs"] == {n: str(d) for n, d in runs}
        # restart: saved order wins, even with another --seed or another input order
        assert keys(rm.order_items(r1, list(reversed(items)), 5, runs)) == first
        # restart with a different item set is refused instead of silently re-indexing
        try:
            rm.order_items(r1, items[:-1], 0, runs)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for a changed item set")


def test_compare_decisions_and_summary():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        runs = [("base", make_run(tmp / "base", tmp / "raw", ["a.jpg", "b.jpg", "c.jpg"])),
                ("new", make_run(tmp / "new", tmp / "raw", ["a.jpg", "b.jpg", "c.jpg"]))]
        review = tmp / "review"
        started(review)
        items = rm.order_items(review, rm.build_compare_items(runs), 0, runs)
        idx = {(it["run"], it["file_name"]): k for k, it in enumerate(items)}
        decisions = {}

        def judge(run, name, verdict, cats=()):
            k = idx[(run, name)]
            rm.record_compare_decision(review, decisions, k, items[k], verdict, list(cats), allowed=NAMES)

        judge("base", "a.jpg", "fix", ["fin_cut", "bleed"])
        judge("new", "a.jpg", "ok")
        judge("base", "b.jpg", "fix", ["fin_cut"])
        judge("new", "b.jpg", "fix", ["fin_cut"])  # agreement
        judge("base", "c.jpg", "drop", ["partial"])
        judge("new", "c.jpg", "ok")
        judge("new", "c.jpg", None)  # cleared again -> unjudged
        line = json.loads((review / "decisions.jsonl").read_text().splitlines()[0])
        assert line["item"] == idx[("base", "a.jpg")] and line["run"] == "base" and line["file_name"] == "a.jpg"
        assert line["categories"] == ["fin_cut", "bleed"] and line["verdict"] == "fix"
        assert not any((d / "review").exists() for _, d in runs)  # nothing written into run dirs

        loaded = rm.load_decisions(review, compare=True)
        assert loaded[("new", "a.jpg")] == {"verdict": "ok", "categories": []}
        assert ("new", "c.jpg") not in loaded and len(loaded) == 5

        rows, disagree = rm.summarize(review)
        base, new = rows
        assert base["run"] == "base" and new["run"] == "new"
        assert (base["items"], base["judged"], base["ok"], base["fix"], base["drop"]) == (3, 3, 0, 2, 1)
        assert (base["fin_cut"], base["bleed"], base["partial"], base["merged"]) == (2, 1, 1, 0)
        assert base["fix_rate"] == "1.000"
        assert (new["items"], new["judged"], new["ok"], new["fix"], new["drop"]) == (3, 2, 1, 1, 0)
        assert new["fin_cut"] == 1 and new["fix_rate"] == "0.500"
        assert [name for name, _ in disagree] == ["a.jpg"]  # b agrees, c only judged in one run
        rm.print_summary(review)
        with open(review / "summary.csv") as f:
            out = list(csv.DictReader(f))
        assert [r["run"] for r in out] == ["base", "new"] and out[0]["fix"] == "2" and out[1]["ok"] == "1"
        assert (review / "disagreements.csv").read_text().count("a.jpg") == 2


def test_compare_load_refuses_single_run_review_dir():
    with tempfile.TemporaryDirectory() as tmp:
        record(Path(tmp), {}, "a.jpg", "fix")
        try:
            rm.load_decisions(Path(tmp), compare=True)
        except ValueError:
            return
        raise AssertionError("expected ValueError")


def test_compare_server_never_reveals_the_run():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        raw = tmp / "rawimgs"
        raw.mkdir()
        (raw / "fish1.jpg").write_bytes(b"\xff\xd8raw-bytes")
        runs = [("SECRETRUNALPHA", make_run(tmp / "dir_alpha_xyz", raw, ["fish1.jpg"], {"fish1.jpg": ["tiny"]})),
                ("SECRETRUNBETA", make_run(tmp / "dir_beta_xyz", raw, ["fish1.jpg"]))]
        review = tmp / "review"
        started(review)
        items = rm.order_items(review, rm.build_compare_items(runs), 0, runs)
        decisions = {}
        server = serve(rm.make_handler(items, decisions, review, threading.Lock(), compare=True, categories=CATS))
        fetch = lambda path, body=None: http_fetch(server, path, body)  # noqa: E731
        try:
            responses = [fetch("/"), fetch("/overlay/0"), fetch("/overlay/1"), fetch("/raw/0"),
                         fetch("/api/decide", {"item": 0, "action": "toggle", "category": "bleed"}),
                         fetch("/api/decide", {"item": 1, "action": "drop"}),
                         fetch("/api/decide", {"item": 1, "action": "toggle", "category": "nope"}),
                         fetch("/api/decide", {"item": 7, "action": "ok"}),
                         fetch("/api/state"), fetch("/overlay/9"), fetch("/order.json"),
                         fetch("/decisions.jsonl")]
        finally:
            server.shutdown()
            server.server_close()
        assert [s for s, _, _ in responses] == [200] * 6 + [400, 400] + [200] + [404] * 3
        forbidden = [b"SECRETRUN", b"dir_alpha", b"dir_beta", b"xyz", str(tmp).encode(), b"fish1", b"tiny",
                     b"overlays", b"rawimgs"]
        for _, headers, body in responses:
            for word in forbidden:
                assert word not in headers + body, (word, headers + body[:300])
        assert b"item " in responses[0][2] and responses[1][2] == b"\xff\xd8overlay-bytes"
        state = json.loads(responses[8][2])
        assert state["compare"] is True and state["items"] == [{"key": "0"}, {"key": "1"}]
        assert state["decisions"] == {"0": {"verdict": "fix", "categories": ["bleed"]},
                                      "1": {"verdict": "drop", "categories": []}}
        # ...but the review dir knows which run each item came from
        assert rm.load_decisions(review, compare=True)[(items[0]["run"], "fish1.jpg")]["verdict"] == "fix"


def test_single_run_server_still_works():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        run = make_run(tmp / "run", tmp / "raw", ["a.jpg", "b.jpg"], {"b.jpg": ["tiny"]})
        review = run / "review"
        started(review)
        record(review, {}, "a.jpg", "fix")  # an old-style line from an earlier session
        items, decisions = rm.load_items(run), rm.load_decisions(review)
        server = serve(rm.make_handler(items, decisions, review, threading.Lock(), categories=CATS))
        try:
            state = json.loads(http_fetch(server, "/api/state")[2])
            assert state["compare"] is False
            assert state["items"][1] == {"key": "b.jpg", "file_name": "b.jpg", "flags": ["tiny"], "n_instances": 1}
            assert state["decisions"] == {"a.jpg": {"verdict": "fix", "categories": []}}
            for action, cat in [("toggle", "bleed"), ("toggle", "merged"), ("drop", None)]:
                assert http_fetch(server, "/api/decide", {"item": 1, "action": action, "category": cat})[0] == 200
            assert http_fetch(server, "/overlay/1")[2] == b"\xff\xd8overlay-bytes"
        finally:
            server.shutdown()
            server.server_close()
        assert (review / "fix.txt").read_text() == "a.jpg\n" and (review / "drop.txt").read_text() == "b.jpg\n"
        assert "b.jpg,drop,bleed;merged" in (review / "categories.csv").read_text()


def expect_value_error(fn, *args):
    try:
        fn(*args)
    except ValueError as e:
        return str(e)
    raise AssertionError("expected ValueError")


def test_sample_is_seeded_saved_grows_and_refuses_changes():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        names = [f"f{k:02d}.jpg" for k in range(20)]
        run = make_run(tmp / "run", tmp / "raw", names)
        review = tmp / "review"
        started(review)
        items = rm.load_items(run)
        first = [it["file_name"] for it in rm.draw_sample(review, items, 5, 0, run)]
        expected = sorted(names)
        rm.random.Random(0).shuffle(expected)
        assert first == expected[:5] and first != sorted(first)
        saved = json.loads((review / "sample.json").read_text())
        assert saved["sample"] == first and saved["population"] == 20 and saved["seed"] == 0
        # resume (same N or None) gives the same list; growing keeps the prefix
        assert [it["file_name"] for it in rm.draw_sample(review, items, None, 0, run)] == first
        grown = [it["file_name"] for it in rm.draw_sample(review, items, 8, 0, run)]
        assert grown[:5] == first and grown == expected[:8]
        assert json.loads((review / "sample.json").read_text())["n"] == 8
        assert "only grow" in expect_value_error(rm.draw_sample, review, items, 5, 0, run)
        assert "seed" in expect_value_error(rm.draw_sample, review, items, 8, 1, run)
        assert "changed" in expect_value_error(rm.draw_sample, review, items[1:], 8, 0, run)
        assert "larger" in expect_value_error(rm.draw_sample, tmp / "fresh", items, 21, 0, run)
        assert "needed" in expect_value_error(rm.draw_sample, tmp / "fresh", items, None, 0, run)


def test_sample_excludes_dropped_and_skipped():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        run = make_run(tmp / "run", tmp / "raw", ["a.jpg", "b.jpg", "c.jpg"])
        recs = [json.loads(line) for line in (run / "annotations.jsonl").read_text().splitlines()]
        recs[1]["status"], recs[2]["status"] = "dropped", "skipped"
        (run / "annotations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
        review = tmp / "review"
        started(review)
        assert [it["file_name"] for it in rm.draw_sample(review, rm.load_items(run), 1, 0, run)] == ["a.jpg"]


def test_sample_exclude_list_is_left_out_saved_and_enforced_on_resume():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        names = [f"f{k:02d}.jpg" for k in range(20)]
        run = make_run(tmp / "run", tmp / "raw", names)
        items = rm.load_items(run)
        (tmp / "old").mkdir()
        old = rm.draw_sample(tmp / "old", items, 5, 0, run)
        exclude = [it["file_name"] for it in old] + ["gone.jpg"]  # gone.jpg: not in this run
        review = tmp / "review"
        started(review)
        new = [it["file_name"] for it in rm.draw_sample(review, items, 15, 1, run, exclude)]
        assert sorted(new) == sorted(set(names) - set(exclude))  # whole rest, nothing excluded
        expected = sorted(set(names) - set(exclude))
        rm.random.Random(1).shuffle(expected)
        assert new == expected
        saved = json.loads((review / "sample.json").read_text())
        assert (saved["population"], saved["n_exclude"], saved["n_excluded"]) == (15, 6, 5)
        # same list in another order / with duplicates = same list; anything else is refused
        assert [it["file_name"] for it in rm.draw_sample(review, items, None, 1, run, exclude[::-1] * 2)] == new
        assert "same --exclude" in expect_value_error(rm.draw_sample, review, items, None, 1, run)
        assert "same --exclude" in expect_value_error(rm.draw_sample, review, items, None, 1, run, exclude[1:])
        assert "same --exclude" in expect_value_error(rm.draw_sample, tmp / "old", items, None, 0, run, exclude)
        assert "larger" in expect_value_error(rm.draw_sample, tmp / "fresh", items, 16, 1, run, exclude)


def test_wilson_interval():
    low, high = rm.wilson(15, 300)  # 5 % of 300
    assert abs(low - 0.0305) < 1e-3 and abs(high - 0.0810) < 1e-3
    low, high = rm.wilson(0, 300)
    assert low == 0.0 and abs(high - 0.0126) < 1e-3
    assert rm.wilson(0, 0) == (0.0, 1.0)


def test_sample_summary_counts_drops_as_errors():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        run = make_run(tmp / "run", tmp / "raw", [f"f{k}.jpg" for k in range(10)])
        review = tmp / "review"
        started(review)
        items = rm.draw_sample(review, rm.load_items(run), 4, 0, run)
        decisions = {}
        record(review, decisions, items[0]["file_name"], "ok")
        record(review, decisions, items[1]["file_name"], "fix", ["bleed", "fin_cut"])
        record(review, decisions, items[2]["file_name"], "drop", ["wrong_object"])
        outside = next(n for n in (f"f{k}.jpg" for k in range(10)) if n not in {it["file_name"] for it in items})
        record(review, decisions, outside, "fix", ["bleed"])  # not in the sample: ignored
        row = rm.summarize_sample(review)
        assert (row["sample"], row["judged"], row["ok"], row["fix"], row["drop"]) == (4, 3, 1, 1, 1)
        assert row["error_rate"] == "0.6667" and row["bleed"] == 1 and row["wrong_object"] == 1
        low, high = rm.wilson(2, 3)
        assert row["ci_low"] == f"{low:.4f}" and row["ci_high"] == f"{high:.4f}"


def test_sample_server_is_blind_but_saves_by_file_name():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        run = make_run(tmp / "run", tmp / "raw", ["zanderA.jpg", "zanderB.jpg"], {"zanderB.jpg": ["giant"]})
        review = tmp / "review"
        started(review)
        record(review, {}, "zanderB.jpg", "fix", ["bleed"])
        items = rm.draw_sample(review, rm.load_items(run), 2, 0, run)
        decisions = rm.load_decisions(review)
        server = serve(rm.make_handler(items, decisions, review, threading.Lock(), blind=True, categories=CATS))
        try:
            responses = [http_fetch(server, "/"), http_fetch(server, "/api/state"),
                         http_fetch(server, "/api/decide", {"item": 0, "action": "drop"}),
                         http_fetch(server, "/overlay/0")]
        finally:
            server.shutdown()
            server.server_close()
        assert [s for s, _, _ in responses] == [200] * 4
        for _, headers, body in responses:
            for word in [b"zander", b"giant", str(tmp).encode()]:
                assert word not in headers + body, (word, headers + body[:300])
        state = json.loads(responses[1][2])
        b_idx = str([it["file_name"] for it in items].index("zanderB.jpg"))
        assert state["blind"] is True and state["compare"] is False
        assert state["items"] == [{"key": "0"}, {"key": "1"}]
        assert state["decisions"] == {b_idx: {"verdict": "fix", "categories": ["bleed"]}}
        assert rm.load_decisions(review)[items[0]["file_name"]]["verdict"] == "drop"


REPO = Path(__file__).resolve().parents[1]


def write_config(path, categories, max_rate=0.1):
    import yaml
    path.write_text(yaml.safe_dump({"review": {"max_rate": max_rate, "categories": categories}}))
    return path


def test_review_config_of_both_repo_configs_and_refusals():
    neutral = rm.load_review_config(REPO / "pipeline" / "config.yaml")
    thesis = rm.load_review_config(REPO / "pipeline" / "config_thesis_nhm.yaml")
    assert [c["name"] for c in neutral["categories"]] == ["bleed", "wrong_object", "merged", "partial",
                                                          "missed_object", "missing_part"]
    assert [c["name"] for c in thesis["categories"]] == NAMES  # the names in every recorded thesis decision
    assert neutral["attribution"] is None and "NHM" in thesis["attribution"]
    assert thesis["labels"] == {"full_body": "Photos", "Röntgen": "X-rays"} and thesis["max_rate"] == 0.1
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ok = {"name": "missing_fin", "text": "a fin missing"}
        assert rm.load_review_config(write_config(tmp / "a.yaml", [ok]))["categories"][0]["fixes"] == []
        for cats, why in [([{"name": f"c{k}", "text": "t"} for k in range(10)], "at most 9"),
                          ([{"name": "Fin Cut", "text": "t"}], "lowercase"),
                          ([ok, ok], "twice"),
                          ([{"name": "x"}], "no text"),
                          ([{**ok, "fixes": [{"setting": "mask_threshold", "turn": "lower"}]}], "setting, turn and cost"),
                          ([], "no review.categories")]:
            msg = expect_value_error(rm.load_review_config, write_config(tmp / "b.yaml", cats))
            assert why in msg, (why, msg)
        import yaml
        (tmp / "c.yaml").write_text(yaml.safe_dump({"review": {"categories": [ok]}}))
        assert "max_rate" in expect_value_error(rm.load_review_config, tmp / "c.yaml")


def test_review_json_is_written_once_and_guards_the_categories():
    with tempfile.TemporaryDirectory() as tmp:
        review = Path(tmp) / "review"
        review.mkdir()
        assert "--reviewer" in expect_value_error(rm.start_review, review, CATS, None, "c.yaml")
        rm.start_review(review, CATS, "Neo", "c.yaml")
        meta = rm.read_review_meta(review)
        assert meta["categories"] == CATS and meta["reviewers"] == ["Neo"] and rm.category_names(review) == NAMES
        rm.start_review(review, CATS, None, "c.yaml")  # resume without --reviewer
        rm.start_review(review, CATS, "Ana", "c.yaml")  # a second reviewer is added
        assert rm.read_review_meta(review)["reviewers"] == ["Neo", "Ana"]
        for other in (CATS[::-1], CATS[:-1], CATS + [{"name": "missing_fin", "text": "t"}]):
            assert "started with the categories" in expect_value_error(rm.start_review, review, other, "Neo", "d.yaml")
        old = Path(tmp) / "old"
        old.mkdir()
        rm.record_decision(old, {}, "a.jpg", "fix", allowed=NAMES)  # decisions, but no review.json
        assert "no review.json" in expect_value_error(rm.start_review, old, CATS, "Neo", "c.yaml")
        assert "review.json missing" in expect_value_error(rm.category_names, old)


def test_own_categories_reach_the_page_and_the_decisions():
    own = [{"name": "missing_tail", "text": "tail cut off"}, {"name": "bleed", "text": "background in the mask"}]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        run = make_run(tmp / "run", tmp / "raw", ["a.jpg"])
        review = run / "review"
        review.mkdir()
        rm.start_review(review, own, "Neo", "own.yaml")
        server = serve(rm.make_handler(rm.load_items(run), {}, review, threading.Lock(), categories=own))
        try:
            state = json.loads(http_fetch(server, "/api/state")[2])
            ok = http_fetch(server, "/api/decide", {"item": 0, "action": "toggle", "category": "missing_tail"})[0]
            unknown = http_fetch(server, "/api/decide", {"item": 0, "action": "toggle", "category": "fin_cut"})[0]
        finally:
            server.shutdown()
            server.server_close()
        assert state["categories"] == ["missing_tail", "bleed"]
        assert state["category_text"] == ["tail cut off", "background in the mask"]
        assert (ok, unknown) == (200, 400)
        assert rm.load_decisions(review) == {"a.jpg": {"verdict": "fix", "categories": ["missing_tail"]}}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
