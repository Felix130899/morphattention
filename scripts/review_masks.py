"""Keyboard-driven review of segment_fish.py overlays: sort good masks from bad.

Serves a local page (127.0.0.1 only - nothing leaves the machine; VS Code
forwards the port to your browser) that shows one overlay at a time:

    ->  / Space   next (an image you haven't judged yet is recorded as "ok")
    <-            previous
    1 - 6         toggle an error category (several allowed), stays on the image:
                  1 fin_cut  2 bleed  3 merged  4 partial  5 missed_fish  6 wrong_object
    f             "fix"  - then next; without a category it stays and asks for one
                  (-> goes on without)
    x             "drop" - image unusable for training (not fixable), then next;
                  keeps any toggled categories
    u / Backspace clear the verdict and categories of the current image
    n             jump to the next unjudged image
    r (hold)      show the raw image without the overlay

Verdict rule (vault/thesis-log/decisions/2026-10-01-mask-policy.md): no
category and not dropped -> ok, any category -> fix, x -> drop (with or
without categories). Toggling a category on a dropped image keeps it dropped.

Single-run mode: every keypress is saved immediately to ``<run-dir>/review/``
(or ``--review-dir``):

    decisions.jsonl   one line per keypress {file_name, verdict, categories, time},
                      the last line per image wins (append-only, so a crash or
                      Ctrl+C loses nothing; old lines without categories load as [])
    fix.txt           file names judged "fix"  (rewritten on every change)
    drop.txt          file names judged "drop"
    categories.csv    file_name, verdict, categories (";"-joined), every judged image

Restarting resumes at the first unjudged image. Skipped images have no overlay
and are not shown - they are already listed in ``<run-dir>/skipped.txt``.
The image list is read once at startup, so restart to pick up images a still
running segmentation has added since.

    python scripts/review_masks.py --run-dir data/processed/segmented/full_body
    python scripts/review_masks.py --run-dir data/processed/segmented/Röntgen --flagged-only
    python scripts/review_masks.py --run-dir data/processed/segmented/full_body --image-list dev_set.txt

``--image-list`` (both modes): one file name per line - the COCO/annotations
``file_name`` (e.g. ``sub/a.png``) or the overlay name (``a.jpg``); blank lines
and lines starting with ``#`` are ignored. Only those images are shown.

Blind compare mode: judge the same images from several runs without knowing
which run an overlay comes from. Items are all (image, run) pairs where the
run has an overlay and the image is in ``--image-list`` (without a list: the
images all runs have). They are shuffled with ``--seed`` and the order is
saved to ``<review-dir>/order.json``, so a restart resumes in the same order
(a restart whose item set differs is refused - use a new ``--review-dir``).
The page shows only "item k / N": no file name, QA flags or run, and images are
served by item index. Results go to ``--review-dir`` only, never into a run:

    order.json        seed, run name -> dir, the shuffled [run, file_name] list
    decisions.jsonl   one line per keypress {item, run, file_name, verdict,
                      categories, time}, the last line per item wins
    summary.csv       per-run counts, written by --summary
    disagreements.csv images where the runs' verdicts/categories differ (--summary)

    python scripts/review_masks.py \\
        --compare base=data/processed/segmented/full_body \\
        --compare thr05=data/processed/segmented/dev_thr05 \\
        --compare thr07=data/processed/segmented/dev_thr07 \\
        --compare marg=data/processed/segmented/dev_margin \\
        --image-list dev_set.txt --review-dir data/processed/segmented/review_dev --seed 0
    python scripts/review_masks.py --summary --review-dir data/processed/segmented/review_dev

``--summary`` (no server) prints per run: items, judged, ok, fix, drop,
fix_rate = fix / (ok + fix) - drops are about the image, not the mask, so
they are left out - and the count per category; then every image the runs
judged differently.

Caveat: the overlay pixels themselves are what segment_fish.py drew (QA flag
labels, grey "dup" boxes), so a run whose settings produce typical flags can
still be recognisable from the image.
"""

import argparse
import csv
import json
import random
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

VERDICTS = ("ok", "fix", "drop")
# Number keys 1-6 on the page, in this order. Meanings: mask policy (see docstring).
CATEGORIES = ("fin_cut", "bleed", "merged", "partial", "missed_fish", "wrong_object")
ACTIONS = ("ok", "fix", "drop", "clear", "toggle")
WORKSPACE = Path("/workspace")


def read_image_list(path):
    """File names from an --image-list file, in file order, without duplicates."""
    names = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and line not in names:
            names.append(line)
    return names


def load_items(run_dir, flagged_only=False, wanted=None):
    """Reviewable images (status ok, overlay on disk) in file-name order.

    ``wanted``: optional set of names; an image is kept if its file_name or its
    overlay name is in it.
    """
    settings = json.loads((run_dir / "run_config.json").read_text())["settings"]
    raw_dir = Path(settings["raw_dir"])
    if not raw_dir.is_absolute():  # segment_fish.py is run from /workspace
        raw_dir = WORKSPACE / raw_dir
    items = []
    with open(run_dir / "annotations.jsonl") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:  # half-written last line of a running segmentation
                continue
            if rec["status"] != "ok":
                continue
            overlay = run_dir / "overlays" / f"{Path(rec['file_name']).stem}.jpg"
            if wanted is not None and rec["file_name"] not in wanted and overlay.name not in wanted:
                continue
            if not overlay.exists() or (flagged_only and not rec["qa"]["flags"]):
                continue
            items.append({"file_name": rec["file_name"], "flags": rec["qa"]["flags"],
                          "n_instances": len(rec["instances"]),
                          "overlay": overlay, "raw": raw_dir / rec["file_name"]})
    items.sort(key=lambda it: it["file_name"])
    return items


def derive_verdict(categories, drop=False):
    """Mask policy: drop wins, any category -> fix, none -> ok."""
    if drop:
        return "drop"
    return "fix" if categories else "ok"


def apply_action(current, action, category=None):
    """The decision after one keypress on an image whose decision is ``current``.

    Decisions are {"verdict", "categories"} or None (unjudged / cleared).
    """
    if action not in ACTIONS:
        raise ValueError(f"unknown action {action!r}")
    if action == "clear":
        return None
    verdict = current["verdict"] if current else None
    cats = list(current["categories"]) if current else []
    if action == "toggle":
        if category not in CATEGORIES:
            raise ValueError(f"unknown category {category!r}")
        cats = [c for c in CATEGORIES if (c in cats) != (c == category)]
        verdict = derive_verdict(cats, drop=verdict == "drop")
    elif action == "ok":
        verdict, cats = "ok", []
    else:  # fix / drop keep the toggled categories
        verdict = action
    return {"verdict": verdict, "categories": cats}


def check_decision(verdict, categories):
    """Validate one decision; returns the categories deduplicated in key order."""
    if verdict is not None and verdict not in VERDICTS:
        raise ValueError(f"unknown verdict {verdict!r}")
    unknown = [c for c in categories if c not in CATEGORIES]
    if unknown:
        raise ValueError(f"unknown category {unknown[0]!r}")
    if categories and verdict in ("ok", None):
        raise ValueError(f"categories need verdict fix or drop, got {verdict!r}")
    return [c for c in CATEGORIES if c in categories]


def load_decisions(review_dir, compare=False):
    """Replay decisions.jsonl (last line wins, verdict None = cleared).

    Returns key -> {"verdict", "categories"}; key is file_name, or
    (run, file_name) in compare mode. Lines from before categories existed
    load with categories [].
    """
    path = review_dir / "decisions.jsonl"
    decisions = {}
    if path.exists():
        for n, line in enumerate(path.read_text().splitlines(), start=1):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:  # truncated last line after a crash
                continue
            if compare and "run" not in d:
                raise ValueError(f"{path} line {n} has no run - not a compare-mode review dir")
            key = (d["run"], d["file_name"]) if compare else d["file_name"]
            if d["verdict"] is None:
                decisions.pop(key, None)
            else:
                decisions[key] = {"verdict": d["verdict"], "categories": d.get("categories") or []}
    return decisions


def _append(review_dir, line):
    line["time"] = datetime.now().isoformat(timespec="seconds")
    with open(review_dir / "decisions.jsonl", "a") as f:
        f.write(json.dumps(line) + "\n")


def _update(decisions, key, verdict, categories):
    if verdict is None:
        decisions.pop(key, None)
    else:
        decisions[key] = {"verdict": verdict, "categories": categories}


def record_decision(review_dir, decisions, file_name, verdict, categories=()):
    """Single-run mode: append one decision, rewrite fix.txt / drop.txt / categories.csv."""
    categories = check_decision(verdict, categories)
    _append(review_dir, {"file_name": file_name, "verdict": verdict, "categories": categories})
    _update(decisions, file_name, verdict, categories)
    for v in ("fix", "drop"):
        names = sorted(n for n, d in decisions.items() if d["verdict"] == v)
        (review_dir / f"{v}.txt").write_text("".join(n + "\n" for n in names))
    with open(review_dir / "categories.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file_name", "verdict", "categories"])
        for name in sorted(decisions):
            w.writerow([name, decisions[name]["verdict"], ";".join(decisions[name]["categories"])])


def record_compare_decision(review_dir, decisions, idx, item, verdict, categories=()):
    """Compare mode: append one decision for item ``idx`` (decisions keyed by idx)."""
    categories = check_decision(verdict, categories)
    _append(review_dir, {"item": idx, "run": item["run"], "file_name": item["file_name"],
                         "verdict": verdict, "categories": categories})
    _update(decisions, idx, verdict, categories)


def build_compare_items(runs, wanted=None):
    """All (image, run) pairs to judge, unshuffled. ``runs``: [(name, run_dir)].

    With ``wanted`` (image-list names): every pair whose run has an overlay for a
    listed image. Without: only images every run has, so each run gets the same set.
    """
    per_run = [(name, load_items(run_dir, wanted=wanted)) for name, run_dir in runs]
    common = None
    if wanted is None:
        common = set.intersection(*({it["file_name"] for it in its} for _, its in per_run))
    items = [{**it, "run": name} for name, its in per_run for it in its
             if common is None or it["file_name"] in common]
    run_rank = {name: k for k, (name, _) in enumerate(runs)}
    items.sort(key=lambda it: (it["file_name"], run_rank[it["run"]]))
    return items


def order_items(review_dir, items, seed, runs):
    """Shuffle ``items`` with ``seed`` and save the order to order.json - or, if
    order.json exists, restore that order (the item set must be unchanged)."""
    path = review_dir / "order.json"
    by_key = {(it["run"], it["file_name"]): it for it in items}
    if path.exists():
        saved = json.loads(path.read_text())
        order = [tuple(k) for k in saved["items"]]
        missing, new = set(order) - set(by_key), set(by_key) - set(order)
        if missing or new:
            raise ValueError(
                f"{path} doesn't match the current runs/image list: {len(missing)} saved items are "
                f"gone, {len(new)} are new. Use the same --compare/--image-list, or a new --review-dir.")
        if saved["seed"] != seed:
            print(f"Note: resuming the order saved with seed {saved['seed']} (ignoring --seed {seed}).")
        return [by_key[k] for k in order]
    keys = sorted(by_key)
    random.Random(seed).shuffle(keys)
    path.write_text(json.dumps({"seed": seed, "runs": {name: str(d) for name, d in runs},
                                "items": [list(k) for k in keys]}, indent=1))
    return [by_key[k] for k in keys]


def summarize(review_dir):
    """Per-run rows and per-image disagreements from order.json + decisions.jsonl."""
    saved = json.loads((review_dir / "order.json").read_text())
    order = [tuple(k) for k in saved["items"]]
    decisions = load_decisions(review_dir, compare=True)
    rows = []
    for run in saved["runs"]:
        row = {"run": run, "items": sum(r == run for r, _ in order), "judged": 0,
               **{v: 0 for v in VERDICTS}, "fix_rate": "", **{c: 0 for c in CATEGORIES}}
        for key in order:
            d = decisions.get(key)
            if key[0] != run or d is None:
                continue
            row["judged"] += 1
            row[d["verdict"]] += 1
            for c in d["categories"]:
                row[c] += 1
        if row["ok"] + row["fix"]:
            row["fix_rate"] = f"{row['fix'] / (row['ok'] + row['fix']):.3f}"
        rows.append(row)
    per_image = {}
    for run, name in order:
        if (run, name) in decisions:
            per_image.setdefault(name, {})[run] = decisions[(run, name)]
    disagree = [(name, by_run) for name, by_run in sorted(per_image.items())
                if len(by_run) > 1 and len({(d["verdict"], tuple(d["categories"])) for d in by_run.values()}) > 1]
    return rows, disagree


def _fmt(d):
    return d["verdict"] + (f"[{','.join(d['categories'])}]" if d["categories"] else "")


def print_summary(review_dir):
    rows, disagree = summarize(review_dir)
    cols = ["run", "items", "judged", *VERDICTS, "fix_rate", *CATEGORIES]
    widths = [max(len(c), *(len(str(r[c])) for r in rows)) for c in cols]
    for line in [dict(zip(cols, cols))] + rows:
        print("  ".join(str(line[c]).ljust(w) if c == "run" else str(line[c]).rjust(w)
                        for c, w in zip(cols, widths)))
    with open(review_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with open(review_dir / "disagreements.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file_name", "run", "verdict", "categories"])
        for name, by_run in disagree:
            for run, d in by_run.items():
                w.writerow([name, run, d["verdict"], ";".join(d["categories"])])
    print("fix_rate = fix / (ok + fix); drops are left out")
    print(f"\n{len(disagree)} images where the runs disagree (verdict or categories):")
    for name, by_run in disagree:
        print(f"  {name}  " + "  ".join(f"{run}={_fmt(d)}" for run, d in by_run.items()))
    print(f"\nWrote {review_dir}/summary.csv and disagreements.csv")


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Mask review</title>
<style>
  html, body { margin: 0; height: 100%; background: #111; color: #ddd; font: 14px system-ui, sans-serif; }
  body { display: flex; flex-direction: column; }
  header, footer { padding: 6px 12px; display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }
  header { background: #1c1c1c; }
  #name { font-family: monospace; word-break: break-all; }
  #flags span { background: #444; border-radius: 3px; padding: 1px 6px; margin-right: 4px; }
  #verdict { font-weight: bold; padding: 2px 10px; border-radius: 3px; }
  .ok { background: #2e7d32; } .fix { background: #c62828; } .drop { background: #6a1b9a; } .none { background: #444; }
  #legend span { padding: 1px 6px; border-radius: 3px; color: #888; }
  #legend span.on { background: #e65100; color: #fff; }
  #hint { color: #ffb74d; }
  main { flex: 1; min-height: 0; display: flex; justify-content: center; align-items: center; }
  main img { max-width: 100%; max-height: 100%; object-fit: contain; }
  footer { background: #1c1c1c; color: #999; }
  kbd { background: #333; border-radius: 3px; padding: 0 5px; color: #eee; }
</style></head>
<body>
<header><span id="pos"></span><span id="verdict"></span><span id="legend"></span><span id="hint"></span>
  <span id="name"></span><span id="flags"></span></header>
<main><img id="img" alt=""></main>
<footer><span id="counts"></span>
  <span><kbd>&rarr;</kbd>/<kbd>Space</kbd> next (ok) &nbsp; <kbd>&larr;</kbd> back &nbsp; <kbd>1</kbd>-<kbd>6</kbd> category &nbsp;
  <kbd>f</kbd> fix &nbsp; <kbd>x</kbd> drop &nbsp; <kbd>u</kbd> clear &nbsp; <kbd>n</kbd> next unjudged &nbsp;
  hold <kbd>r</kbd> raw</span></footer>
<script>
let items = [], decisions = {}, cats = [], compare = false, i = 0, showRaw = false, hint = '';
const $ = id => document.getElementById(id);

async function load() {
  const s = await (await fetch('/api/state')).json();
  items = s.items; decisions = s.decisions; cats = s.categories; compare = s.compare;
  $('legend').innerHTML = cats.map((c, k) => `<span id="cat${k}"><kbd>${k + 1}</kbd> ${c}</span>`).join('');
  i = items.findIndex(it => !(it.key in decisions));
  if (i < 0) i = 0;
  show();
}

function show() {
  if (!items.length) { $('name').textContent = 'No images to review.'; return; }
  const it = items[i], d = decisions[it.key], on = d ? d.categories : [];
  $('img').src = (showRaw ? '/raw/' : '/overlay/') + i;
  $('pos').textContent = (compare ? 'item ' : '') + `${i + 1} / ${items.length}`;
  if (!compare) {
    $('name').textContent = it.file_name;
    $('flags').innerHTML = `${it.n_instances} fish ` + it.flags.map(f => `<span>${f}</span>`).join('');
  }
  $('verdict').textContent = d ? d.verdict : 'unjudged';
  $('verdict').className = d ? d.verdict : 'none';
  cats.forEach((c, k) => { $('cat' + k).className = on.includes(c) ? 'on' : ''; });
  $('hint').textContent = hint;
  const c = {ok: 0, fix: 0, drop: 0};
  for (const v of Object.values(decisions)) c[v.verdict]++;
  const judged = c.ok + c.fix + c.drop;
  $('counts').textContent = `ok ${c.ok} · fix ${c.fix} · drop ${c.drop} · unjudged ${items.length - judged}`;
  for (const k of [1, 2]) if (items[i + k]) new Image().src = '/overlay/' + (i + k);  // preload
}

async function act(action, category) {
  const r = await fetch('/api/decide', {method: 'POST', headers: {'Content-Type': 'application/json'},
                                         body: JSON.stringify({item: i, action, category})});
  if (!r.ok) { alert('Saving failed: ' + await r.text()); return false; }
  const d = (await r.json()).decision, key = items[i].key;
  if (d === null) delete decisions[key]; else decisions[key] = d;
  return true;
}

function go(step) { hint = ''; i = Math.min(items.length - 1, Math.max(0, i + step)); show(); }

let busy = false;  // ignore keys while a save is in flight, so a fast "f ->" can't overwrite fix with ok
document.addEventListener('keydown', async e => {
  if (!items.length || e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.key === ' ' || e.key === 'Backspace') e.preventDefault();
  if (busy) return;
  busy = true;
  try { await onKey(e); } finally { busy = false; }
});

async function onKey(e) {
  const k = e.key, key = items[i].key;
  if (k === 'ArrowRight' || k === ' ') {
    if (e.repeat) return;
    if (!(key in decisions) && !(await act('ok'))) return;
    go(1);
  } else if (k === 'ArrowLeft') { go(-1); }
  else if (/^[1-9]$/.test(k) && +k <= cats.length) {
    if (e.repeat) return;
    if (await act('toggle', cats[+k - 1])) { hint = ''; show(); }
  } else if (k === 'f') {
    if (e.repeat || !(await act('fix'))) return;
    if (decisions[key].categories.length) go(1);
    else { hint = 'fix without category: press 1-6 to add one, \\u2192 to go on'; show(); }
  } else if (k === 'x') {
    if (e.repeat) return;
    if (await act('drop')) go(1);
  } else if (k === 'u' || k === 'Backspace') {
    if (await act('clear')) { hint = ''; show(); }
  } else if (k === 'n') {
    const next = items.findIndex((it, j) => j > i && !(it.key in decisions));
    if (next >= 0) { hint = ''; i = next; show(); }
  } else if (k === 'r' && !showRaw) { showRaw = true; show(); }
}
document.addEventListener('keyup', e => { if (e.key === 'r') { showRaw = false; show(); } });
load();
</script></body></html>
"""


def make_handler(items, decisions, review_dir, lock, compare=False):
    """HTTP handler. In compare mode nothing run-specific (run name, dir, path,
    file name, QA flags) ever leaves the server: items are addressed by index."""
    if compare:
        public_items = [{"key": str(k)} for k in range(len(items))]
    else:
        public_items = [{"key": it["file_name"], **{k: it[k] for k in ("file_name", "flags", "n_instances")}}
                        for it in items]

    def key_of(idx):
        return idx if compare else items[idx]["file_name"]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the terminal quiet
            pass

        def _send(self, body, ctype, status=200):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                return self._send(PAGE.encode(), "text/html; charset=utf-8")
            if self.path == "/api/state":
                with lock:
                    public = {(str(k) if compare else k): d for k, d in decisions.items()}
                    state = {"compare": compare, "categories": CATEGORIES, "items": public_items,
                             "decisions": public}
                    body = json.dumps(state).encode()
                return self._send(body, "application/json")
            kind, _, idx = self.path.strip("/").partition("/")
            if kind in ("overlay", "raw") and idx.isdigit() and int(idx) < len(items):
                path = items[int(idx)][kind]
                if path.exists():
                    ctype = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
                    return self._send(path.read_bytes(), ctype)
            self._send(b"not found", "text/plain", 404)

        def do_POST(self):
            if self.path != "/api/decide":
                return self._send(b"not found", "text/plain", 404)
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                idx = body["item"]
                if not isinstance(idx, int) or not 0 <= idx < len(items):
                    raise ValueError("unknown item")
                with lock:
                    new = apply_action(decisions.get(key_of(idx)), body["action"], body.get("category"))
                    verdict, cats = (new["verdict"], new["categories"]) if new else (None, [])
                    if compare:
                        record_compare_decision(review_dir, decisions, idx, items[idx], verdict, cats)
                    else:
                        record_decision(review_dir, decisions, items[idx]["file_name"], verdict, cats)
            except (ValueError, KeyError, TypeError) as e:
                return self._send(f"bad request: {type(e).__name__} {e}".encode(), "text/plain", 400)
            except OSError:  # message would contain the review dir path
                return self._send(b"could not write the decision", "text/plain", 500)
            self._send(json.dumps({"decision": new}).encode(), "application/json")

    return Handler


def compare_arg(text):
    name, sep, run_dir = text.partition("=")
    if not sep or not name or not run_dir:
        raise argparse.ArgumentTypeError(f"expected NAME=RUN_DIR, got {text!r}")
    return name, Path(run_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path,
                        help="Single-run mode: a segment_fish.py output folder, e.g. "
                             "data/processed/segmented/full_body.")
    parser.add_argument("--compare", type=compare_arg, action="append", metavar="NAME=RUN_DIR",
                        help="Blind compare mode: one run per flag, at least two. Needs --review-dir.")
    parser.add_argument("--review-dir", type=Path,
                        help="Where decisions go. Required with --compare/--summary; "
                             "single-run default: <run-dir>/review.")
    parser.add_argument("--seed", type=int, default=0, help="Shuffle seed for --compare (default 0).")
    parser.add_argument("--image-list", type=Path,
                        help="Only show these images: one file name per line, # comments allowed.")
    parser.add_argument("--summary", action="store_true",
                        help="Print the per-run table of a compare --review-dir and write summary.csv; no server.")
    parser.add_argument("--flagged-only", action="store_true",
                        help="Single-run mode: only show images with at least one QA flag (multi_instance, tiny, ...).")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    if args.summary:
        if not args.review_dir or not (args.review_dir / "order.json").exists():
            parser.error("--summary needs --review-dir of a compare-mode review (with order.json)")
        return print_summary(args.review_dir)
    if bool(args.run_dir) == bool(args.compare):
        parser.error("give either --run-dir or --compare (at least twice)")
    wanted = set(read_image_list(args.image_list)) if args.image_list else None

    if args.compare:
        runs = args.compare
        names = [n for n, _ in runs]
        if len(runs) < 2 or len(set(names)) != len(names):
            parser.error("--compare needs at least two runs with distinct names")
        if args.flagged_only:
            parser.error("--flagged-only differs per run; use --image-list in compare mode")
        if not args.review_dir:
            parser.error("--compare needs --review-dir (results never go into the run dirs)")
        for _, run_dir in runs:
            if not (run_dir / "run_config.json").exists():
                parser.error(f"{run_dir} is not a segment_fish.py run dir (no run_config.json)")
            if args.review_dir.resolve().is_relative_to(run_dir.resolve()):
                parser.error("--review-dir must not be inside a run dir")
        review_dir = args.review_dir
        review_dir.mkdir(parents=True, exist_ok=True)
        try:
            items = order_items(review_dir, build_compare_items(runs, wanted), args.seed, runs)
            saved = load_decisions(review_dir, compare=True)
        except ValueError as e:
            raise SystemExit(str(e))
        decisions = {k: saved[(it["run"], it["file_name"])] for k, it in enumerate(items)
                     if (it["run"], it["file_name"]) in saved}
        n_images = len({it["file_name"] for it in items})
        print(f"{len(items)} items ({n_images} images x {len(runs)} runs, shuffled), "
              f"{len(decisions)} already judged. Decisions -> {review_dir}/")
    else:
        items = load_items(args.run_dir, args.flagged_only, wanted)
        review_dir = args.review_dir or args.run_dir / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        decisions = load_decisions(review_dir)
        judged = sum(it["file_name"] in decisions for it in items)
        print(f"{len(items)} images to review, {judged} already judged. Decisions -> {review_dir}/")
    if wanted is not None:
        found = {n for it in items for n in (it["file_name"], it["overlay"].name)}
        missing = sorted(wanted - found)
        if missing:
            print(f"Note: {len(missing)} image-list names are not shown (no overlay, skipped"
                  f"{', not flagged' if args.flagged_only else ''}), e.g. {missing[:3]}")

    server = ThreadingHTTPServer(("127.0.0.1", args.port),
                                 make_handler(items, decisions, review_dir, threading.Lock(), bool(args.compare)))
    print(f"Open http://localhost:{args.port}  (Ctrl+C to stop - every keypress is already saved)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
