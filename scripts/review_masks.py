"""Keyboard-driven review of segment_fish.py overlays: sort good masks from bad.

Serves a local page (127.0.0.1 only - nothing leaves the machine; VS Code
forwards the port to your browser) that shows one overlay at a time:

    ->  / Space   next (an image you haven't judged yet is recorded as "ok")
    <-            previous
    f             "fix"  - mask needs correcting in CVAT/X-AnyLabeling, then next
    x             "drop" - image unusable for training (not fixable), then next
    u / Backspace clear the verdict of the current image
    n             jump to the next unjudged image
    r (hold)      show the raw image without the overlay

Every keypress is saved immediately to ``<run-dir>/review/``:

    decisions.jsonl   one line per keypress, the last line per image wins
                      (append-only, so a crash or Ctrl+C loses nothing)
    fix.txt           file names judged "fix"  (rewritten on every change)
    drop.txt          file names judged "drop"

Restarting resumes at the first unjudged image. Skipped images have no overlay
and are not shown - they are already listed in ``<run-dir>/skipped.txt``.
The image list is read once at startup, so restart to pick up images a still
running segmentation has added since.

    python scripts/review_masks.py --run-dir data/processed/segmented/full_body
    python scripts/review_masks.py --run-dir data/processed/segmented/Röntgen --flagged-only
"""

import argparse
import json
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

VERDICTS = ("ok", "fix", "drop")
WORKSPACE = Path("/workspace")


def load_items(run_dir, flagged_only=False):
    """Reviewable images (status ok, overlay on disk) in file-name order."""
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
            if not overlay.exists() or (flagged_only and not rec["qa"]["flags"]):
                continue
            items.append({"file_name": rec["file_name"], "flags": rec["qa"]["flags"],
                          "n_instances": len(rec["instances"]),
                          "overlay": overlay, "raw": raw_dir / rec["file_name"]})
    items.sort(key=lambda it: it["file_name"])
    return items


def load_decisions(review_dir):
    """file_name -> verdict, replaying decisions.jsonl (last line wins, None = cleared)."""
    path = review_dir / "decisions.jsonl"
    decisions = {}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:  # truncated last line after a crash
                continue
            if d["verdict"] is None:
                decisions.pop(d["file_name"], None)
            else:
                decisions[d["file_name"]] = d["verdict"]
    return decisions


def record_decision(review_dir, decisions, file_name, verdict):
    """Append one decision and rewrite fix.txt / drop.txt to match."""
    if verdict is not None and verdict not in VERDICTS:
        raise ValueError(f"unknown verdict {verdict!r}")
    with open(review_dir / "decisions.jsonl", "a") as f:
        f.write(json.dumps({"file_name": file_name, "verdict": verdict,
                            "time": datetime.now().isoformat(timespec="seconds")}) + "\n")
    if verdict is None:
        decisions.pop(file_name, None)
    else:
        decisions[file_name] = verdict
    for v in ("fix", "drop"):
        names = sorted(n for n, d in decisions.items() if d == v)
        (review_dir / f"{v}.txt").write_text("".join(n + "\n" for n in names))


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
  main { flex: 1; min-height: 0; display: flex; justify-content: center; align-items: center; }
  main img { max-width: 100%; max-height: 100%; object-fit: contain; }
  footer { background: #1c1c1c; color: #999; }
  kbd { background: #333; border-radius: 3px; padding: 0 5px; color: #eee; }
</style></head>
<body>
<header><span id="pos"></span><span id="verdict"></span><span id="name"></span><span id="flags"></span></header>
<main><img id="img" alt=""></main>
<footer><span id="counts"></span>
  <span><kbd>&rarr;</kbd>/<kbd>Space</kbd> next (ok) &nbsp; <kbd>&larr;</kbd> back &nbsp; <kbd>f</kbd> fix &nbsp;
  <kbd>x</kbd> drop &nbsp; <kbd>u</kbd> clear &nbsp; <kbd>n</kbd> next unjudged &nbsp; hold <kbd>r</kbd> raw</span></footer>
<script>
let items = [], decisions = {}, i = 0, showRaw = false;
const $ = id => document.getElementById(id);

async function load() {
  const s = await (await fetch('/api/state')).json();
  items = s.items; decisions = s.decisions;
  i = items.findIndex(it => !(it.file_name in decisions));
  if (i < 0) i = 0;
  show();
}

function show() {
  if (!items.length) { $('name').textContent = 'No images to review.'; return; }
  const it = items[i], v = decisions[it.file_name];
  $('img').src = (showRaw ? '/raw/' : '/overlay/') + i;
  $('pos').textContent = `${i + 1} / ${items.length}`;
  $('name').textContent = it.file_name;
  $('flags').innerHTML = `${it.n_instances} fish ` + it.flags.map(f => `<span>${f}</span>`).join('');
  $('verdict').textContent = v || 'unjudged';
  $('verdict').className = v || 'none';
  const c = {ok: 0, fix: 0, drop: 0};
  for (const d of Object.values(decisions)) c[d]++;
  const judged = c.ok + c.fix + c.drop;
  $('counts').textContent = `ok ${c.ok} · fix ${c.fix} · drop ${c.drop} · unjudged ${items.length - judged}`;
  for (const k of [1, 2]) if (items[i + k]) new Image().src = '/overlay/' + (i + k);  // preload
}

async function decide(verdict) {
  const name = items[i].file_name;
  const r = await fetch('/api/decide', {method: 'POST', headers: {'Content-Type': 'application/json'},
                                         body: JSON.stringify({file_name: name, verdict})});
  if (!r.ok) { alert('Saving failed: ' + await r.text()); return false; }
  if (verdict === null) delete decisions[name]; else decisions[name] = verdict;
  return true;
}

function go(step) { i = Math.min(items.length - 1, Math.max(0, i + step)); show(); }

let busy = false;  // ignore keys while a save is in flight, so a fast "f ->" can't overwrite fix with ok
document.addEventListener('keydown', async e => {
  if (!items.length || e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.key === ' ' || e.key === 'Backspace') e.preventDefault();
  if (busy) return;
  busy = true;
  try { await onKey(e); } finally { busy = false; }
});

async function onKey(e) {
  const k = e.key, name = items[i].file_name;
  if (k === 'ArrowRight' || k === ' ') {
    if (e.repeat) return;
    if (!(name in decisions) && !(await decide('ok'))) return;
    go(1);
  } else if (k === 'ArrowLeft') { go(-1); }
  else if (k === 'f' || k === 'x') {
    if (e.repeat) return;
    if (await decide(k === 'f' ? 'fix' : 'drop')) go(1);
  } else if (k === 'u' || k === 'Backspace') {
    if (await decide(null)) show();
  } else if (k === 'n') {
    const next = items.findIndex((it, j) => j > i && !(it.file_name in decisions));
    if (next >= 0) { i = next; show(); }
  } else if (k === 'r' && !showRaw) { showRaw = true; show(); }
}
document.addEventListener('keyup', e => { if (e.key === 'r') { showRaw = false; show(); } });
load();
</script></body></html>
"""


def make_handler(items, decisions, review_dir, lock):
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
                state = {"items": [{k: it[k] for k in ("file_name", "flags", "n_instances")} for it in items],
                         "decisions": decisions}
                return self._send(json.dumps(state).encode(), "application/json")
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
                with lock:
                    record_decision(review_dir, decisions, body["file_name"], body["verdict"])
            except (ValueError, KeyError, OSError) as e:
                return self._send(str(e).encode(), "text/plain", 400)
            self._send(b"{}", "application/json")

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="A segment_fish.py output folder, e.g. data/processed/segmented/full_body.")
    parser.add_argument("--flagged-only", action="store_true",
                        help="Only show images with at least one QA flag (multi_instance, tiny, ...).")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    items = load_items(args.run_dir, args.flagged_only)
    review_dir = args.run_dir / "review"
    review_dir.mkdir(exist_ok=True)
    decisions = load_decisions(review_dir)
    judged = sum(it["file_name"] in decisions for it in items)
    print(f"{len(items)} images to review, {judged} already judged. Decisions -> {review_dir}/")

    server = ThreadingHTTPServer(("127.0.0.1", args.port),
                                 make_handler(items, decisions, review_dir, threading.Lock()))
    print(f"Open http://localhost:{args.port}  (Ctrl+C to stop - every keypress is already saved)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
