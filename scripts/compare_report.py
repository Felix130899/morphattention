"""Blind compare report (B vs. D style): one self-contained HTML chart page.

Reads each compare-mode review dir (``review_masks.py --compare``: order.json +
decisions.jsonl; one dir per domain, two runs judged blind on the same images)
and writes a page with, per domain: the wrong rate (fix + drop) / judged of
each run with a 95 % Wilson interval, what happened to each image (fixed,
still wrong, ok in both, broken; McNemar exact test on the fixed/broken pairs),
and per error category how often it was gone, kept or new in the after run.

If the after run dir holds ``changed_vs_<before>.json`` (``mask_changes.py``),
the page scales the result to every changed image of the domain; with
``--baseline`` (the before run's random-sample review dir, one per domain, same
order) it also gives the expected error rate of the after run. That is an
estimate only - a fresh random sample of the after run gives the real number.
Nothing is fetched from outside; the page works offline and contains no file names.

``--mode`` says what the judged images are, and the page words its rates by it:
  * ``changed`` (default): a random sample of the images whose mask changed
    (as above; rates are much higher than over the whole set);
  * ``random``: a random sample of the whole domain, drawn by
    ``review_masks.draw_sample`` (``--sample`` = its sample.json, one per
    review dir; the judged images must be exactly that sample). The rates
    are each run's error rate under one review standard; ``--max-rate``
    adds the acceptance verdict. No scaling section;
  * ``list``: a hand-picked list (e.g. a dev list); rates describe these
    images only. ``--note`` says how the list was made.
``--groups`` (CSV file_name,group) splits every domain into one row per group,
e.g. fin_cut vs. ok images of a dev list.

The categories and the reviewers come from each review dir's review.json; the
limit for random mode (review.max_rate), extra caveats, the domain labels and
the footer line from ``--config`` (default pipeline/config.yaml). The page
always has the same sections in the same order (SECTIONS); a section that does
not apply to the mode says so in one line.

    python scripts/compare_report.py --mode random --before B --after E \\
        --review-dir data/processed/segmented/review_compare_BE_Röntgen_seed2 \\
        --sample data/processed/segmented/E_merged_<DATE>/Röntgen/sample_seed2/sample.json \\
        --config pipeline/config_thesis_nhm.yaml --out vault/thesis-log/experiments/<DATE>-compare-B-vs-E.html

    python scripts/compare_report.py --before B --after D \\
        --review-dir data/processed/segmented/review_compare_BD_full_body \\
        --review-dir data/processed/segmented/review_compare_BD_Röntgen \\
        --baseline data/processed/segmented/B_merged_2026-10-03/full_body/review_random_seed0 \\
        --baseline data/processed/segmented/B_merged_2026-10-03/Röntgen/review_random_seed0 \\
        --config pipeline/config_thesis_nhm.yaml --out vault/thesis-log/experiments/2026-10-05-compare-B-vs-D.html
"""

import argparse
import csv
import json
import math
from datetime import date
from pathlib import Path

from error_report import (PAGE_CSS, TIP_JS, accepted, axis, category_info, category_list, esc, fmt_p, interval_row,
                          nice_max, page_footer, pct, reviewer_caveat, section)
from review_masks import DEFAULT_CONFIG, load_decisions, load_review_config, read_review_meta, summarize_sample, wilson

SECTIONS = ("Result", "Error rate before and after", "What happened to each image",
            "Error categories: gone, kept, new", "Scaling to the whole set", "Acceptance rule", "Caveats",
            "Rebuild this page")

# Outcome of one image, before -> after. Status colours (dataviz reference
# palette, fixed in both modes); each ships with an icon and a label.
OUTCOMES = [
    ("fixed", "✓", "fixed", "wrong before, ok after", "var(--st-good)"),
    ("still", "!", "still wrong", "wrong in both runs", "var(--st-warning)"),
    ("both_ok", "=", "ok in both", "ok in both runs", "var(--neutral)"),
    ("broken", "✗", "broken", "ok before, wrong after", "var(--st-critical)"),
]

EXTRA_CSS = """
:root { --st-good: #0ca30c; --st-warning: #fab219; --st-critical: #d03b3b; --neutral: #c3c2b7;
  --before: #86b6ef; --after: #256abf; --seg-ink: #0b0b0b; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) { --neutral: #898781; --before: #184f95; --after: #5598e7; }
}
:root[data-theme="dark"] { --neutral: #898781; --before: #184f95; --after: #5598e7; }
.facets { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 12px; }
.facets h3 { font-size: 15px; margin: 0 0 6px; }
.arrow { color: var(--text-muted); font-weight: 400; }
"""


def mcnemar_exact_p(b, c):
    """Two-sided exact McNemar p-value for b and c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


MODES = ("changed", "random", "list")


def domain_stats(review_dir, before, after, baseline=None, sample=None, group=None, labels=None):
    """Stats of one compare review dir. ``sample``: the sample.json the judged
    images were drawn as (random mode; must match exactly). ``group``:
    (name, set of file names) - only those images, labelled with the name.
    ``labels``: {domain: name on the page} (config)."""
    review_dir = Path(review_dir)
    meta = read_review_meta(review_dir)
    saved = json.loads((review_dir / "order.json").read_text())
    runs = saved["runs"]
    for r in (before, after):
        if r not in runs:
            raise SystemExit(f"run {r!r} not in {review_dir} (runs: {sorted(runs)})")
    decisions = load_decisions(review_dir, compare=True)
    names = sorted({name for _, name in saved["items"]})
    smp = None
    if sample is not None:
        smp = json.loads(Path(sample).read_text())
        if set(names) != set(smp["sample"]):
            raise SystemExit(f"{review_dir}: the reviewed images are not the sample in {sample} "
                             f"({len(set(names) - set(smp['sample']))} extra, "
                             f"{len(set(smp['sample']) - set(names))} missing)")
    if group is not None:
        names = [n for n in names if n in group[1]]
    paired = [n for n in names if (before, n) in decisions and (after, n) in decisions]
    wrong = {r: [n for n in paired if decisions[(r, n)]["verdict"] in ("fix", "drop")] for r in (before, after)}
    wb, wa = set(wrong[before]), set(wrong[after])
    n = len(paired)
    outcome = {"fixed": len(wb - wa), "still": len(wb & wa), "both_ok": n - len(wb | wa), "broken": len(wa - wb)}
    rates = {}
    for r, w in ((before, wb), (after, wa)):
        low, high = wilson(len(w), n)
        rates[r] = {"k": len(w), "rate": len(w) / n if n else 0.0, "low": low, "high": high}
    cats = {}
    for c in (c["name"] for c in meta["categories"]):
        has = {r: {m for m in paired if c in decisions[(r, m)]["categories"]} for r in (before, after)}
        cats[c] = {"before": len(has[before]), "after": len(has[after]), "gone": len(has[before] - has[after]),
                   "kept": len(has[before] & has[after]), "new": len(has[after] - has[before])}
    run_dir = Path(runs[after])
    key = run_dir.name
    label = (labels or {}).get(key, key) + (f" · {group[0]}" if group is not None else "")
    s = {"key": key, "label": label, "review_dir": str(review_dir), "runs": runs,
         "seed": saved["seed"], "images": len(names), "n": n, "outcome": outcome, "rates": rates, "cats": cats,
         "cat_text": {c["name"]: c["text"] for c in meta["categories"]}, "reviewers": meta["reviewers"],
         "p": mcnemar_exact_p(outcome["fixed"], outcome["broken"])}
    if smp is not None:
        s["population"], s["sample_seed"] = smp["population"], smp["seed"]
    changed_json = run_dir / f"changed_vs_{before}.json"
    if changed_json.exists() and n:  # n = 0: nothing judged in both runs yet
        ch = json.loads(changed_json.read_text())
        s["changed"], s["compared"] = ch["changed"], ch["compared"]
        f_low, f_high = wilson(outcome["fixed"], n)
        _, b_high = wilson(outcome["broken"], n)
        s["est_fixed"] = [round(v * ch["changed"]) for v in (outcome["fixed"] / n, f_low, f_high)]
        s["est_broken_max"] = round(b_high * ch["changed"])
        if baseline:
            row = summarize_sample(Path(baseline))
            base_rate = (row["fix"] + row["drop"]) / row["judged"]
            net = (outcome["fixed"] - outcome["broken"]) / n
            s["baseline"] = {"dir": str(baseline), "rate": base_rate, "judged": row["judged"],
                             "expected": base_rate - ch["changed"] / ch["compared"] * net}
    return s


def seg_path(x, y, w, h, r_left, r_right):
    """Rect with independently rounded left/right ends (radius clipped to the size)."""
    rl, rr = (min(r, w / 2, h / 2) for r in (r_left, r_right))
    return (f"M{x + rl:.1f},{y}H{x + w - rr:.1f}Q{x + w:.1f},{y} {x + w:.1f},{y + rr:.1f}"
            f"V{y + h - rr:.1f}Q{x + w:.1f},{y + h} {x + w - rr:.1f},{y + h}H{x + rl:.1f}"
            f"Q{x:.1f},{y + h} {x:.1f},{y + h - rl:.1f}V{y + rl:.1f}Q{x:.1f},{y} {x + rl:.1f},{y}Z")


def outcome_chart(stats):
    """One 100 % stacked bar per domain: fixed / still wrong / ok in both / broken."""
    W, L, R, bar_h, row_h, top = 640, 70, 10, 30, 52, 8
    H = top + row_h * len(stats)
    svg = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="What happened to each image, per domain">']
    for i, s in enumerate(stats):
        y = top + row_h * i
        svg.append(f'<text x="{L - 12}" y="{y + bar_h / 2 + 5}" text-anchor="end" font-size="13" '
                   f'fill="var(--text-primary)">{esc(s["label"])}</text>')
        x, full = L, W - L - R
        shown = [(o, s["outcome"][o[0]]) for o in OUTCOMES if s["outcome"][o[0]]]
        for j, ((key, icon, name, meaning, color), k) in enumerate(shown):
            w = full * k / s["n"]
            gap = 2 if j < len(shown) - 1 else 0
            tip = f'{s["label"]}: {k} of {s["n"]} images {name} ({meaning})'
            svg.append(f'<g data-tip="{esc(tip)}"><path d="{seg_path(x, y, w - gap, bar_h, 4 if j == 0 else 0, 4 if not gap else 0)}" '
                       f'fill="{color}"/>')
            if w > 44:
                svg.append(f'<text x="{x + (w - gap) / 2:.1f}" y="{y + bar_h / 2 + 5}" text-anchor="middle" '
                           f'font-size="13" font-weight="600" fill="var(--seg-ink)">{icon} {k}</text>')
            svg.append("</g>")
            x += w
        svg.append(f'<text x="{L}" y="{y + bar_h + 15}" font-size="12" fill="var(--text-muted)">'
                   f'{s["n"]} images · {s["outcome"]["fixed"]} fixed, {s["outcome"]["broken"]} broken · '
                   f'McNemar exact {esc(fmt_p(s["p"]))}</text>')
    svg.append("</svg>")
    return "".join(svg)


def category_chart(s, cats, axis_top):
    """Grouped horizontal bars: images with each category, before vs. after."""
    W, L, R, bar_h, gap, group_gap, top = 420, 100, 34, 11, 2, 14, 6
    group_h = 2 * bar_h + gap + group_gap
    H = top + group_h * len(cats) + 16
    x = lambda v: L + v / axis_top * (W - L - R)  # noqa: E731
    before, after = s["before"], s["after"]
    svg = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(s["label"])}: images per error category, '
           f'{esc(before)} vs. {esc(after)}">']
    step = 20 if axis_top > 40 else 10
    for t in range(0, axis_top + 1, step):
        svg.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top - 2}" y2="{H - 16}" stroke="var(--grid)"/>'
                   f'<text x="{x(t):.1f}" y="{H - 3}" text-anchor="middle" font-size="11" '
                   f'fill="var(--text-muted)">{t}</text>')
    for g, c in enumerate(cats):
        y0 = top + group_h * g
        svg.append(f'<text x="{L - 10}" y="{y0 + bar_h + 4}" text-anchor="end" font-size="12" '
                   f'fill="var(--text-primary)">{esc(c)}</text>')
        d = s["cats"][c]
        for j, (run, k, color) in enumerate(((before, d["before"], "var(--before)"),
                                             (after, d["after"], "var(--after)"))):
            y = y0 + j * (bar_h + gap)
            w = x(k) - L
            tip = f'{s["label"]}, {c}, run {run}: {k} of {s["n"]} images'
            if run == after:
                tip += f' ({d["gone"]} gone, {d["kept"]} kept, {d["new"]} new vs. {before})'
            svg.append(f'<g data-tip="{esc(tip)}"><rect x="{L}" y="{y - 1}" width="{W - L - R}" height="{bar_h + 2}" '
                       f'fill="transparent"/>')
            if k:
                svg.append(f'<path d="{seg_path(L, y, w, bar_h, 0, 4)}" fill="{color}"/>')
            svg.append(f'<text x="{L + w + 6:.1f}" y="{y + bar_h - 2}" font-size="11" '
                       f'fill="var(--text-secondary)">{k}</text></g>')
        svg.append(f'<line x1="{L}" x2="{L}" y1="{y0 - 2}" y2="{y0 + 2 * bar_h + gap + 2}" stroke="var(--text-muted)"/>')
    svg.append("</svg>")
    return "".join(svg)


def describe(s, mode):
    """(what the n images are, tile label) for one stats row."""
    if mode == "random":
        return f'random images of all {s["population"]:,}', "error rate (whole set)"
    if mode == "list":
        return "listed images", "wrong among listed"
    return "random changed images", "wrong among changed"


def rate_chart(stats, before, after, max_rate=None):
    """Wrong rate of both runs per domain: dot + 95 % Wilson interval; dashed limit in random mode."""
    W, L, R, row_h, top = 640, 150, 150, 26, 22
    ends = [s["rates"][r]["high"] for s in stats for r in (before, after)] + ([max_rate] if max_rate is not None else [])
    axis_top, step = nice_max(max(ends + [0.01]) * 1.05)
    H = top + row_h * 2 * len(stats) + 8 * len(stats) + 22
    x = lambda v: L + v / axis_top * (W - L - R)  # noqa: E731
    svg = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Wrong rate per run and domain with 95 % Wilson interval">']
    axis(svg, x, top - 6, H - 22, axis_top, step)
    if max_rate is not None:
        svg.append(f'<line x1="{x(max_rate):.1f}" x2="{x(max_rate):.1f}" y1="{top - 10}" y2="{H - 22}" '
                   'stroke="var(--text-secondary)" stroke-width="1.5" stroke-dasharray="4 3"/>')
        svg.append(f'<text x="{x(max_rate):.1f}" y="{top - 14}" text-anchor="middle" font-size="11" '
                   f'fill="var(--text-secondary)">limit {pct(max_rate, 0)}</text>')
    for i, s in enumerate(stats):
        for j, (run, color) in enumerate(((before, "var(--before)"), (after, "var(--after)"))):
            y = top + (2 * row_h + 8) * i + row_h * j + row_h // 2
            d = s["rates"][run]
            svg.append(f'<text x="{L - 12}" y="{y + 4}" text-anchor="end" font-size="13" '
                       f'fill="var(--text-primary)">{esc(s["label"])} · {esc(run)}</text>')
            tip = (f'{s["label"]}, run {run}: {d["k"]} of {s["n"]} wrong = {pct(d["rate"])}, '
                   f'95 % CI {pct(d["low"])} – {pct(d["high"])}')
            interval_row(svg, x, y, d, color, tip, f'{pct(d["rate"])}  ({pct(d["low"])} – {pct(d["high"])})')
    svg.append("</svg>")
    return "".join(svg)


def scaling_section(stats, before, after, mode):
    if mode != "changed":
        why = "random sample of the whole set" if mode == "random" else "chosen list, not a random sample"
        return f"<p>Not applicable: mode {mode} ({why}).</p>"
    if not all("changed" in s for s in stats):
        return (f"<p>Not applicable: no <code>changed_vs_{esc(before)}.json</code> (scripts/mask_changes.py) "
                "in the after run of every domain.</p>")
    rows = []
    for s in stats:
        ef = s["est_fixed"]
        base = s.get("baseline")
        exp = (f'{pct(base["rate"])} → <strong>≈ {pct(base["expected"])}</strong>' if base else "–")
        rows.append(f'<tr><td>{esc(s["label"])}</td><td>{s["changed"]:,} of {s["compared"]:,} '
                    f'({pct(s["changed"] / s["compared"])})</td>'
                    f'<td>≈ {ef[0]:,} ({ef[1]:,} – {ef[2]:,})</td><td>≤ {s["est_broken_max"]:,}</td>'
                    f'<td>{exp}</td></tr>')
    return f"""<p>Unchanged masks are identical in both runs; the difference is in the changed images only. Sample
     shares scaled to all changed images (95 % Wilson).</p>
  <div class="card table-wrap"><table>
    <tr><th>Domain</th><th>Changed images (IoU &lt; 0.99)</th><th>Masks fixed by {esc(after)}</th>
        <th>Masks broken by {esc(after)} (95 % upper bound)</th><th>Expected error rate {esc(before)} → {esc(after)}</th></tr>
    {"".join(rows)}
  </table></div>
  <p>Expected error rate = {esc(before)}'s rate in its random review − share of changed images × (fixed − broken) /
     judged (only with <code>--baseline</code>). Estimate only; a fresh random sample of {esc(after)} measures the rate.</p>"""


def acceptance_section(stats, before, after, mode, max_rate):
    if mode != "random":
        why = "the images are changed images only" if mode == "changed" else "the images are a chosen list"
        return f"<p>Not applicable: mode {mode} ({why}); the rule needs a random sample of the whole set.</p>"
    rows = []
    for s in stats:
        for r in (before, after):
            x = s["rates"][r]
            ok = accepted(x["k"], s["n"], max_rate)
            rows.append(f'<tr><td>{esc(s["label"])}</td><td>{esc(r)}</td><td>{x["k"]} / {s["n"]}</td>'
                        f'<td><strong>{pct(x["rate"])}</strong></td><td>{pct(x["low"])} – {pct(x["high"])}</td>'
                        f'<td class="{"good" if ok else "bad"}">{"✓ accepted" if ok else "✗ not accepted"}</td></tr>')
    return f"""<p>Accepted if the error rate is <strong>≤ {pct(max_rate, 0)}</strong> (point estimate). Both runs judged on
     the same random images in one blind session. The 95 % Wilson interval is shown, not used for the decision.</p>
  <div class="card table-wrap"><table>
    <tr><th>Domain</th><th>Run</th><th>Wrong / judged</th><th>Error rate</th><th>95 % CI</th><th>Verdict</th></tr>
    {"".join(rows)}
  </table></div>"""


def render(stats, cfg, before, after, created, cmd, mode="changed", max_rate=None, note=None):
    """The page; ``cfg``: review_masks.load_review_config(); sections in SECTIONS order.
    ``max_rate``: the limit (random mode only)."""
    head = []
    for s in stats:
        o, rb, ra = s["outcome"], s["rates"][before], s["rates"][after]
        head.append(f'<p><strong>{esc(s["label"])}:</strong> {s["n"]} {describe(s, mode)[0]}. Wrong {esc(before)} '
                    f'{pct(rb["rate"], 0)} → {esc(after)} {pct(ra["rate"], 0)}. Fixed <strong>{o["fixed"]}</strong>, '
                    f'broken <strong>{o["broken"]}</strong> (McNemar exact {esc(fmt_p(s["p"]))}): '
                    f'{"significant" if s["p"] < 0.05 else "not significant"} at 5 %.</p>')
    tiles = []
    for s in stats:
        rb, ra = s["rates"][before], s["rates"][after]
        tiles.append(f'<div class="card tile"><div class="label">{esc(s["label"])}: {describe(s, mode)[1]}</div>'
                     f'<div class="value">{pct(rb["rate"], 0)} <span class="arrow">→</span> {pct(ra["rate"], 0)}</div>'
                     f'<div class="note muted">{esc(after)}: CI {pct(ra["low"], 0)} – {pct(ra["high"], 0)} · '
                     f'{esc(before)}: {pct(rb["low"], 0)} – {pct(rb["high"], 0)}</div></div>')
    for s in stats:
        o = s["outcome"]
        _, high = wilson(o["broken"], s["n"])
        tiles.append(f'<div class="card tile"><div class="label">{esc(s["label"])}: fixed / broken by {esc(after)}</div>'
                     f'<div class="value">{o["fixed"]} <span class="arrow">/</span> {o["broken"]}</div>'
                     f'<div class="note {"good" if o["broken"] == 0 else "bad"}">'
                     f'{"✓ none broken" if o["broken"] == 0 else "✗ some broken"} · broken ≤ {pct(high)} (95 %)</div></div>')
    result = f'<div class="callout">{"".join(head)}</div>\n  <div class="tiles">{"".join(tiles)}</div>'

    rates = (f"<p>Wrong = verdict fix or drop. Dot: share of the {esc(' / '.join(str(s['n']) for s in stats))} images "
             "judged in both runs. Line: 95 % Wilson interval."
             + (" Dashed line: limit." if max_rate is not None and mode == "random" else "") + "</p>\n"
             f'  <div class="card">{rate_chart(stats, before, after, max_rate if mode == "random" else None)}</div>')

    legend = "".join(f'<span><span class="sw" style="background:{color}"></span>{icon} {name}</span>'
                     for _, icon, name, _, color in OUTCOMES)
    outcome_rows = "".join(
        f'<tr><td>{esc(s["label"])}</td><td>{s["n"]}</td>'
        + "".join(f'<td>{s["outcome"][key]}</td>' for key, *_ in OUTCOMES)
        + f'<td>{esc(fmt_p(s["p"]))}</td></tr>' for s in stats)
    outcomes = f"""<p>Each image judged once per run, in random order, run hidden. McNemar exact test: whether fixed and
     broken images are equally likely.</p>
  <div class="legend">{legend}</div>
  <div class="card">{outcome_chart(stats)}
    <div class="table-wrap"><table>
      <tr><th>Domain</th><th>Images</th>{"".join(f"<th>{icon} {name}</th>" for _, icon, name, _, _ in OUTCOMES)}<th>McNemar</th></tr>
      {outcome_rows}
    </table></div>
  </div>"""

    all_cats = category_list(stats)
    cats = [c for c in all_cats if any(c in s["cats"] and (s["cats"][c]["before"] or s["cats"][c]["after"]) for s in stats)]
    empty = [c for c in all_cats if c not in cats]
    if cats:
        axis_top = max(20, math.ceil(max(max(s["cats"][c]["before"], s["cats"][c]["after"])
                                         for s in stats for c in cats if c in s["cats"]) / 20) * 20)
        facets = "".join(f'<div class="card"><h3>{esc(s["label"])}</h3>'
                         f'{category_chart({**s, "before": before, "after": after}, [c for c in cats if c in s["cats"]], axis_top)}</div>'
                         for s in stats)
    else:
        facets = ""
    trans_head = "".join(f'<th>{esc(s["label"])}<br><span class="muted">{esc(before)} → {esc(after)}</span></th>'
                         f'<th>gone</th><th>kept</th><th>new</th>' for s in stats)
    trans_rows = "".join(
        f"<tr><td><code>{esc(c)}</code><br><span class='muted'>{esc(category_info(c, stats, cfg)[0])}</span></td>"
        + "".join(f"<td>{s['cats'][c]['before']} → {s['cats'][c]['after']}</td><td>{s['cats'][c]['gone']}</td>"
                  f"<td>{s['cats'][c]['kept']}</td><td>{s['cats'][c]['new']}</td>" if c in s["cats"]
                  else "<td>not reviewed</td><td></td><td></td><td></td>" for s in stats) + "</tr>"
        for c in all_cats)
    n_text = esc(" / ".join(str(s["n"]) for s in stats))
    categories = f"""<p>Images with each category, out of {n_text} per domain; one image can have several. Gone: in
     {esc(before)}, not in {esc(after)} for the same image. New: in {esc(after)}, not in {esc(before)}.
     No images with: {", ".join(f"<code>{esc(c)}</code>" for c in empty) if empty else "none"}.</p>
  <div class="legend"><span><span class="sw" style="background:var(--before)"></span>{esc(before)} (before)</span>
    <span><span class="sw" style="background:var(--after)"></span>{esc(after)} (after)</span></div>
  <div class="facets">{facets}</div>
  <div class="card table-wrap"><table>
    <tr><th>Category</th>{trans_head}</tr>{trans_rows}
  </table></div>"""

    counts = esc(", ".join(f'{s["label"]} {s["n"]}' for s in stats))
    if mode == "random":
        what = (f'{counts} random images of the whole set (sample seed {esc(stats[0]["sample_seed"])}), '
                f'both runs judged in one session')
        caveat = ("Images: a random sample of the whole set; the rates estimate each run's error rate. Both runs "
                  "judged in the same session; rates from other sessions are not directly comparable.")
    elif mode == "list":
        what = f"{counts} listed images"
        caveat = "Images: a chosen list, not a random sample; the rates describe these images only."
    else:
        what = f'{counts} random changed images (seed {esc(stats[0]["seed"])})'
        caveat = "Images: a random sample of the changed images only; the rates are higher than over the whole set."
    caveats = [reviewer_caveat(stats), "Blind: the page showed no run name or file name.",
               caveat + (f" {note}" if note else ""),
               "The verdict (ok / wrong) decides; categories only name the kind of error.", *cfg["caveats"]]
    caveat_list = "<ul>\n" + "".join(f"    <li>{esc(c)}</li>\n" for c in caveats) + "  </ul>"

    bodies = dict(zip(SECTIONS, (result, rates, outcomes, categories, scaling_section(stats, before, after, mode),
                                 acceptance_section(stats, before, after, mode, max_rate), caveat_list,
                                 f"<pre>{esc(cmd)}</pre>")))
    runs = stats[0]["runs"]
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Blind compare {esc(before)} vs. {esc(after)}</title>
<style>
{PAGE_CSS}{EXTRA_CSS}</style>
</head>
<body>
<main>
  <h1>Blind compare: {esc(before)} vs. {esc(after)}</h1>
  <p class="sub">{created} · {esc(before)} = {esc(Path(runs[before]).parent.name)}, {esc(after)} = {esc(Path(runs[after]).parent.name)} ·
     {what} · blind review</p>
{"".join(section(t, bodies[t]) for t in SECTIONS)}
  {page_footer(cfg)}
</main>
<div class="tip" id="tip"></div>
<script>
{TIP_JS}</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--review-dir", type=Path, action="append", required=True,
                        help="A compare-mode review dir (review_masks.py --compare). Repeat per domain.")
    parser.add_argument("--before", required=True, help="run name of the old run in the review (e.g. B)")
    parser.add_argument("--after", required=True, help="run name of the new run in the review (e.g. D)")
    parser.add_argument("--baseline", type=Path, action="append",
                        help="random-sample review dir of the old run, one per --review-dir (same order)")
    parser.add_argument("--mode", choices=MODES, default="changed",
                        help="what the judged images are: random changed images (default), a random sample "
                             "of the whole set (needs --sample), or a hand-picked list")
    parser.add_argument("--sample", type=Path, action="append",
                        help="random mode: the sample.json of each --review-dir (same order)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Part-1 config: limit, caveats, labels, footer (default: %(default)s).")
    parser.add_argument("--max-rate", type=float,
                        help="random mode: acceptance limit for the error rate (default: the config's review.max_rate)")
    parser.add_argument("--groups", type=Path,
                        help="CSV with columns file_name,group: one row per group and domain (e.g. fin_cut / ok)")
    parser.add_argument("--note", help="one sentence for the caveats, e.g. how a list was made")
    parser.add_argument("--out", type=Path, required=True, help="HTML file to write.")
    args = parser.parse_args()
    if args.baseline and len(args.baseline) != len(args.review_dir):
        parser.error("give one --baseline per --review-dir, or none")
    if args.mode == "random":
        if not args.sample or len(args.sample) != len(args.review_dir):
            parser.error("--mode random needs one --sample per --review-dir")
        if args.baseline:
            parser.error("--baseline only works with --mode changed")
    elif args.sample or args.max_rate is not None:
        parser.error("--sample and --max-rate only work with --mode random")
    if args.mode == "list" and args.baseline:
        parser.error("--baseline only works with --mode changed")

    groups = [None]
    if args.groups:
        members = {}
        with args.groups.open(newline="") as f:
            for row in csv.DictReader(f):
                members.setdefault(row["group"], set()).add(row["file_name"])
        groups = list(members.items())
    baselines = args.baseline or [None] * len(args.review_dir)
    samples = args.sample or [None] * len(args.review_dir)
    try:
        cfg = load_review_config(args.config)
        stats = [domain_stats(d, args.before, args.after, b, smp, g, cfg["labels"])
                 for d, b, smp in zip(args.review_dir, baselines, samples) for g in groups]
    except ValueError as e:
        raise SystemExit(str(e))
    max_rate = (cfg["max_rate"] if args.max_rate is None else args.max_rate) if args.mode == "random" else None
    for s in stats:
        if s["n"] < s["images"]:
            print(f'Note: {s["label"]}: only {s["n"]} of {s["images"]} images judged in both runs')
    cmd = " \\\n    ".join([f"python scripts/compare_report.py --mode {args.mode} --before {args.before} --after {args.after}"]
                           + [f"--review-dir {d}" for d in args.review_dir]
                           + [f"--baseline {b}" for b in args.baseline or []]
                           + [f"--sample {smp}" for smp in args.sample or []]
                           + [f"--config {args.config}"]
                           + ([f"--max-rate {args.max_rate}"] if args.max_rate is not None else [])
                           + ([f"--groups {args.groups}"] if args.groups else [])
                           + ([f"--note {json.dumps(args.note, ensure_ascii=False)}"] if args.note else [])
                           + [f"--out {args.out}"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(stats, cfg, args.before, args.after, date.today().isoformat(), cmd,
                               args.mode, max_rate, args.note))
    for s in stats:
        o, rb, ra = s["outcome"], s["rates"][args.before], s["rates"][args.after]
        line = (f'{s["label"]}: wrong {pct(rb["rate"])} -> {pct(ra["rate"])}, fixed {o["fixed"]}, '
                f'broken {o["broken"]}, still wrong {o["still"]}, ok in both {o["both_ok"]}, McNemar {fmt_p(s["p"])}')
        if "baseline" in s:
            line += f', expected error rate {pct(s["baseline"]["rate"])} -> {pct(s["baseline"]["expected"])}'
        print(line)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
