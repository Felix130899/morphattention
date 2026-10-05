"""Error-rate report for the random mask review: one self-contained HTML chart page.

Reads each random review sample (``review_masks.py --sample``: sample.json +
decisions.jsonl) and writes a page with, per domain: the error rate
(fix + drop) / judged with a 95 % Wilson interval, the verdict against the
acceptance limit, the count and rate per error category, and which
segment_fish.py setting usually fixes each category (with the value the run
used, read from its run_config.json). Nothing is fetched from outside; the page
works offline and contains no file names.

    python scripts/error_report.py \\
        --review-dir data/processed/segmented/B_merged_2026-10-03/full_body/review_random_seed0 \\
        --review-dir data/processed/segmented/B_merged_2026-10-03/Röntgen/review_random_seed0 \\
        --max-rate 0.10 --out vault/thesis-log/experiments/2026-10-04-mask-error-rate.html

Acceptance rule: a domain is accepted if its error rate (point estimate) is
<= --max-rate. The upper Wilson bound is shown next to it, but does not decide.
"""

import argparse
import html
import json
import math
from datetime import date
from fractions import Fraction
from pathlib import Path

from review_masks import CATEGORIES, summarize_sample, wilson

DOMAIN_LABELS = {"full_body": "Photos", "Röntgen": "X-rays"}

CATEGORY_TEXT = {
    "fin_cut": "more than ~5 % of a fin missing",
    "bleed": "background, shadow, tray, text or tag inside the mask",
    "merged": "two or more fish in one mask",
    "partial": "only part of a fish masked",
    "missed_fish": "a fish in the image has no mask",
    "wrong_object": "mask on something that is not a fish",
}

# Category -> (setting, run_config key or None, which way to turn it, trade-off).
# "Usually fixes" = what fixed this category in earlier stages
# (vault/thesis-log/experiments/mask-quality-history.md) or what the setting is built for.
FIXES = {
    "fin_cut": [
        ("--mask-threshold", "mask_threshold", "lower, e.g. −2: keeps translucent fin membrane",
         "masks grow everywhere, more bleed; −1 cut fins 22 → 4 on the dev set (stage 2)"),
        ("--candidate", "candidate", "largest instead of score",
         "picks the biggest SAM candidate, which can include background"),
        ("--margin-frac (training images)", None, "already 3 % when building training images",
         "reviewed overlays have no margin; it may cover thin edge losses counted here"),
    ],
    "bleed": [
        ("--bleed-outside-frac", "bleed_outside_frac", "lower, e.g. 0.01: fallback fires on smaller leaks",
         "replacement masks are tighter, a few more fin_cut / partial (stage 4)"),
        ("--bleed-max-box-edge", "bleed_max_box_edge", "lower: stricter inverted-background masks",
         "more images stay unresolved"),
        ("(none)", None, "tags tied to the fish, X-ray plates and rods",
         "no setting separates them; drop the image"),
    ],
    "merged": [
        ("--dino-box-threshold", "dino_box_threshold", "higher: fewer boxes spanning two fish",
         "can lose small fish (missed_fish)"),
    ],
    "partial": [
        ("--candidate", "candidate", "largest instead of score",
         "can include background (bleed)"),
        ("--mask-threshold", "mask_threshold", "lower", "more bleed"),
    ],
    "missed_fish": [
        ("--dino-box-threshold", "dino_box_threshold", "lower, e.g. 0.25–0.30: more boxes",
         "more wrong_object (rulers, labels)"),
        ("--dino-text", "dino_text", "add terms, e.g. 'fish. eel. ray.'",
         "more boxes on non-fish"),
        ("--dedup-containment", "dedup_containment", "higher: keeps a small fish lying inside another's box",
         "more duplicate masks of the same fish"),
    ],
    "wrong_object": [
        ("--dino-box-threshold", "dino_box_threshold", "higher", "more missed_fish"),
        ("--dino-text", "dino_text", "narrower prompt", "more missed_fish"),
    ],
}


# Page style and hover tooltips, shared with compare_report.py. Colours: the
# dataviz reference palette (light + dark), status colours for good / bad.
PAGE_CSS = """:root {
  color-scheme: light;
  --page: #f9f9f7; --surface-1: #fcfcfb;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #898781;
  --grid: #e1e0d9; --border: rgba(11,11,11,0.10);
  --series-1: #2a78d6; --series-2: #eb6834;
  --callout: #eaf2fc; --good-text: #006300; --bad-text: #b3261e;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d; --surface-1: #1a1a19;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #898781;
    --grid: #2c2c2a; --border: rgba(255,255,255,0.10);
    --series-1: #3987e5; --series-2: #d95926;
    --callout: #16243a; --good-text: #0ca30c; --bad-text: #e66767;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface-1: #1a1a19;
  --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #898781;
  --grid: #2c2c2a; --border: rgba(255,255,255,0.10);
  --series-1: #3987e5; --series-2: #d95926;
  --callout: #16243a; --good-text: #0ca30c; --bad-text: #e66767;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--text-primary);
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 980px; margin: 0 auto; padding: 32px 16px 64px; }
h1 { font-size: 26px; margin: 0 0 4px; }
h2 { font-size: 19px; margin: 40px 0 6px; }
p, li { color: var(--text-secondary); }
.sub { color: var(--text-muted); margin: 0 0 20px; }
.card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 18px; }
.callout { background: var(--callout); border: 1px solid var(--border); border-radius: 10px; padding: 16px 18px; }
.callout p { color: var(--text-primary); margin: 0; }
.callout p + p { margin-top: 8px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-top: 16px; }
.tile .label { color: var(--text-muted); font-size: 13px; }
.tile .value { font-size: 28px; font-weight: 650; margin: 2px 0; font-variant-numeric: tabular-nums; }
.note { font-size: 13px; }
.good { color: var(--good-text); } .bad { color: var(--bad-text); } .muted { color: var(--text-muted); }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
th, td { padding: 6px 8px; text-align: right; border-bottom: 1px solid var(--grid); vertical-align: top; }
th:first-child, td:first-child { text-align: left; }
th { color: var(--text-muted); font-weight: 500; }
td { font-variant-numeric: tabular-nums; }
td.left, th.left { text-align: left; }
.table-wrap { overflow-x: auto; }
.legend { display: flex; gap: 16px; flex-wrap: wrap; font-size: 13px; color: var(--text-secondary); margin: 4px 0 10px; }
.sw { display: inline-block; width: 12px; height: 12px; border-radius: 3px; vertical-align: -1px; margin-right: 5px; }
svg { display: block; width: 100%; height: auto; overflow: visible; }
svg text { font-family: inherit; }
.tip { position: fixed; pointer-events: none; z-index: 10; display: none; max-width: 320px;
  background: var(--surface-1); color: var(--text-primary); border: 1px solid var(--border);
  border-radius: 8px; padding: 6px 9px; font-size: 13px; box-shadow: 0 4px 14px rgba(0,0,0,.15); }
details { margin-top: 10px; }
summary { cursor: pointer; color: var(--text-primary); font-weight: 500; }
code { font-size: 13px; }
pre { background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px; padding: 12px; overflow-x: auto; font-size: 13px; }
footer { margin-top: 48px; color: var(--text-muted); font-size: 13px; }
"""

TIP_JS = """const tip = document.getElementById("tip");
document.querySelectorAll("[data-tip]").forEach(g => {
  g.addEventListener("mousemove", ev => {
    tip.textContent = g.dataset.tip; tip.style.display = "block";
    tip.style.left = Math.min(ev.clientX + 12, window.innerWidth - tip.offsetWidth - 8) + "px";
    tip.style.top = (ev.clientY + 12) + "px";
  });
  g.addEventListener("mouseleave", () => { tip.style.display = "none"; });
});
"""


def domain_key(review_dir):
    return Path(review_dir).resolve().parent.name


def run_settings(run_dir):
    """Settings the masks were made with; for a merged run, base overlaid with the patch run
    (drop-only merges have no patch run)."""
    path = Path(run_dir) / "run_config.json"
    if not path.exists():
        return {}
    config = json.loads(path.read_text())
    if "merged" in config:
        return {**config["base_config"]["settings"], **((config["patch_config"] or {}).get("settings") or {})}
    return config.get("settings", {})


def accepted(bad, judged, max_rate):
    """Point estimate <= limit, exact (30 / 300 at a 10 % limit is accepted)."""
    return judged > 0 and Fraction(bad, judged) <= Fraction(str(max_rate))


def domain_stats(review_dir, max_rate):
    row = summarize_sample(Path(review_dir))
    judged = row["judged"]
    bad = row["fix"] + row["drop"]
    low, high = wilson(bad, judged)
    key = domain_key(review_dir)
    cats = {}
    for c in CATEGORIES:
        c_low, c_high = wilson(row[c], judged)
        cats[c] = {"k": row[c], "rate": row[c] / judged if judged else 0.0, "low": c_low, "high": c_high}
    return {
        "key": key, "label": DOMAIN_LABELS.get(key, key), "review_dir": str(review_dir),
        "run_dir": row["run_dir"], "seed": row["seed"],
        "population": row["population"], "sample": row["sample"], "judged": judged,
        "ok": row["ok"], "fix": row["fix"], "drop": row["drop"], "bad": bad,
        "rate": bad / judged if judged else 0.0, "low": low, "high": high,
        "accepted": accepted(bad, judged, max_rate), "complete": judged == row["sample"],
        "est": [round(v * row["population"]) for v in (bad / judged if judged else 0.0, low, high)],
        "cats": cats, "settings": run_settings(row["run_dir"]),
    }


def two_proportion_p(k1, n1, k2, n2):
    """Two-sided p-value of the pooled two-proportion z-test."""
    if not n1 or not n2:
        return 1.0
    pooled = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (k1 / n1 - k2 / n2) / se
    return math.erfc(abs(z) / math.sqrt(2))


def nice_max(v):
    """Smallest of 1/2/5 x 10^k steps (as a fraction) giving an axis end >= v with <= 6 ticks."""
    for step in (0.01, 0.02, 0.05, 0.1, 0.2, 0.25):
        top = math.ceil(v / step - 1e-9) * step
        if top / step <= 6:
            return top, step
    return 1.0, 0.25


def pct(v, digits=1):
    return f"{100 * v:.{digits}f} %"


def esc(text):
    return html.escape(str(text), quote=True)


def interval_row(svg, x, y, d, color, tip, label=None):
    """Whisker (Wilson interval) + dot (point estimate) + invisible hit target."""
    x0, x1, xp = x(d["low"]), x(d["high"]), x(d["rate"])
    svg.append(f'<g data-tip="{esc(tip)}">'
               f'<rect x="{x0 - 8:.1f}" y="{y - 12}" width="{x1 - x0 + 16:.1f}" height="24" fill="transparent"/>'
               f'<line x1="{x0:.1f}" x2="{x1:.1f}" y1="{y}" y2="{y}" stroke="{color}" stroke-width="2"/>'
               f'<line x1="{x0:.1f}" x2="{x0:.1f}" y1="{y - 4}" y2="{y + 4}" stroke="{color}" stroke-width="2"/>'
               f'<line x1="{x1:.1f}" x2="{x1:.1f}" y1="{y - 4}" y2="{y + 4}" stroke="{color}" stroke-width="2"/>'
               f'<circle cx="{xp:.1f}" cy="{y}" r="5" fill="{color}" stroke="var(--surface-1)" stroke-width="2"/>'
               "</g>")
    if label:
        svg.append(f'<text x="{x1 + 10:.1f}" y="{y + 4}" font-size="12" fill="var(--text-secondary)">{esc(label)}</text>')


def axis(svg, x, top, bottom, axis_top, step):
    n = round(axis_top / step)
    for i in range(n + 1):
        t = i * step
        svg.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top}" y2="{bottom}" stroke="var(--grid)"/>')
        svg.append(f'<text x="{x(t):.1f}" y="{bottom + 14}" text-anchor="middle" font-size="11" '
                   f'fill="var(--text-muted)">{100 * t:.0f} %</text>')


def rate_chart(stats, max_rate):
    W, L, R, row_h, top = 640, 70, 150, 40, 22
    axis_top, step = nice_max(max(max(s["high"] for s in stats), max_rate) * 1.05)
    H = top + row_h * len(stats) + 22
    x = lambda v: L + v / axis_top * (W - L - R)  # noqa: E731
    svg = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Error rate per domain with 95 % Wilson interval">']
    axis(svg, x, top - 6, H - 22, axis_top, step)
    svg.append(f'<line x1="{x(max_rate):.1f}" x2="{x(max_rate):.1f}" y1="{top - 10}" y2="{H - 22}" '
               'stroke="var(--text-secondary)" stroke-width="1.5" stroke-dasharray="4 3"/>')
    svg.append(f'<text x="{x(max_rate):.1f}" y="{top - 14}" text-anchor="middle" font-size="11" '
               f'fill="var(--text-secondary)">limit {pct(max_rate, 0)}</text>')
    for i, s in enumerate(stats):
        y = top + row_h * i + row_h // 2
        color = f"var(--series-{i + 1})"
        svg.append(f'<text x="{L - 12}" y="{y + 4}" text-anchor="end" font-size="13" '
                   f'fill="var(--text-primary)">{esc(s["label"])}</text>')
        tip = (f'{s["label"]}: {s["bad"]} of {s["judged"]} wrong = {pct(s["rate"])}, '
               f'95 % CI {pct(s["low"])} – {pct(s["high"])}')
        interval_row(svg, x, y, s, color, tip, f'{pct(s["rate"])}  ({pct(s["low"])} – {pct(s["high"])})')
    svg.append("</svg>")
    return "".join(svg)


def category_chart(stats):
    W, L, R, sub_h, gap, top = 640, 110, 70, 18, 14, 6
    axis_top, step = nice_max(max(s["cats"][c]["high"] for s in stats for c in CATEGORIES) * 1.05)
    group_h = sub_h * len(stats) + gap
    H = top + group_h * len(CATEGORIES) + 18
    x = lambda v: L + v / axis_top * (W - L - R)  # noqa: E731
    svg = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Rate per error category and domain">']
    axis(svg, x, top - 2, H - 18, axis_top, step)
    for g, c in enumerate(CATEGORIES):
        y0 = top + group_h * g
        svg.append(f'<text x="{L - 12}" y="{y0 + sub_h * len(stats) / 2 + 4:.1f}" text-anchor="end" '
                   f'font-size="13" fill="var(--text-primary)">{c}</text>')
        for i, s in enumerate(stats):
            d = s["cats"][c]
            y = y0 + sub_h * i + sub_h // 2
            tip = (f'{s["label"]}, {c}: {d["k"]} of {s["judged"]} = {pct(d["rate"])}, '
                   f'95 % CI {pct(d["low"])} – {pct(d["high"])}')
            interval_row(svg, x, y, d, f"var(--series-{i + 1})", tip)
            svg.append(f'<text x="{W - R + 12}" y="{y + 4}" font-size="12" '
                       f'fill="var(--text-secondary)">{d["k"]} / {s["judged"]}</text>')
    svg.append("</svg>")
    return "".join(svg)


def fmt_setting(stats, key):
    if key is None:
        return "–"
    values = [s["settings"].get(key) for s in stats]
    if all(v is None for v in values):
        return "?"
    if len(set(map(str, values))) == 1:
        return esc(values[0])
    return " / ".join(f'{esc(v)} ({esc(s["label"])})' for v, s in zip(values, stats))


def verdict_sentence(s, max_rate):
    limit = pct(max_rate, 0)
    if s["accepted"]:
        text = f'<strong>{esc(s["label"])}: accepted</strong>, {pct(s["rate"])} ≤ {limit}.'
        if s["high"] > max_rate:
            text += (f' The 95 % interval reaches {pct(s["high"])}, so the true rate may still be above '
                     f'{limit}{" (the estimate sits exactly on the limit)" if Fraction(s["bad"], s["judged"]) == Fraction(str(max_rate)) else ""}.')
        else:
            text += f" Even the upper end of the 95 % interval ({pct(s['high'])}) is below the limit."
    else:
        text = (f'<strong>{esc(s["label"])}: not accepted</strong>, {pct(s["rate"])} > {limit}. '
                "Change the setting for the largest category, rerun, blind compare, review a fresh sample.")
    if not s["complete"]:
        text += f' Only {s["judged"]} of {s["sample"]} sample images judged so far.'
    return text


def render(stats, max_rate, created, out):
    limit = pct(max_rate, 0)
    legend = "".join(f'<span><span class="sw" style="background:var(--series-{i + 1})"></span>{esc(s["label"])}</span>'
                     for i, s in enumerate(stats))
    tiles = []
    for s in stats:
        mark = "✓ accepted" if s["accepted"] else "✗ not accepted"
        tiles.append(f'<div class="card tile"><div class="label">{esc(s["label"])}: wrong masks</div>'
                     f'<div class="value">{pct(s["rate"])}</div>'
                     f'<div class="note {"good" if s["accepted"] else "bad"}">{mark} · CI {pct(s["low"])} – {pct(s["high"])}</div></div>')
    for s in stats:
        b = s["cats"]["bleed"]
        tiles.append(f'<div class="card tile"><div class="label">{esc(s["label"])}: bleed (can leak labels)</div>'
                     f'<div class="value">{pct(b["rate"])}</div>'
                     f'<div class="note muted">{b["k"]} of {s["judged"]} · CI {pct(b["low"])} – {pct(b["high"])}</div></div>')

    rows = "".join(
        f'<tr><td>{esc(s["label"])}</td><td>{s["population"]:,}</td><td>{s["judged"]}</td><td>{s["ok"]}</td>'
        f'<td>{s["fix"]}</td><td>{s["drop"]}</td><td>{pct(s["rate"])}</td>'
        f'<td>{pct(s["low"])} – {pct(s["high"])}</td><td>≈ {s["est"][0]:,} ({s["est"][1]:,} – {s["est"][2]:,})</td>'
        f'<td>{"✓ accepted" if s["accepted"] else "✗ not accepted"}</td></tr>' for s in stats)

    cat_head = "".join(f"<th>{esc(s['label'])}</th>" for s in stats)
    cat_rows = "".join(
        f"<tr><td><code>{c}</code></td><td class='left'>{CATEGORY_TEXT[c]}</td>"
        + "".join(f"<td>{s['cats'][c]['k']} ({pct(s['cats'][c]['rate'])}, CI {pct(s['cats'][c]['low'])} – "
                  f"{pct(s['cats'][c]['high'])})</td>" for s in stats) + "</tr>" for c in CATEGORIES)

    fix_rows = []
    for c in CATEGORIES:
        counts = " / ".join(str(s["cats"][c]["k"]) for s in stats)
        for j, (flag, key, direction, tradeoff) in enumerate(FIXES[c]):
            first = (f'<td rowspan="{len(FIXES[c])}"><code>{c}</code><br><span class="muted">{counts}</span></td>'
                     if j == 0 else "")
            fix_rows.append(f"<tr>{first}<td class='left'><code>{esc(flag)}</code></td>"
                            f"<td class='left'>{fmt_setting(stats, key)}</td><td class='left'>{esc(direction)}</td>"
                            f"<td class='left'>{esc(tradeoff)}</td></tr>")

    compare = ""
    if len(stats) == 2:
        a, b = stats
        p = two_proportion_p(a["bad"], a["judged"], b["bad"], b["judged"])
        diff = abs(a["rate"] - b["rate"]) * 100
        sig = "a real difference at the 5 % level" if p < 0.05 else "not a significant difference"
        compare = (f'<h2>Is one domain worse?</h2><p>{esc(b["label"])} {pct(b["rate"])} vs. {esc(a["label"])} '
                   f'{pct(a["rate"])}: {diff:.1f} percentage points, p = {p:.2f} (two-proportion z-test), '
                   f"so {sig}. The intervals overlap; with 300 images per domain a gap this size can be chance.</p>")

    order = ", ".join(f"{s['label']} {s['sample']}" for s in stats)
    run_names = sorted({Path(s["run_dir"]).parent.name for s in stats})
    seeds = sorted({str(s["seed"]) for s in stats})
    cmd = " \\\n    ".join(["python scripts/error_report.py"]
                           + [f"--review-dir {esc(s['review_dir'])}" for s in stats]
                           + [f"--max-rate {max_rate}", f"--out {esc(out)}"])

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mask error rate</title>
<style>
{PAGE_CSS}</style>
</head>
<body>
<main>
  <h1>Mask error rate</h1>
  <p class="sub">{created} · {esc(", ".join(run_names))} · random sample ({esc(order)}), seed {esc(", ".join(seeds))} · blind review by Neo · NHM Wien imagery</p>

  <div class="callout">
    {"".join(f"<p>{verdict_sentence(s, max_rate)}</p>" for s in stats)}
  </div>

  <div class="tiles">{"".join(tiles)}</div>

  <h2>Acceptance rule</h2>
  <p>A mask set is accepted if its error rate is <strong>≤ {limit}</strong> (Neo, 2026-10-04). Error rate =
     images judged <em>fix</em> or <em>drop</em> / images judged, in a seeded random sample of the usable images
     (dropped and skipped images are not in the population). An image counts as wrong if any of its masks is wrong.
     The rule judges the point estimate; the 95 % Wilson interval shows how far the true rate could be off.</p>
  <div class="card table-wrap"><table>
    <tr><th>Domain</th><th>Usable images</th><th>Judged</th><th>ok</th><th>fix</th><th>drop</th>
        <th>Error rate</th><th>95 % CI</th><th>Wrong images in the full set</th><th>Verdict</th></tr>
    {rows}
  </table></div>

  <h2>Error rate per domain</h2>
  <p>Dot = error rate in the sample, line = 95 % Wilson interval, dashed line = limit.</p>
  <div class="card">{rate_chart(stats, max_rate)}</div>
  {compare}

  <h2>Which errors?</h2>
  <p>Share of sample images with each error (an image can have several). Counts are small, so the intervals
     overlap a lot: the categories cannot be ranked against each other, only told apart from zero.
     Only <code>bleed</code> can leak label information (printed names, tags) into training, which is the
     Clever Hans shortcut the masks exist to remove; the other categories lose morphology or a fish, not labels.</p>
  <div class="legend">{legend}</div>
  <div class="card">{category_chart(stats)}
    <details><summary>Table view</summary><div class="table-wrap"><table>
      <tr><th>Category</th><th class="left">Meaning</th>{cat_head}</tr>{cat_rows}
    </table></div></details>
  </div>

  <h2>Which setting fixes each category</h2>
  <p>The <code>segment_fish.py</code> setting to change if a category has to come down, the value these masks were
     made with, and what it costs. Any change means: rerun, blind compare against this run, then review a
     <strong>fresh</strong> random sample (new seed), not this one.</p>
  <div class="card table-wrap"><table>
    <tr><th>Category<br><span class="muted">{esc(" / ".join(s["label"] for s in stats))}</span></th>
        <th class="left">Setting</th><th class="left">Current</th><th class="left">Turn it</th><th class="left">Cost / evidence</th></tr>
    {"".join(fix_rows)}
  </table></div>

  <h2>Caveats</h2>
  <ul>
    <li>One reviewer (Neo); categories were judged by eye against the mask policy
        (<code>decisions/2026-10-01-mask-policy.md</code>).</li>
    <li>The overlays were reviewed <strong>without</strong> the 3 % margin that is added when building training
        images, so <code>fin_cut</code> is an upper bound for what the model sees.</li>
    <li>The sample is random over images, not stratified by genus; the rate says nothing about single genera.</li>
    <li>The rate applies to the usable set only. Dropped images (status "dropped") are excluded by design.</li>
  </ul>

  <h2>Rebuild this page</h2>
  <pre>{cmd}</pre>

  <footer>Images: Naturhistorisches Museum Wien (NHM Wien), Fish Collection. Internal report, not for publication.</footer>
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
                        help="A random-sample review dir (review_masks.py --sample). Repeat per domain.")
    parser.add_argument("--max-rate", type=float, default=0.10,
                        help="Acceptance limit for the error rate, inclusive (default: %(default)s).")
    parser.add_argument("--out", type=Path, required=True, help="HTML file to write.")
    args = parser.parse_args()

    stats = [domain_stats(d, args.max_rate) for d in args.review_dir]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(stats, args.max_rate, date.today().isoformat(), args.out))
    for s in stats:
        print(f'{s["label"]}: {s["bad"]} / {s["judged"]} = {pct(s["rate"])} '
              f'(95 % CI {pct(s["low"])} - {pct(s["high"])}) -> {"accepted" if s["accepted"] else "NOT accepted"}'
              f' at <= {pct(args.max_rate, 0)}{"" if s["complete"] else "  [sample not fully judged]"}')
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
