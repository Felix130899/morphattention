"""Error-rate report for the random mask review: one self-contained HTML chart page.

Reads each random review sample (``review_masks.py --sample``: sample.json,
review.json, decisions.jsonl) and writes a page with, per domain: the error
rate (fix + drop) / judged with a 95 % Wilson interval, the verdict against the
acceptance limit, the count and rate per error category, and the settings that
fix each category (with the value the run used, read from its run_config.json).
Nothing is fetched from outside; the page works offline and contains no file names.

The categories and the reviewers come from each review dir's review.json; the
category advice ("fixes"), the limit (review.max_rate), extra caveats, the
domain labels and the footer line come from ``--config``
(default pipeline/config.yaml).

    python scripts/review/error_report.py --config pipeline/config_thesis_nhm.yaml \\
        --review-dir data/processed/segmented/B_merged_2026-10-03/full_body/review_random_seed0 \\
        --review-dir data/processed/segmented/B_merged_2026-10-03/Röntgen/review_random_seed0 \\
        --out vault/thesis-log/experiments/2026-10-04-mask-error-rate.html

Acceptance rule: a domain is accepted if its error rate (point estimate) is
<= the limit. The upper Wilson bound is shown next to it, but does not decide.

The page always has the same sections in the same order (SECTIONS); a section
that does not apply says so in one line.
"""

import argparse
import html
import json
import math
from datetime import date
from fractions import Fraction
from pathlib import Path

from review_masks import DEFAULT_CONFIG, load_review_config, read_review_meta, summarize_sample, wilson

SECTIONS = ("Result", "Sample and acceptance rule", "Error rate per domain", "Difference between domains",
            "Error categories", "Settings per category", "Caveats", "Rebuild this page")


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


def domain_stats(review_dir, max_rate, labels=None):
    """Stats of one random-sample review dir; ``labels``: {domain: name on the page} (config)."""
    row = summarize_sample(Path(review_dir))
    meta = read_review_meta(review_dir)
    judged = row["judged"]
    bad = row["fix"] + row["drop"]
    low, high = wilson(bad, judged)
    key = domain_key(review_dir)
    cats = {}
    for c in meta["categories"]:
        k = row[c["name"]]
        c_low, c_high = wilson(k, judged)
        cats[c["name"]] = {"k": k, "rate": k / judged if judged else 0.0, "low": c_low, "high": c_high}
    return {
        "key": key, "label": (labels or {}).get(key, key), "review_dir": str(review_dir),
        "run_dir": row["run_dir"], "seed": row["seed"],
        "population": row["population"], "sample": row["sample"], "judged": judged,
        "ok": row["ok"], "fix": row["fix"], "drop": row["drop"], "bad": bad,
        "rate": bad / judged if judged else 0.0, "low": low, "high": high,
        "accepted": accepted(bad, judged, max_rate), "complete": judged == row["sample"],
        "est": [round(v * row["population"]) for v in (bad / judged if judged else 0.0, low, high)],
        "cats": cats, "cat_text": {c["name"]: c["text"] for c in meta["categories"]},
        "reviewers": meta["reviewers"], "settings": run_settings(row["run_dir"]),
    }


def category_list(stats):
    """Category names of all domains, in key order (first domain's, then any new ones)."""
    out = []
    for s in stats:
        out += [c for c in s["cat_text"] if c not in out]
    return out


def category_info(name, stats, cfg):
    """(text, highlight, fixes): the text the reviewer had (review.json), the advice from the config."""
    conf = next((c for c in cfg["categories"] if c["name"] == name), None)
    text = next((s["cat_text"][name] for s in stats if name in s["cat_text"]), "") or (conf or {}).get("text", "")
    return text, bool(conf and conf["highlight"]), conf["fixes"] if conf else []


def reviewer_caveat(stats):
    names = []
    for s in stats:
        names += [r for r in s["reviewers"] if r not in names]
    if not names:
        return "Reviewer not recorded."
    if len(names) == 1:
        return f"One reviewer ({names[0]})."
    return f"{len(names)} reviewers ({', '.join(names)}); results not split per reviewer."


def page_footer(cfg):
    return f"<footer>{esc(cfg['attribution'])}</footer>" if cfg.get("attribution") else ""


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


def fmt_p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


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


def category_chart(stats, cats):
    W, L, R, sub_h, gap, top = 640, 110, 70, 18, 14, 6
    axis_top, step = nice_max(max([s["cats"][c]["high"] for s in stats for c in cats if c in s["cats"]] or [0.01]) * 1.05)
    group_h = sub_h * len(stats) + gap
    H = top + group_h * len(cats) + 18
    x = lambda v: L + v / axis_top * (W - L - R)  # noqa: E731
    svg = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Rate per error category and domain">']
    axis(svg, x, top - 2, H - 18, axis_top, step)
    for g, c in enumerate(cats):
        y0 = top + group_h * g
        svg.append(f'<text x="{L - 12}" y="{y0 + sub_h * len(stats) / 2 + 4:.1f}" text-anchor="end" '
                   f'font-size="13" fill="var(--text-primary)">{esc(c)}</text>')
        for i, s in enumerate(stats):
            y = y0 + sub_h * i + sub_h // 2
            if c not in s["cats"]:
                svg.append(f'<text x="{W - R + 12}" y="{y + 4}" font-size="12" fill="var(--text-muted)">not reviewed</text>')
                continue
            d = s["cats"][c]
            tip = (f'{s["label"]}, {c}: {d["k"]} of {s["judged"]} = {pct(d["rate"])}, '
                   f'95 % CI {pct(d["low"])} – {pct(d["high"])}')
            interval_row(svg, x, y, d, f"var(--series-{i + 1})", tip)
            svg.append(f'<text x="{W - R + 12}" y="{y + 4}" font-size="12" '
                       f'fill="var(--text-secondary)">{d["k"]} / {s["judged"]}</text>')
    svg.append("</svg>")
    return "".join(svg)


def fmt_setting(stats, key):
    """The value each run used for a setting; "–" if it is not a run setting."""
    if not any(key in s["settings"] for s in stats):
        return "–"
    values = ["off" if s["settings"].get(key) is None else s["settings"][key] for s in stats]
    if len(set(map(str, values))) == 1:
        return esc(values[0])
    return " / ".join(f'{esc(v)} ({esc(s["label"])})' for v, s in zip(values, stats))


def verdict_sentence(s, max_rate):
    limit = pct(max_rate, 0)
    ci = f'95 % CI {pct(s["low"])} – {pct(s["high"])}'
    if s["accepted"]:
        text = f'<strong>{esc(s["label"])}: accepted.</strong> Error rate {pct(s["rate"])} ≤ limit {limit}; {ci}.'
        if s["high"] > max_rate:
            text += " Upper CI bound above the limit."
            if Fraction(s["bad"], s["judged"]) == Fraction(str(max_rate)):
                text += " Estimate equals the limit."
        else:
            text += " Upper CI bound below the limit."
    else:
        text = (f'<strong>{esc(s["label"])}: not accepted.</strong> Error rate {pct(s["rate"])} > limit {limit}; {ci}. '
                "Next: change the setting for the largest category, rerun, blind compare, review a fresh sample.")
    if not s["complete"]:
        text += f' Judged: {s["judged"]} of {s["sample"]} sample images.'
    return text


def section(title, body):
    return f"\n  <h2>{esc(title)}</h2>\n  {body}\n"


def render(stats, cfg, max_rate, created, cmd):
    """The page; ``cfg``: review_masks.load_review_config(); sections in SECTIONS order."""
    limit = pct(max_rate, 0)
    cats = category_list(stats)
    info = {c: category_info(c, stats, cfg) for c in cats}

    tiles = []
    for s in stats:
        mark = "✓ accepted" if s["accepted"] else "✗ not accepted"
        tiles.append(f'<div class="card tile"><div class="label">{esc(s["label"])}: wrong masks</div>'
                     f'<div class="value">{pct(s["rate"])}</div>'
                     f'<div class="note {"good" if s["accepted"] else "bad"}">{mark} · CI {pct(s["low"])} – {pct(s["high"])}</div></div>')
    for c in (c for c in cats if info[c][1]):
        for s in (s for s in stats if c in s["cats"]):
            d = s["cats"][c]
            tiles.append(f'<div class="card tile"><div class="label">{esc(s["label"])}: {esc(c)}</div>'
                         f'<div class="value">{pct(d["rate"])}</div>'
                         f'<div class="note muted">{d["k"]} of {s["judged"]} · CI {pct(d["low"])} – {pct(d["high"])}</div></div>')
    result = (f'<div class="callout">{"".join(f"<p>{verdict_sentence(s, max_rate)}</p>" for s in stats)}</div>\n'
              f'  <div class="tiles">{"".join(tiles)}</div>')

    rows = "".join(
        f'<tr><td>{esc(s["label"])}</td><td>{s["population"]:,}</td><td>{s["judged"]}</td><td>{s["ok"]}</td>'
        f'<td>{s["fix"]}</td><td>{s["drop"]}</td><td>{pct(s["rate"])}</td>'
        f'<td>{pct(s["low"])} – {pct(s["high"])}</td><td>≈ {s["est"][0]:,} ({s["est"][1]:,} – {s["est"][2]:,})</td>'
        f'<td>{"✓ accepted" if s["accepted"] else "✗ not accepted"}</td></tr>' for s in stats)
    rule = f"""<p>Accepted if the error rate is <strong>≤ {limit}</strong> (point estimate). Error rate = images judged
     <em>fix</em> or <em>drop</em> / images judged, in a seeded random sample of the usable images (dropped and
     skipped images are not in the population). An image is wrong if any of its masks is wrong. The 95 % Wilson
     interval is shown, not used for the decision.</p>
  <div class="card table-wrap"><table>
    <tr><th>Domain</th><th>Usable images</th><th>Judged</th><th>ok</th><th>fix</th><th>drop</th>
        <th>Error rate</th><th>95 % CI</th><th>Wrong images in the full set</th><th>Verdict</th></tr>
    {rows}
  </table></div>"""

    rate = ("<p>Dot: error rate in the sample. Line: 95 % Wilson interval. Dashed line: limit.</p>\n"
            f'  <div class="card">{rate_chart(stats, max_rate)}</div>')

    if len(stats) == 2:
        a, b = stats
        p = two_proportion_p(a["bad"], a["judged"], b["bad"], b["judged"])
        diff = abs(a["rate"] - b["rate"]) * 100
        difference = (f'<p>{esc(b["label"])} {pct(b["rate"])} vs. {esc(a["label"])} {pct(a["rate"])}: difference '
                      f'{diff:.1f} points, {fmt_p(p)} (two-proportion z-test): '
                      f'{"significant" if p < 0.05 else "not significant"} at 5 %.</p>')
    else:
        difference = f"<p>Not applicable: needs exactly 2 domains, this report has {len(stats)}.</p>"

    legend = "".join(f'<span><span class="sw" style="background:var(--series-{i + 1})"></span>{esc(s["label"])}</span>'
                     for i, s in enumerate(stats))
    cat_head = "".join(f"<th>{esc(s['label'])}</th>" for s in stats)
    cat_rows = "".join(
        f"<tr><td><code>{esc(c)}</code></td><td class='left'>{esc(info[c][0])}</td>"
        + "".join(f"<td>{s['cats'][c]['k']} ({pct(s['cats'][c]['rate'])}, CI {pct(s['cats'][c]['low'])} – "
                  f"{pct(s['cats'][c]['high'])})</td>" if c in s["cats"] else "<td>not reviewed</td>" for s in stats)
        + "</tr>" for c in cats)
    categories = f"""<p>Share of judged sample images with each category; one image can have several. Interval: 95 % Wilson.
     Categories with overlapping intervals are not ranked against each other.</p>
  <div class="legend">{legend}</div>
  <div class="card">{category_chart(stats, cats)}
    <div class="table-wrap"><table>
      <tr><th>Category</th><th class="left">Meaning</th>{cat_head}</tr>{cat_rows}
    </table></div>
  </div>"""

    fix_rows = []
    for c in cats:
        counts = " / ".join(str(s["cats"][c]["k"]) if c in s["cats"] else "–" for s in stats)
        fixes = info[c][2]
        if not fixes:
            fix_rows.append(f"<tr><td><code>{esc(c)}</code><br><span class='muted'>{counts}</span></td>"
                            "<td class='left' colspan='4'>no advice in the config</td></tr>")
        for j, f in enumerate(fixes):
            first = (f'<td rowspan="{len(fixes)}"><code>{esc(c)}</code><br><span class="muted">{counts}</span></td>'
                     if j == 0 else "")
            fix_rows.append(f"<tr>{first}<td class='left'><code>{esc(f['setting'])}</code></td>"
                            f"<td class='left'>{fmt_setting(stats, f['setting'])}</td><td class='left'>{esc(f['turn'])}</td>"
                            f"<td class='left'>{esc(f['cost'])}</td></tr>")
    settings = f"""<p>Setting to change per category (from the config), the value these runs used, direction, cost.
     After a change: rerun under a new name, blind compare against this run, review a fresh random sample (new seed).</p>
  <div class="card table-wrap"><table>
    <tr><th>Category<br><span class="muted">{esc(" / ".join(s["label"] for s in stats))}</span></th>
        <th class="left">Setting</th><th class="left">Used</th><th class="left">Turn it</th><th class="left">Cost / evidence</th></tr>
    {"".join(fix_rows)}
  </table></div>"""

    caveats = [reviewer_caveat(stats),
               "Blind: the page showed no file name, QA flags or fish count.",
               "The sample is random over images, not stratified by class: the rate does not apply to single classes.",
               "The rate applies to the usable set only; dropped and skipped images are excluded by design.",
               *cfg["caveats"]]
    caveat_list = "<ul>\n" + "".join(f"    <li>{esc(c)}</li>\n" for c in caveats) + "  </ul>"

    order = ", ".join(f"{s['label']} {s['sample']}" for s in stats)
    run_names = sorted({Path(s["run_dir"]).parent.name for s in stats})
    seeds = sorted({str(s["seed"]) for s in stats})
    bodies = dict(zip(SECTIONS, (result, rule, rate, difference, categories, settings, caveat_list,
                                 f"<pre>{esc(cmd)}</pre>")))
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
  <p class="sub">{created} · runs {esc(", ".join(run_names))} · random sample ({esc(order)}), seed {esc(", ".join(seeds))} · blind review</p>
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
                        help="A random-sample review dir (review_masks.py --sample). Repeat per domain.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Part-1 config: category advice, limit, caveats, labels, footer (default: %(default)s).")
    parser.add_argument("--max-rate", type=float,
                        help="Acceptance limit for the error rate, inclusive (default: the config's review.max_rate).")
    parser.add_argument("--out", type=Path, required=True, help="HTML file to write.")
    args = parser.parse_args()

    try:
        cfg = load_review_config(args.config)
        max_rate = cfg["max_rate"] if args.max_rate is None else args.max_rate
        stats = [domain_stats(d, max_rate, cfg["labels"]) for d in args.review_dir]
    except ValueError as e:
        raise SystemExit(str(e))
    cmd = " \\\n    ".join([f"python scripts/review/error_report.py --config {args.config}"]
                           + [f"--review-dir {d}" for d in args.review_dir]
                           + ([f"--max-rate {args.max_rate}"] if args.max_rate is not None else [])
                           + [f"--out {args.out}"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(stats, cfg, max_rate, date.today().isoformat(), cmd))
    for s in stats:
        print(f'{s["label"]}: {s["bad"]} / {s["judged"]} = {pct(s["rate"])} '
              f'(95 % CI {pct(s["low"])} - {pct(s["high"])}) -> {"accepted" if s["accepted"] else "NOT accepted"}'
              f' at <= {pct(max_rate, 0)}{"" if s["complete"] else "  [sample not fully judged]"}')
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
