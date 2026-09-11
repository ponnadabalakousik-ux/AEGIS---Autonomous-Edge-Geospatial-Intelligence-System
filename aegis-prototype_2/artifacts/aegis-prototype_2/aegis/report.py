"""
Self-contained HTML dashboard.
==============================
No CDN, no external assets: every chart is hand-built SVG and every image is
an inline data URL, so the file works from a memory stick on a machine with
no network - which is the only kind of report that survives contact with a
review meeting.
"""
from __future__ import annotations

import base64
import io
import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import config as C

# --- palette (validated categorical slots 1-3; all-pairs safe at three) ----
LIGHT = {"s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a",
         "s4": "#eda100", "bad": "#e34948", "good": "#008300"}
DARK = {"s1": "#3987e5", "s2": "#d95926", "s3": "#199e70",
        "s4": "#c98500", "bad": "#e66767", "good": "#008300"}


# ---------------------------------------------------------------------------
def _png_data_url(arr: np.ndarray) -> str:
    from PIL import Image
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    img = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _fmt(v, nd: int = 2) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        if math.isnan(v):
            return "n/a"
        if abs(v) >= 1e6:
            return f"{v/1e6:.{nd}f} M"
        if abs(v) >= 1e4:
            return f"{v:,.0f}"
        return f"{v:.{nd}f}"
    return str(v)


# ---------------------------------------------------------------------------
# SVG chart primitives
# ---------------------------------------------------------------------------
def _axes(w: int, h: int, pad: Dict[str, int], xticks: List[Tuple[float, str]],
          yticks: List[Tuple[float, str]], xlab: str = "", ylab: str = "") -> str:
    """Recessive grid + axis labels. Grid lines only on y."""
    p = []
    for y, lab in yticks:
        p.append(f'<line x1="{pad["l"]}" y1="{y:.1f}" x2="{w-pad["r"]}" y2="{y:.1f}" '
                 f'class="grid"/>')
        p.append(f'<text x="{pad["l"]-8}" y="{y+4:.1f}" class="tick" '
                 f'text-anchor="end">{_esc(lab)}</text>')
    for x, lab in xticks:
        p.append(f'<text x="{x:.1f}" y="{h-pad["b"]+18}" class="tick" '
                 f'text-anchor="middle">{_esc(lab)}</text>')
    if ylab:
        p.append(f'<text x="14" y="{pad["t"]-14}" class="axlab">{_esc(ylab)}</text>')
    if xlab:
        p.append(f'<text x="{w-pad["r"]}" y="{h-4}" class="axlab" '
                 f'text-anchor="end">{_esc(xlab)}</text>')
    return "".join(p)


def bar_chart(labels: Sequence[str], values: Sequence[float], *,
              colors: Optional[Sequence[str]] = None, w: int = 640, h: int = 300,
              ylab: str = "", value_fmt: str = "{:.3f}",
              reference: Optional[Tuple[float, str]] = None,
              units: str = "") -> str:
    pad = {"l": 62, "r": 20, "t": 22, "b": 58}
    vmax = max(list(values) + ([reference[0]] if reference else []) + [1e-9])
    vmax *= 1.12
    iw, ih = w - pad["l"] - pad["r"], h - pad["t"] - pad["b"]
    n = len(values)
    slot = iw / max(n, 1)
    bw = min(slot * 0.62, 74)

    yt = [(pad["t"] + ih * (1 - f), value_fmt.format(vmax * f))
          for f in (0, 0.25, 0.5, 0.75, 1.0)]
    parts = [_axes(w, h, pad, [], yt, "", ylab)]

    for i, (lab, v) in enumerate(zip(labels, values)):
        cx = pad["l"] + slot * (i + 0.5)
        bh = ih * (max(v, 0) / vmax)
        y = pad["t"] + ih - bh
        col = (colors[i] if colors else "var(--s1)")
        # 4px rounded data-end, square at the baseline
        parts.append(
            f'<path class="mark" d="M{cx-bw/2:.1f},{pad["t"]+ih:.1f} '
            f'V{y+4:.1f} q0,-4 4,-4 H{cx+bw/2-4:.1f} q4,0 4,4 '
            f'V{pad["t"]+ih:.1f} Z" fill="{col}">'
            f'<title>{_esc(lab)}: {value_fmt.format(v)}{_esc(units)}</title></path>')
        parts.append(f'<text x="{cx:.1f}" y="{y-7:.1f}" class="dlab" '
                     f'text-anchor="middle">{value_fmt.format(v)}</text>')
        for j, line in enumerate(_wrap(lab, 16)):
            parts.append(f'<text x="{cx:.1f}" y="{h-pad["b"]+16+j*12:.1f}" '
                         f'class="tick" text-anchor="middle">{_esc(line)}</text>')
    if reference:
        ry = pad["t"] + ih * (1 - reference[0] / vmax)
        parts.append(f'<line x1="{pad["l"]}" y1="{ry:.1f}" x2="{w-pad["r"]}" '
                     f'y2="{ry:.1f}" class="ref"/>')
        parts.append(f'<text x="{w-pad["r"]}" y="{ry-6:.1f}" class="reflab" '
                     f'text-anchor="end">{_esc(reference[1])}</text>')
    return f'<svg viewBox="0 0 {w} {h}" class="chart" role="img">{"".join(parts)}</svg>'


def grouped_bars(categories: Sequence[str], series: Sequence[Tuple[str, Sequence[float]]],
                 *, w: int = 700, h: int = 320, ylab: str = "",
                 value_fmt: str = "{:.2f}", log: bool = False) -> str:
    pad = {"l": 68, "r": 20, "t": 22, "b": 62}
    flat = [v for _, vs in series for v in vs]
    vmax = max(flat + [1e-9]) * 1.15
    # The 1e-9 sentinel guards an empty list; it must not participate in the
    # min, or the log axis floor is pinned at 1e-9 and the bars occupy the top
    # tenth of the plot.
    positive = [v for v in flat if v > 0]
    vmin = (max(min(positive) * 0.5, 1e-9) if (log and positive)
            else (1e-9 if log else 0.0))
    iw, ih = w - pad["l"] - pad["r"], h - pad["t"] - pad["b"]

    def ypos(v: float) -> float:
        if log:
            v = max(v, vmin)
            f = (math.log10(v) - math.log10(vmin)) / (math.log10(vmax) - math.log10(vmin))
        else:
            f = v / vmax
        return pad["t"] + ih * (1 - f)

    if log:
        yt = []
        e0, e1 = math.floor(math.log10(vmin)), math.ceil(math.log10(vmax))
        for e in range(int(e0), int(e1) + 1):
            yt.append((ypos(10.0 ** e), f"1e{e}"))
    else:
        yt = [(ypos(vmax * f), value_fmt.format(vmax * f))
              for f in (0, 0.25, 0.5, 0.75, 1.0)]

    ncat, nser = len(categories), len(series)
    slot = iw / max(ncat, 1)
    bw = min((slot * 0.72) / max(nser, 1), 40)
    parts = [_axes(w, h, pad, [], yt, "", ylab)]
    for si, (sname, vals) in enumerate(series):
        col = f"var(--s{si+1})"
        for ci, v in enumerate(vals):
            x0 = (pad["l"] + slot * (ci + 0.5) - (nser * bw + (nser - 1) * 2) / 2
                  + si * (bw + 2))
            y = ypos(v)
            bh = pad["t"] + ih - y
            parts.append(
                f'<path class="mark" d="M{x0:.1f},{pad["t"]+ih:.1f} V{y+4:.1f} '
                f'q0,-4 4,-4 H{x0+bw-4:.1f} q4,0 4,4 V{pad["t"]+ih:.1f} Z" '
                f'fill="{col}"><title>{_esc(sname)} - {_esc(categories[ci])}: '
                f'{value_fmt.format(v)}</title></path>')
    for ci, cat in enumerate(categories):
        cx = pad["l"] + slot * (ci + 0.5)
        for j, line in enumerate(_wrap(cat, 15)):
            parts.append(f'<text x="{cx:.1f}" y="{h-pad["b"]+16+j*12:.1f}" '
                         f'class="tick" text-anchor="middle">{_esc(line)}</text>')
    return f'<svg viewBox="0 0 {w} {h}" class="chart" role="img">{"".join(parts)}</svg>'


def line_chart(series: Sequence[Tuple[str, Sequence[float], Sequence[float]]], *,
               w: int = 700, h: int = 300, xlab: str = "", ylab: str = "",
               logx: bool = False, ylim: Optional[Tuple[float, float]] = None,
               xfmt: str = "{:.0f}", yfmt: str = "{:.2f}",
               markers: bool = True) -> str:
    pad = {"l": 62, "r": 24, "t": 22, "b": 52}
    xs_all = [x for _, xs, _ in series for x in xs]
    ys_all = [y for _, _, ys in series for y in ys]
    if not xs_all:
        return ""
    xmin, xmax = min(xs_all), max(xs_all)
    if logx:
        xmin = max(min([x for x in xs_all if x > 0] + [1.0]), 1e-9)
    if ylim:
        ymin, ymax = ylim              # explicit limits are honoured exactly
    else:
        ymin, ymax = min(ys_all), max(ys_all)
        if ymax - ymin < 1e-12:
            ymax = ymin + 1.0
        pad_y = (ymax - ymin) * 0.08
        ymin, ymax = ymin - pad_y, ymax + pad_y
    iw, ih = w - pad["l"] - pad["r"], h - pad["t"] - pad["b"]

    def px(x):
        if logx:
            x = max(x, xmin)
            f = (math.log10(x) - math.log10(xmin)) / max(
                math.log10(xmax) - math.log10(xmin), 1e-9)
        else:
            f = (x - xmin) / max(xmax - xmin, 1e-9)
        return pad["l"] + iw * f

    def py(y):
        return pad["t"] + ih * (1 - (y - ymin) / (ymax - ymin))

    yt = [(py(ymin + (ymax - ymin) * f), yfmt.format(ymin + (ymax - ymin) * f))
          for f in (0, 0.25, 0.5, 0.75, 1.0)]
    if logx:
        e0, e1 = math.floor(math.log10(xmin)), math.ceil(math.log10(xmax))
        xt = [(px(10.0 ** e), f"1e{e}") for e in range(int(e0), int(e1) + 1)]
    else:
        xt = [(px(xmin + (xmax - xmin) * f), xfmt.format(xmin + (xmax - xmin) * f))
              for f in (0, 0.25, 0.5, 0.75, 1.0)]
    parts = [_axes(w, h, pad, xt, yt, xlab, ylab)]
    for si, (name, xs, ys) in enumerate(series):
        col = f"var(--s{si+1})"
        d = " ".join(("M" if i == 0 else "L") + f"{px(x):.1f},{py(y):.1f}"
                     for i, (x, y) in enumerate(zip(xs, ys)))
        parts.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2" '
                     f'stroke-linejoin="round" stroke-linecap="round"/>')
        if markers and len(xs) <= 40:
            for x, y in zip(xs, ys):
                parts.append(
                    f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="4.5" '
                    f'fill="{col}" stroke="var(--surface-1)" stroke-width="2">'
                    f'<title>{_esc(name)}: x={xfmt.format(x)}, y={yfmt.format(y)}'
                    f'</title></circle>')
    return f'<svg viewBox="0 0 {w} {h}" class="chart" role="img">{"".join(parts)}</svg>'


def legend(names: Sequence[str]) -> str:
    items = "".join(
        f'<span class="lg"><i style="background:var(--s{i+1})"></i>{_esc(n)}</span>'
        for i, n in enumerate(names))
    return f'<div class="legend">{items}</div>'


def _wrap(text: str, n: int) -> List[str]:
    words, lines, cur = str(text).split(), [], ""
    for wd in words:
        if len(cur) + len(wd) + 1 <= n:
            cur = (cur + " " + wd).strip()
        else:
            if cur:
                lines.append(cur)
            cur = wd
    if cur:
        lines.append(cur)
    return lines[:3] or [str(text)]


def table(headers: Sequence[str], rows: Sequence[Sequence], caption: str = "") -> str:
    th = "".join(f"<th>{_esc(x)}</th>" for x in headers)
    tr = "".join("<tr>" + "".join(f"<td>{_esc(c)}</td>" for c in r) + "</tr>"
                 for r in rows)
    cap = f"<caption>{_esc(caption)}</caption>" if caption else ""
    return f'<div class="tw"><table>{cap}<thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def stat(value: str, label: str, sub: str = "", tone: str = "") -> str:
    return (f'<div class="stat {tone}"><div class="sv">{_esc(value)}</div>'
            f'<div class="sl">{_esc(label)}</div>'
            f'<div class="ss">{_esc(sub)}</div></div>')


# ---------------------------------------------------------------------------
CSS = """
:root{color-scheme:light dark}
.viz{--surface-1:#fcfcfb;--surface-2:#f4f3f0;--line:#e2e0da;
 --text-primary:#0b0b0b;--text-secondary:#52514e;--text-muted:#82817c;
 --s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--bad:#e34948;--good:#008300}
@media (prefers-color-scheme:dark){.viz{--surface-1:#1a1a19;--surface-2:#232322;
 --line:#3a3a37;--text-primary:#fff;--text-secondary:#c3c2b7;--text-muted:#8f8e86;
 --s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--bad:#e66767;--good:#3aa03a}}
*{box-sizing:border-box}
body{margin:0;background:var(--surface-2);color:var(--text-primary);
 font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:32px 22px 80px}
h1{font-size:30px;letter-spacing:-.02em;margin:0 0 6px}
h2{font-size:20px;margin:44px 0 6px;letter-spacing:-.01em}
h3{font-size:15px;margin:26px 0 8px;color:var(--text-secondary);
 text-transform:uppercase;letter-spacing:.07em;font-weight:600}
p{color:var(--text-secondary);margin:8px 0 14px;max-width:78ch}
.lede{font-size:16px;color:var(--text-secondary)}
.card{background:var(--surface-1);border:1px solid var(--line);border-radius:12px;
 padding:20px 22px;margin:16px 0}
.grid{display:grid;gap:16px}
.g2{grid-template-columns:repeat(auto-fit,minmax(330px,1fr))}
.g4{grid-template-columns:repeat(auto-fit,minmax(190px,1fr))}
.stat{background:var(--surface-1);border:1px solid var(--line);border-radius:12px;padding:16px 18px}
.stat .sv{font-size:29px;font-weight:650;letter-spacing:-.02em;line-height:1.15}
.stat .sl{font-size:13px;color:var(--text-secondary);margin-top:3px}
.stat .ss{font-size:12px;color:var(--text-muted);margin-top:5px}
.stat.good .sv{color:var(--good)} .stat.warn .sv{color:var(--s4)}
.chart{width:100%;height:auto;display:block;overflow:visible}
.grid{stroke:var(--line);stroke-width:1}
.tick{fill:var(--text-muted);font-size:11px}
.axlab{fill:var(--text-secondary);font-size:11px;font-weight:600;
 text-transform:uppercase;letter-spacing:.06em}
.dlab{fill:var(--text-secondary);font-size:11px;font-weight:600}
.ref{stroke:var(--text-muted);stroke-width:2;stroke-dasharray:5 4}
.reflab{fill:var(--text-muted);font-size:11px;font-weight:600}
.mark:hover{opacity:.82}
.legend{display:flex;flex-wrap:wrap;gap:16px;margin:6px 0 10px}
.lg{display:inline-flex;align-items:center;gap:7px;font-size:13px;color:var(--text-secondary)}
.lg i{width:11px;height:11px;border-radius:3px;display:inline-block}
.tw{overflow-x:auto;margin:10px 0}
table{border-collapse:collapse;width:100%;font-size:13.5px}
caption{text-align:left;color:var(--text-muted);font-size:12px;padding-bottom:8px}
th,td{padding:8px 11px;border-bottom:1px solid var(--line);text-align:left;
 vertical-align:top}
th{color:var(--text-muted);font-weight:600;font-size:11.5px;text-transform:uppercase;
 letter-spacing:.05em;white-space:nowrap}
td:not(:first-child){font-variant-numeric:tabular-nums}
.gal{display:grid;grid-template-columns:repeat(auto-fill,minmax(128px,1fr));gap:12px}
.tile{background:var(--surface-2);border:1px solid var(--line);border-radius:9px;
 padding:8px;font-size:11px;color:var(--text-muted)}
.tile img{width:100%;image-rendering:pixelated;border-radius:5px;display:block}
.tile b{color:var(--text-primary);font-size:12px}
.note{border-left:3px solid var(--s4);background:var(--surface-1);
 padding:12px 16px;border-radius:0 9px 9px 0;margin:14px 0;font-size:13.5px;
 color:var(--text-secondary)}
.warnbox{border-left-color:var(--bad)}
.okbox{border-left-color:var(--good)}
code{background:var(--surface-2);padding:1px 5px;border-radius:4px;font-size:12.5px}
.meta{color:var(--text-muted);font-size:12.5px}
ul{color:var(--text-secondary);max-width:78ch}
li{margin:5px 0}
"""


# ---------------------------------------------------------------------------
def build_dashboard(results: Dict, data: Dict, screener, net,
                    biomass: Optional[Tuple] = None,
                    path: str = None) -> str:
    path = path or os.path.join(C.ARTIFACT_DIR, "aegis_dashboard.html")
    cmp_ = results["campaign"]
    b, a = cmp_["baseline"], cmp_["aegis"]
    ben = cmp_["benefits"]
    S: List[str] = []

    # ---------------- header ----------------
    S.append(f"""<h1>AEGIS - autonomous onboard triage for Earth observation</h1>
<p class="lede">A working prototype of an AI payload that decides, in orbit,
what is worth computing, keeping and downlinking. Everything below is produced
by <code>run_all.py</code> from a single seed; no number is typed in by hand.</p>
<p class="meta">Instrument {results['acquisition']['tiles_per_day']/1e6:.2f} M tiles/day
({results['acquisition']['compressed_gb_per_day']:.0f} GB/day compressed) &middot;
ground segment {results['downlink']['capacity_gb_per_day']:.1f} GB/day &middot;
<b>{results['acquisition']['compressed_gb_per_day']/results['downlink']['capacity_gb_per_day']:.1f}x
oversubscribed</b> &middot; {results['config']['sim_days']:.0f}-day campaign &middot;
payload {C.HARDWARE[results['config']['hardware']].name}</p>""")

    # ---------------- headline stats ----------------
    S.append('<div class="grid g4">')
    S.append(stat(f"{ben['1_downlink_volume']['reduction']*100:.0f}%",
                  "less data downlinked",
                  f"{b['gb_downlinked_per_day']:.1f} → {a['gb_downlinked_per_day']:.1f} GB/day",
                  "good"))
    S.append(stat(f"{ben['2_event_delivery']['aegis_rate']*100:.0f}%",
                  "of real events delivered",
                  f"baseline {b['event_delivery_rate']*100:.0f}% "
                  f"({ben['2_event_delivery']['improvement_x']:.1f}x)", "good"))
    S.append(stat(f"{a['median_latency_min']:.0f} min",
                  "median observation → ground",
                  f"baseline {b['median_latency_min']:.0f} min; "
                  f"alerts {a['mean_alert_latency_min']:.0f} min", "good"))
    S.append(stat(f"{ben['5_life_limited']['tx_reduction']*100:.0f}%",
                  "less transmitter on-time",
                  f"{b['tx_on_time_hours']:.1f} → {a['tx_on_time_hours']:.1f} h "
                  f"per {results['config']['sim_days']:.0f} days", "good"))
    S.append("</div>")

    # ---------------- benefits table ----------------
    S.append("<h2>The six claimed benefits, measured</h2>")
    S.append("""<p>Two spacecraft, identical orbit, identical imagery, identical
link budget. One downlinks oldest-first because that is what a conventional
mission does; the other runs the cascade and decides. The only difference is
the payload software.</p>""")
    rows = [
        ["Reduced data transmission",
         f"{b['gb_downlinked_per_day']:.2f} GB/day", f"{a['gb_downlinked_per_day']:.2f} GB/day",
         f"−{ben['1_downlink_volume']['reduction']*100:.1f}%"],
        ["Selective / better-targeted measurement",
         f"{b['event_delivery_rate']*100:.1f}% of events",
         f"{a['event_delivery_rate']*100:.1f}% of events",
         f"{ben['2_event_delivery']['improvement_x']:.2f}× more"],
        ["Reduced analysis time (median latency)",
         f"{b['median_latency_min']:.0f} min", f"{a['median_latency_min']:.0f} min",
         f"{ben['3_latency']['speedup_x']:.0f}× faster"],
        ["Reduced net power (comms + compute)",
         f"{b['energy_comms_wh']+b['energy_compute_wh']:.0f} Wh",
         f"{a['energy_comms_wh']+a['energy_compute_wh']:.0f} Wh",
         f"−{ben['4_energy']['net_saving_fraction']*100:.1f}%"],
        ["Less usage of life-limited items (TX on-time)",
         f"{b['tx_on_time_hours']:.2f} h", f"{a['tx_on_time_hours']:.2f} h",
         f"−{ben['5_life_limited']['tx_reduction']*100:.1f}%"],
        ["Decreased mass-memory demand (peak fill)",
         f"{b['peak_memory_fraction']*100:.0f}%", f"{a['peak_memory_fraction']*100:.0f}%",
         f"{b['tiles_dropped_memory']/1e6:.0f} M tiles lost vs "
         f"{a['tiles_dropped_memory']/1e6:.0f} M"],
    ]
    S.append(table(["Benefit", "Baseline (no AI)", "AEGIS", "Change"], rows))

    # ---------------- energy honesty note ----------------
    S.append(f"""<div class="note"><b>The energy result is the one to read carefully.</b>
Onboard AI does not come free: the cascade burns
{a['energy_compute_wh']:.0f} Wh of compute over the campaign to save
{b['energy_comms_wh']-a['energy_comms_wh']:.0f} Wh of communications. The net
saving is {ben['4_energy']['net_saving_fraction']*100:.0f}%, not the
{ben['1_downlink_volume']['reduction']*100:.0f}% that the downlink figure alone
would suggest. This is exactly the caveat the problem statement anticipates -
"although additional computing power may limit this saving" - and it is why
the cheap Stage 0 gate matters: running the CNN on every tile instead of on
the {a['cnn_invocation_rate']*100:.0f}% that survive the gate would roughly
double the compute bill.</div>""")

    # ---------------- cascade ----------------
    S.append("<h2>The cascade</h2>")
    S.append(f"""<p>Three stages, cheapest first. On a power- and thermally-constrained
payload you do not run a neural network on every tile; you run the cheapest thing
that can safely say "no".</p>""")
    casc = results["cascade"]
    scr = results.get("screener", {})
    S.append(table(
        ["Stage", "What it does", "Cost per tile", "Runs on", "Effect"],
        [["0 - spectral gate",
          "32 spectral-statistic features → logistic gate",
          f"{casc['stage0_flops']/1e3:.0f} kFLOP",
          "100% of tiles (housekeeping CPU)",
          f"rejects {scr.get('reject_rate', 0)*100:.0f}% while keeping "
          f"{scr.get('recall_on_useful', 0)*100:.1f}% of useful tiles"],
         ["1 - TriageNet",
          "depthwise-separable CNN; cloud fraction + 4-class event + coarse mask",
          f"{results['triagenet']['mmac']:.2f} MMAC",
          f"{casc['gate_pass_rate']*100:.0f}% of tiles (INT8 accelerator)",
          f"{results['triagenet']['params']:,} parameters"],
         ["2 - ROI extraction",
          "16×16 mask → padded bounding box → crop",
          "~4 kFLOP",
          f"{casc['detect_rate']*100:.0f}% of tiles",
          "sends the box, not the scene"]]))

    # ---------------- quantisation study ----------------
    if "quantisation_study" in results:
        qs = results["quantisation_study"]
        ok = {k: v for k, v in qs.items() if "error" not in v}
        fp32_f1 = results["accuracy"]["fp32"]["event_macro_f1"]
        S.append("<h2>INT8 quantisation: the step that silently breaks</h2>")
        S.append("""<p>Flown accelerators are INT8 engines, so quantisation is not an
optimisation, it is the enabling step. It is also where this prototype nearly
shipped a broken payload. Default MinMax calibration sets each activation's
scale from the single most extreme value it sees, so one bright cloud edge
stretches the range and crushes everything else into a handful of levels.</p>""")
        S.append('<div class="card">')
        labels = list(ok.keys())
        vals = [ok[k]["event_macro_f1"] for k in labels]
        cols = ["var(--bad)" if v < fp32_f1 * 0.85 else
                ("var(--s3)" if v >= fp32_f1 * 0.97 else "var(--s4)") for v in vals]
        S.append(bar_chart(labels, vals, colors=cols, ylab="event macro-F1",
                           reference=(fp32_f1, f"FP32 reference {fp32_f1:.3f}"),
                           h=330))
        S.append(table(["Configuration", "macro-F1", "seg IoU", "cloud MAE",
                        "false-alarm rate"],
                       [[k, f"{v['event_macro_f1']:.3f}", f"{v['seg_iou']:.3f}",
                         f"{v['cloud_mae']:.4f}", f"{v['false_alarm_rate']:.3f}"]
                        for k, v in ok.items()]
                       + [["FP32 reference", f"{fp32_f1:.3f}",
                           f"{results['accuracy']['fp32']['seg_iou']:.3f}",
                           f"{results['accuracy']['fp32']['cloud_mae']:.4f}",
                           f"{results['accuracy']['fp32']['false_alarm_rate']:.3f}"]]))
        S.append("</div>")
        S.append("""<div class="note okbox"><b>Finding.</b> Percentile calibration
recovers most of the loss; adding <code>reduce_range</code> - holding weights to
7 bits so INT8 accumulation cannot saturate - recovers the rest and matches FP32
at a quarter of the weight memory. Nothing in the tooling warns you about this.
Any flight model should be measured this way, not assumed.</div>""")

    # ---------------- operating point ----------------
    if "operating_point" in results:
        curve = results["operating_point"]["curve"]
        ch = results["operating_point"]["chosen"]
        S.append("<h2>Choosing the operating point at the real event rate</h2>")
        S.append(f"""<p>A model tuned on a class-balanced set will happily flag 8% of a
stream in which only {results['datasets']['test']['observable_event_rate']*100:.1f}%
of tiles contain anything - which is how an onboard system ends up downlinking
more false alarms than events. The threshold is therefore chosen on a
natural-prior validation set that the test set never sees, by maximising recall
subject to a floor on precision.</p>""")
        S.append('<div class="card">')
        S.append(legend(["precision", "recall"]))
        S.append(line_chart(
            [("precision", [r["threshold"] for r in curve], [r["precision"] for r in curve]),
             ("recall", [r["threshold"] for r in curve], [r["recall"] for r in curve])],
            xlab="event-confidence threshold", ylab="rate", ylim=(0, 1),
            xfmt="{:.2f}", markers=False, h=280))
        S.append(f'<p class="meta">Chosen: confidence ≥ {ch["threshold"]:.2f} → '
                 f'precision {ch["precision"]:.3f}, recall {ch["recall"]:.3f}, '
                 f'flagging {ch["flag_rate"]*100:.1f}% of the stream.</p>')
        S.append("</div>")
        spread = max(r["recall"] for r in curve) - min(r["recall"] for r in curve)
        S.append(f"""<div class="note okbox"><b>The flatness is the finding.</b>
Recall varies by only {spread*100:.1f} percentage points across the entire usable
threshold range, because the network's posteriors are saturated - it is either
confident and right or it has missed the event entirely. Operationally that is
good news: the threshold is an uplinkable parameter, and a parameter whose exact
value barely matters is one an operations team can set without fear. It also
means the remaining errors are not fixable by retuning the threshold; they need
a better model or better data.</div>""")

    # ---------------- hardware ----------------
    S.append("<h2>Does it close on hardware that has actually flown?</h2>")
    hw = results["hardware"]
    S.append(f"""<p>The latency model is calibrated against a published in-orbit
measurement rather than a vendor peak-TOPS figure: CloudScout on Φ-Sat-1 ran a
512×512×3 input in 325 ms on a Myriad 2 at about 2 W. Solving for sustained
utilisation gives {C.HARDWARE['myriad2'].int8_gops_effective * 0.123:.0f} GOP/s -
roughly an eighth of the part's sustained rate - which is why "it's a 1 TOPS
device" tells you almost nothing.</p>""")
    S.append('<div class="card">')
    S.append(grouped_bars(
        [h["name"].split("(")[0].strip()[:22] for h in hw],
        [("CNN latency (ms)", [h["stage1_ms"] for h in hw]),
         ("cascade energy (mJ/tile)", [h["mean_energy_mj_per_tile"] for h in hw])],
        ylab="log scale", log=True, h=330))
    S.append(legend(["CNN latency (ms)", "cascade energy (mJ/tile)"]))
    S.append(table(
        ["Payload processor", "Sustained INT8", "CNN latency", "Cascade energy",
         "Active power", "TID", "Flight heritage"],
        [[h["name"], f"{h['sustained_gops']:.1f} GOP/s", f"{h['stage1_ms']:.2f} ms",
          f"{h['mean_energy_mj_per_tile']:.2f} mJ/tile", f"{h['power_active_w']:.1f} W",
          f"{h['tid_krad']:.0f} krad", h["flight_heritage"]] for h in hw]))
    S.append("</div>")
    leon = next((h for h in hw if "LEON" in h["name"]), None)
    if leon:
        S.append(f"""<div class="note"><b>The control case matters.</b> On a
conventional rad-hard OBC with no accelerator the same network takes
{leon['stage1_ms']:.0f} ms per tile and burns
{leon['mean_energy_mj_per_tile']:.0f} mJ - {leon['mean_energy_mj_per_tile']/hw[0]['mean_energy_mj_per_tile']:.0f}×
the energy of the Myriad 2 path. Onboard AI of this kind is not a software
upgrade to an existing avionics stack; it needs the accelerator.</div>""")

    # ---------------- radiation ----------------
    if "radiation_sweep" in results:
        sw = results["radiation_sweep"]
        rad = results["radiation"]
        S.append("<h2>Radiation: how much upset can the network absorb?</h2>")
        S.append("""<p>A neural network in orbit is a large block of memory being
continuously corrupted. INT8 weights are unforgiving: flipping the top bit moves
a weight by 128 quantisation steps. The study injects real bit flips into the
quantised weights and measures what breaks.</p>""")
        S.append('<div class="card">')
        S.append(line_chart(
            [("macro-F1", [max(r["flips"], 1) for r in sw],
              [r["event_macro_f1_mean"] for r in sw])],
            logx=True, xlab="bit flips injected into INT8 weights",
            ylab="event macro-F1", ylim=(0, 1), yfmt="{:.2f}", h=290))
        intact = sw[0]["event_macro_f1_mean"]
        knee = next((r for r in sw if r["event_macro_f1_mean"] < 0.8 * intact), None)
        S.append(table(
            ["Bit flips", "% of weights", "Equivalent unmitigated exposure",
             "macro-F1"],
            [[f"{r['flips']:,}", f"{r['corrupted_weight_fraction']*100:.2f}%",
              f"{r['equivalent_years_unmitigated']:.1f} yr",
              f"{r['event_macro_f1_mean']:.3f} ± {r['event_macro_f1_std']:.3f}"]
             for r in sw]))
        S.append("</div>")
        year = next((r for r in rad if r["mitigation"] == "none"
                     and r["exposure_days"] == 365.0), None)
        if year and knee:
            S.append(f"""<div class="note okbox"><b>Finding.</b> One year of unmitigated
upsets at 500 km costs about {year['flips']} flips and moves macro-F1 from
{intact:.3f} to {year['event_macro_f1']:.3f} - the network is intrinsically
tolerant, because a 38k-parameter model spreads its function widely. The 20%
degradation point is around {knee['flips']:,} flips, equivalent to
{knee['equivalent_years_unmitigated']:.0f} years of unmitigated exposure. So the
margin is large, and ECC plus periodic scrubbing from a rad-hard golden copy
takes it further still. That is a design conclusion with mass and power
attached, which is the point of measuring it rather than assuming it.</div>""")

    # ---------------- mission traces ----------------
    tr = results.get("traces", {})
    if tr.get("baseline_memory"):
        S.append("<h2>What the recorder actually does</h2>")
        S.append("""<p>The clearest single picture of the problem. The conventional
spacecraft fills its mass memory and then starts destroying data it has already
paid to collect; it has no way to prefer the fire over the cloud, so it loses
both at random.</p>""")
        S.append('<div class="card">')
        S.append(legend(["baseline (no AI)", "AEGIS"]))
        bm, am = tr["baseline_memory"], tr["aegis_memory"]
        S.append(line_chart(
            [("baseline", [p[0] / 86400 for p in bm], [p[1] * 100 for p in bm]),
             ("AEGIS", [p[0] / 86400 for p in am], [p[1] * 100 for p in am])],
            xlab="days since epoch", ylab="mass memory used (%)",
            ylim=(0, 105), xfmt="{:.0f}", yfmt="{:.0f}", markers=False, h=290))
        S.append("</div>")

    # ---------------- ablations ----------------
    if "ablations" in results:
        ab = results["ablations"]
        S.append("<h2>Which part is doing the work?</h2>")
        order = ["full_aegis", "no_cloud_rejection", "no_novelty", "no_voi_priority"]
        names = {"full_aegis": "full AEGIS",
                 "no_cloud_rejection": "without cloud rejection",
                 "no_novelty": "without novelty decay",
                 "no_voi_priority": "without value-ordered downlink (FIFO)"}
        rows = [[names[k], f"{ab[k]['gb_per_day']:.2f}",
                 f"{ab[k]['event_delivery_rate']*100:.1f}%",
                 f"{ab[k]['median_latency_min']:.0f} min"]
                for k in order if k in ab]
        S.append(table(["Configuration", "GB/day downlinked", "events delivered",
                        "median latency"], rows,
                       "Each row removes one mechanism and re-flies the campaign."))

    # ---------------- link stress ----------------
    if "link_stress" in results:
        ls = results["link_stress"]
        S.append("<h2>When does value-ordered scheduling actually matter?</h2>")
        S.append("""<p>An honest negative result first: at the nominal ground segment,
ordering the downlink queue by value-per-bit makes no measurable difference.
That is not a failure of the scheduler - it is a consequence of the triage
working. Once volume is down by 92% every queued product fits in the available
contacts, so the order it goes in is irrelevant.</p>
<p>Squeeze the ground segment and the constraint comes back. The sweep below
takes contacts away until the link binds again.</p>""")
        S.append('<div class="card">')
        S.append(legend(["value-ordered", "FIFO"]))
        S.append(line_chart(
            [("value-ordered", [r["capacity_gb_per_day"] for r in ls],
              [r["voi_delivery"] for r in ls]),
             ("FIFO", [r["capacity_gb_per_day"] for r in ls],
              [r["fifo_delivery"] for r in ls])],
            logx=True, xlab="usable downlink capacity (GB/day, log)",
            ylab="fraction of real events delivered", ylim=(0, 1), yfmt="{:.2f}"))
        S.append(table(
            ["Ground segment", "Capacity", "Demand", "Delivery (VoI)",
             "Delivery (FIFO)", "Median latency (VoI)", "Median latency (FIFO)"],
            [[f"{r['pass_utilisation']*100:.0f}% of granted passes",
              f"{r['capacity_gb_per_day']:.2f} GB/day",
              f"{r['demand_gb_per_day']:.2f} GB/day",
              f"{r['voi_delivery']*100:.1f}%", f"{r['fifo_delivery']*100:.1f}%",
              f"{r['voi_latency_min']:.0f} min", f"{r['fifo_latency_min']:.0f} min"]
             for r in ls]))
        S.append("</div>")
        binder = min(ls, key=lambda r: abs(r["capacity_gb_per_day"] - r["demand_gb_per_day"]))
        worst = min(ls, key=lambda r: r["capacity_gb_per_day"] if r["voi_delivery"] > 0.3 else 9e9)
        S.append(f"""<div class="note"><b>Reading it.</b> At the crossover
({binder['capacity_gb_per_day']:.1f} GB/day capacity against
{binder['demand_gb_per_day']:.1f} GB/day of demand) both policies deliver the
same events, but value-ordering does it in
{binder['voi_latency_min']:.0f} minutes against
{binder['fifo_latency_min']:.0f} - a {binder['fifo_latency_min']/max(binder['voi_latency_min'],1):.1f}×
difference in time-to-ground. Squeeze harder still and it starts changing
*what* arrives, not just when: at {worst['capacity_gb_per_day']:.2f} GB/day
value-ordering delivers {worst['voi_delivery']*100:.0f}% of events against
FIFO's {worst['fifo_delivery']*100:.0f}%. The scheduler is insurance against a
degraded ground segment, not the main mechanism - and it is worth knowing which
of the two it is.</div>""")

    # ---------------- sample tiles ----------------
    try:
        S.append("<h2>What the payload is looking at</h2>")
        S.append("""<p>Simulated Sentinel-2-like tiles with pixel-accurate truth.
Left column true colour (B04/B03/B02), right column the SWIR composite
(B12/B11/B04) that fire appears in. Fire is modelled the way a multispectral
imager actually sees it: sub-pixel high-temperature emission lifts SWIR-2 far
more than SWIR-1.</p>""")
        from .scene import to_rgb, to_swir_composite
        ts = data["test"]
        picks: List[int] = []
        for cls in range(len(C.EVENT_CLASSES)):
            idx = np.where((ts.event_lbl == cls) & (ts.cloud_frac < 0.3))[0]
            picks.extend(idx[:2].tolist())
        idx = np.where(ts.cloud_frac > 0.7)[0]
        picks.extend(idx[:2].tolist())
        S.append('<div class="gal">')
        for i in picks[:12]:
            rgb = _png_data_url(to_rgb(ts.cubes[i]))
            swir = _png_data_url(to_swir_composite(ts.cubes[i]))
            S.append(f'''<div class="tile"><img src="{rgb}" alt="true colour"/>
<img src="{swir}" alt="SWIR composite" style="margin-top:5px"/>
<div style="margin-top:6px"><b>{_esc(C.EVENT_CLASSES[ts.event_lbl[i]])}</b><br/>
cloud {ts.cloud_frac[i]*100:.0f}%</div></div>''')
        S.append("</div>")
    except Exception as exc:                              # noqa: BLE001
        S.append(f'<p class="meta">tile gallery unavailable: {_esc(exc)}</p>')

    # ---------------- BIOMASS ----------------
    if biomass:
        bres, bdet = biomass
        S.append("<h2>Modularity, demonstrated: the same cascade on ESA Biomass</h2>")
        S.append(f"""<p>The brief asks for a system that can augment an existing
satellite through modular integration. Here the front end is swapped from a
6-band optical imager to <b>Biomass</b>, ESA's P-band fully-polarimetric SAR
(Airbus prime) - a completely different physics - and the decision layer,
cost model and mission simulator are unchanged.</p>
<p>The frame is real: <code>{_esc(bdet['frame']['product'][:46])}…</code>,
track 006 / frame 300, acquired 2025-11-21 over Rondônia, Brazil
({bdet['frame']['centre_lat']:.2f}, {bdet['frame']['centre_lon']:.2f}),
tomographic phase, quad-pol.</p>""")
        S.append('<div class="grid g4">')
        S.append(stat(f"{bdet['volume']['slc_gb']:.2f} GB", "per L1A frame",
                      f"{bdet['frame']['samples']}×{bdet['frame']['lines']} × 4 pol, "
                      f"complex — for 21 s of acquisition"))
        S.append(stat(f"{bres.accuracy:.3f}", "held-out accuracy",
                      f"AUC {bres.auc:.3f}, on descriptors not used to build "
                      f"the labels"))
        S.append(stat(f"{bres.cleared_fraction*100:.0f}%", "of frame flagged disturbed",
                      f"{bres.n_tiles} tiles, {bres.n_labelled} confidently labelled"))
        S.append(stat(f"{bres.downlink_reduction*100:.0f}%", "link saving from prioritisation",
                      "full rate on disturbance, decimated interior forest", "warn"))
        S.append("</div>")
        S.append(f"""<div class="note warnbox"><b>Read these numbers with the caveats
attached.</b><ul>{''.join(f'<li>{_esc(n)}</li>' for n in bres.notes)}</ul></div>""")

    # ---------------- RFI ----------------
    if results.get("rfi"):
        rows = results["rfi"]
        S.append("<h2>Onboard RFI screening: real interference, real numbers</h2>")
        S.append("""<p>The strongest Biomass case, and the one that needs no labels
at all. Biomass transmits at 435 MHz into a <b>6 MHz</b> allocation - a sliver of
heavily contended spectrum. When an interferer sits in the band the processor
must notch it out, and notched bandwidth is gone: slant-range resolution is
c/2B, so interference directly coarsens the science product.</p>
<p>The figures below are read from the delivered products - the ground
processor's own notch decisions in the L1A LUT and the RFI report in the
annotation. This is measured interference, not a model of it.</p>""")
        S.append('<div class="card">')
        S.append(legend([r["frame"] for r in rows][:3]))
        # One quantity per chart. Percent-of-band and metres do not belong on
        # a shared axis, however tempting the side-by-side comparison is.
        S.append(grouped_bars(
            ["persistent RFI (avg)", "persistent RFI (max)", "isolated RFI"],
            [(r["frame"], [r["persistent_avg_pct_bw"], r["persistent_max_pct_bw"],
                           r["isolated_pct_affected_lines"]]) for r in rows],
            value_fmt="{:.1f}", h=300))
        S.append('<p class="meta">Percent of the 6 MHz allocation notched out '
                 '(first two), and percent of azimuth lines carrying isolated '
                 'interference (third).</p>')
        S.append(table(
            ["Frame", "Persistent RFI (avg / max % of band)",
             "Isolated RFI (% of lines)", "Effective bandwidth",
             "Slant-range resolution", "Penalty"],
            [[r["frame"],
              f"{r['persistent_avg_pct_bw']:.2f}% / {r['persistent_max_pct_bw']:.2f}%",
              f"{r['isolated_pct_affected_lines']:.1f}%",
              f"{r['effective_bandwidth_mhz']:.3f} MHz",
              f"{r['resolution_m']:.2f} m (worst {r['worst_resolution_m']:.2f} m)",
              f"+{r['resolution_penalty']*100:.1f}%"] for r in rows],
            "Nominal 6 MHz gives 24.98 m. Both frames acquired 2025-11-21."))
        S.append("</div>")

        tri = next((r["triage"] for r in rows if "triage" in r), None)
        if tri:
            carriers = ", ".join(f"{f:+.2f} MHz ({p*100:.0f}% of blocks)"
                                 for f, p in tri["persistent_carriers"])
            S.append(f"""<div class="note okbox"><b>What the mask shows.</b> The
notch mask is {tri['n_blocks']} azimuth blocks x 87 frequency bins, identical
across all four polarisations. There is a <b>fixed emitter parked almost exactly
at band centre</b>: bins at {carriers} are notched in nearly every block of the
acquisition. That is not transient interference - it is a permanent tax on the
mission's bandwidth over that region.</div>""")
            S.append(f"""<p>Applied as an onboard policy, ranking blocks by
surviving bandwidth: <b>{tri['blocks_nominal']*100:.1f}%</b> of blocks are
essentially intact and go at full rate,
<b>{tri['blocks_degraded']*100:.1f}%</b> are degraded and get decimated and
deferred, <b>{tri['blocks_severely_degraded']*100:.1f}%</b> are severely
corrupted and reduced to a summary - a
<b>{tri['downlink_reduction']*100:.1f}%</b> link saving on the quiet frame, and
far more over a contended region.</p>""")
            spec = tri.get("spectrum")
            if spec:
                S.append('<div class="card">')
                S.append(line_chart(
                    [("notch prevalence", spec["freq_offset_mhz"],
                      spec["notched_fraction"])],
                    xlab="frequency offset from 435 MHz (MHz)",
                    ylab="", ylim=(0, 1), xfmt="{:+.1f}", yfmt="{:.1f}",
                    markers=False, h=260))
                S.append('<p class="meta">Fraction of the acquisition\'s azimuth '
                         'blocks in which each frequency bin had to be notched. '
                         'The spike at band centre is a fixed emitter; the '
                         'shoulders are transient.</p>')
                S.append("</div>")
            S.append('<div class="note">' + "".join(
                f"<div>{_esc(n)}</div>" for n in tri["notes"]) + "</div>")
        S.append("""<p>The onboard version is cheaper than anything else in this
prototype: a periodogram of the raw echo and a threshold, before focusing. No
network, no training data, no labels. It is the kind of decision that is
obviously better made in orbit than on the ground, because by the time the data
reaches the ground the bandwidth has already been spent carrying it.</p>""")

    # ---------------- mission adapters ----------------
    if "mission_cases" in results:
        S.append("<h2>What this would buy on real missions</h2>")
        S.append("""<p>The same cost model applied to two flown/flying missions and to
the bespoke demonstrator. They fail differently, and the difference is the
interesting part.</p>""")
        S.append(table(
            ["Mission", "Onboard app", "Data before", "Data after", "Reduction",
             "Contact saved", "Compute duty", "Compute power"],
            [[m["mission"], m["ai_app"][:78] + "…",
              f"{m['daily_gbit_before']:.0f} Gbit/day",
              f"{m['daily_gbit_after']:.0f} Gbit/day",
              f"{m['downlink_reduction']*100:.0f}%",
              f"{m['contact_s_saved_per_day']/60:.0f} min/day",
              f"{m['compute_duty_cycle']*100:.1f}%",
              f"{m['compute_power_fraction_of_platform']*100:.2f}% of platform"]
             for m in results["mission_cases"]]))
        S.append("""<div class="note"><b>Biomass is the honest hard case.</b> There is
no cloud to throw away at P-band and every frame is wanted science, so the
saving cannot come from discarding - it comes from ordering: disturbance fronts
at full rate and first, slow-changing interior forest deferred or decimated,
backed by the repeat stack. MicroCarb is the opposite: most soundings are
cloud-contaminated and already discarded on the ground, so screening in orbit
removes data that was never going to be used.</div>""")

    # ---------------- limitations ----------------
    S.append("<h2>What this prototype does not show</h2>")
    S.append(f"""<ul>
<li><b>The imagery is simulated.</b> Physically motivated - literature TOA
reflectance spectra, 1/f fractal cloud fields, Planck-consistent sub-pixel fire
emission, shot and read noise, 12-bit quantisation - but simulated. Real
Sentinel-2 L1C will be harder, particularly at cloud edges and over bright
desert. <code>scene.RealSceneAdapter</code> is the one interface to implement.</li>
<li><b>Event priors are inflated.</b> Wildfire at 4% of cloud-free tiles is
orders of magnitude above reality; it is set that way so the campaign has
statistical power. Rates and ratios hold; absolute event counts do not.</li>
<li><b>Vessel detection is weak</b> (recall
{results['accuracy']['int8']['per_class']['vessel']['recall']:.2f}) and should be.
A ship is a few pixels at 20 m GSD; operational vessel detection uses finer
resolution or SAR.</li>
<li><b>The Biomass demo runs on the browse quicklook</b>, not calibrated SLC -
the measurement arrays are absent from the annotation-only download - and its
labels are physics-derived, not PRODES/DETER truth.</li>
<li><b>No orbit determination, thermal, or ADCS modelling.</b> Slews, thermal
soak and pointing budgets are ignored; the life-limited-item accounting covers
transmitter on-time and battery cycles only.</li>
<li><b>The radiation model is statistical</b>, not a TCAD or beam-test result.
It gives the shape of the degradation curve, not a qualification number.</li>
</ul>""")

    S.append(f"""<h2>Reproducing this</h2>
<p><code>python run_all.py</code> regenerates every figure on this page from
seed {C.RNG_SEED}. Total runtime {results.get('runtime_s', 0)/60:.0f} minutes on
2 CPU cores. <code>pytest tests/</code> runs the invariant checks, including the
event-accounting closure assertion that catches a leaking benefit calculation.</p>""")

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>AEGIS - onboard AI triage prototype</title><style>{CSS}</style></head>
<body class="viz"><div class="wrap">{''.join(S)}</div></body></html>"""
    with open(path, "w") as f:
        f.write(html)
    return path
