"""Draws the A0 report figures from two trace_analysis.json files (stdlib only, SVG output).

    python3 bench/plot_trace.py results/<t2i_trace_run>/trace_analysis.json \
        results/<edit_trace_run>/trace_analysis.json --out figures
"""

from __future__ import annotations

import argparse
import json
import pathlib

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
T2I, EDIT = "#2a78d6", "#eb6834"
# (label, trace_analysis categories, fill)
CLASSES = [
    ("GEMM", ("gemm",), "#2a78d6"),
    ("Elementwise / copy", ("elementwise/copy",), "#eb6834"),
    ("FlashAttention-3", ("attention",), "#1baf7a"),
    ("Reduce + norm", ("reduce", "norm"), "#eda100"),
    ("Other", ("other", "conv"), "#e87ba4"),
]
FONT = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"


def svg(w: int, h: int, body: list[str]) -> str:
    return "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'font-family="{FONT}" font-size="12">',
        f'<rect width="{w}" height="{h}" fill="{SURFACE}"/>',
        *body,
        "</svg>\n",
    ])


def text(x, y, s, fill=MUTED, anchor="start", size=12, weight="normal") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" text-anchor="{anchor}" '
            f'font-size="{size}" font-weight="{weight}">{s}</text>')


def steps_figure(t2i: dict, edit: dict) -> str:
    w, h, left, right, top, bottom = 760, 380, 56, 150, 64, 44
    lo, hi = 100, 300
    pw, ph = w - left - right, h - top - bottom
    series = [("Edit (1 ref)", EDIT, edit["steps"]), ("T2I", T2I, t2i["steps"])]
    n = max(s["n"] for _, _, s in series)

    def x(i): return left + pw * i / (n - 1)
    def y(v): return top + ph * (hi - v) / (hi - lo)

    out = [
        text(left, 24, "GPU busy time per denoising step", INK, size=15, weight="600"),
        text(left, 42, "Kernel time attributed to each transformer forward, profiled run, ms"),
    ]
    for v in range(lo, hi + 1, 50):
        out.append(f'<line x1="{left}" x2="{left + pw}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="{GRID}"/>')
        out.append(text(left - 8, y(v) + 4, v, anchor="end"))
    for i in range(0, n, 10):
        out.append(text(x(i), top + ph + 18, i, anchor="middle"))
    out.append(text(x(n - 1), top + ph + 18, n - 1, anchor="middle"))
    out.append(text(left + pw / 2, h - 8, "Denoising step", anchor="middle"))

    # labels sit above the upper line and below the lower one so they cannot collide
    for (name, color, s), dy in zip(series, (16, -16)):
        vals = [p["gpu_busy_ms"] for p in s["per_step"]]
        pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(vals))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" '
                   f'stroke-linejoin="round" stroke-linecap="round"/>')
        out.append(f'<circle cx="{x(0):.1f}" cy="{y(vals[0]):.1f}" r="4" fill="{color}" '
                   f'stroke="{SURFACE}" stroke-width="2"/>')
        ly = y(s["rest_mean_gpu_busy_ms"]) - dy
        out.append(f'<circle cx="{left + pw + 12}" cy="{ly - 8:.1f}" r="4" fill="{color}"/>')
        out.append(text(left + pw + 22, ly - 4, name, INK, weight="600"))
        out.append(text(left + pw + 22, ly + 11, f'steps 1–{s["n"] - 1}: {s["rest_mean_gpu_busy_ms"]:.1f} ms'))

    first = edit["steps"]["first"]["gpu_busy_ms"]
    out.append(text(x(0) + 10, y(first) - 2, f"Step 0: {first:.1f} ms", INK, weight="600"))
    out.append(text(x(0) + 10, y(first) + 13, "includes prefix prefill of the reference image"))
    return svg(w, h, out)


def kernels_figure(t2i: dict, edit: dict) -> str:
    w, h, left, right, top = 760, 250, 96, 70, 96
    bar, gap = 34, 30
    pw = w - left - right
    rows = [("T2I", t2i), ("Edit (1 ref)", edit)]

    def totals(d):
        cats = d["kernels"]["by_category"]
        return [sum(cats.get(c, {}).get("ms", 0.0) for c in keys) for _, keys, _ in CLASSES]

    data = [(name, totals(d)) for name, d in rows]
    hi = 8000
    def x(ms): return left + pw * ms / hi

    out = [
        text(left, 24, "GPU kernel time by kernel class", INK, size=15, weight="600"),
        text(left, 42, "Whole request, profiled run, ms; segment labels are the share of that request"),
    ]
    lx = left
    for label, _, color in CLASSES:
        out.append(f'<rect x="{lx}" y="58" width="10" height="10" rx="2" fill="{color}"/>')
        out.append(text(lx + 15, 67, label))
        lx += 36 + 7 * len(label)
    bottom = top + 2 * bar + gap
    for v in range(0, hi + 1, 2000):
        out.append(f'<line x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{top - 8}" y2="{bottom + 8}" stroke="{GRID}"/>')
        out.append(text(x(v), bottom + 24, f"{v:,}", anchor="middle"))
    out.append(text(left + pw / 2, h - 8, "GPU kernel time (ms)", anchor="middle"))

    for r, (name, vals) in enumerate(data):
        y0 = top + r * (bar + gap)
        total, acc = sum(vals), 0.0
        out.append(text(left - 10, y0 + bar / 2 + 4, name, INK, anchor="end", weight="600"))
        for (label, _, color), v in zip(CLASSES, vals):
            x0, x1 = x(acc), x(acc + v)
            # 2px surface gap between adjacent fills
            out.append(f'<rect x="{x0:.1f}" y="{y0}" width="{max(x1 - x0 - 2, 0.5):.1f}" height="{bar}" '
                       f'rx="2" fill="{color}"><title>{label}: {v:,.0f} ms ({100 * v / total:.1f}%)</title></rect>')
            if x1 - x0 > 44:
                out.append(text((x0 + x1) / 2 - 1, y0 + bar / 2 + 4, f"{100 * v / total:.1f}%",
                                "#ffffff", anchor="middle", weight="600"))
            acc += v
        out.append(text(x(acc) + 6, y0 + bar / 2 + 4, f"{total:,.0f} ms", INK))
    return svg(w, h, out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("t2i", help="trace_analysis.json of the T2I trace run")
    ap.add_argument("edit", help="trace_analysis.json of the edit trace run")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    t2i, edit = (json.loads(pathlib.Path(p).read_text()) for p in (args.t2i, args.edit))
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "a0_step_gpu_busy.svg").write_text(steps_figure(t2i, edit))
    (out / "a0_kernel_classes.svg").write_text(kernels_figure(t2i, edit))
    print(f"wrote {out}/a0_step_gpu_busy.svg and {out}/a0_kernel_classes.svg")


if __name__ == "__main__":
    main()
