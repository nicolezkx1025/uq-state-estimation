"""极简 SVG 绘图(不依赖 matplotlib),用于生成可靠性曲线图。"""
from __future__ import annotations

W, H = 620, 420
PAD_L, PAD_R, PAD_T, PAD_B = 70, 30, 40, 55
COLORS = ["#c0392b", "#1f77b4", "#2e8b57", "#8e44ad"]


def _sx(v, x0, x1):
    return PAD_L + (v - x0) / (x1 - x0) * (W - PAD_L - PAD_R)


def _sy(v, y0, y1):
    return H - PAD_B - (v - y0) / (y1 - y0) * (H - PAD_T - PAD_B)


def reliability_svg(curves: dict, out_path: str, title: str):
    """curves: {label: [(nominal, empirical), ...]}"""
    xs = [p[0] for c in curves.values() for p in c] + [0.0, 1.0]
    ys = [p[1] for c in curves.values() for p in c] + [0.0, 1.0]
    x0, x1 = min(xs) - 0.03, max(xs) + 0.03
    y0, y1 = min(ys) - 0.03, max(ys) + 0.03

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="Helvetica,Arial,sans-serif" font-size="12">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        f'<text x="{W/2}" y="22" text-anchor="middle" font-size="14" font-weight="bold">{title}</text>',
    ]

    # 网格
    for i in range(11):
        v = i / 10
        gx, gy = _sx(v, x0, x1), _sy(v, y0, y1)
        parts.append(f'<line x1="{gx:.1f}" y1="{PAD_T}" x2="{gx:.1f}" y2="{H-PAD_B}" stroke="#eee"/>')
        parts.append(f'<line x1="{PAD_L}" y1="{gy:.1f}" x2="{W-PAD_R}" y2="{gy:.1f}" stroke="#eee"/>')

    # 对角线(理想校准)
    parts.append(
        f'<line x1="{_sx(x0, x0, x1):.1f}" y1="{_sy(y0, y0, y1):.1f}" '
        f'x2="{_sx(x1, x0, x1):.1f}" y2="{_sy(y1, y0, y1):.1f}" '
        f'stroke="#999" stroke-dasharray="5,4"/>'
    )

    # 曲线
    for i, (label, pts) in enumerate(curves.items()):
        pts = sorted(pts)
        d = " ".join(
            ("M" if j == 0 else "L") + f"{_sx(p, x0, x1):.1f},{_sy(q, y0, y1):.1f}"
            for j, (p, q) in enumerate(pts)
        )
        color = COLORS[i % len(COLORS)]
        parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2"/>')
        for p, q in pts:
            parts.append(f'<circle cx="{_sx(p, x0, x1):.1f}" cy="{_sy(q, y0, y1):.1f}" r="3" fill="{color}"/>')
        ly = PAD_T + 6 + i * 18
        parts.append(f'<line x1="{W-PAD_R-135}" y1="{ly}" x2="{W-PAD_R-115}" y2="{ly}" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<text x="{W-PAD_R-110}" y="{ly+4}" font-size="11">{label}</text>')

    # 轴与标签
    parts.append(f'<line x1="{PAD_L}" y1="{PAD_T}" x2="{PAD_L}" y2="{H-PAD_B}" stroke="#333"/>')
    parts.append(f'<line x1="{PAD_L}" y1="{H-PAD_B}" x2="{W-PAD_R}" y2="{H-PAD_B}" stroke="#333"/>')
    for i in range(6):
        v = i / 5
        parts.append(f'<text x="{_sx(v, x0, x1):.1f}" y="{H-PAD_B+16}" text-anchor="middle" font-size="10">{v:.1f}</text>')
        parts.append(f'<text x="{PAD_L-8}" y="{_sy(v, y0, y1)+4:.1f}" text-anchor="end" font-size="10">{v:.1f}</text>')
    parts.append(f'<text x="{(PAD_L+W-PAD_R)/2:.0f}" y="{H-14}" text-anchor="middle" font-size="11">标称置信水平 (nominal)</text>')
    parts.append(
        f'<text x="16" y="{(PAD_T+H-PAD_B)/2:.0f}" font-size="11" text-anchor="middle" '
        f'transform="rotate(-90 16 {(PAD_T+H-PAD_B)/2:.0f})">实测覆盖率 (empirical)</text>'
    )
    parts.append("</svg>")

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))
    return out_path
