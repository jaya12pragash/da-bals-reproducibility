"""Generate the print-readable vector DA-BALS architecture figure."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PDF = ROOT / "figures" / "fig_architecture_flow.pdf"
OUTPUT_PNG = ROOT / "figures" / "fig_architecture_flow.png"

BLUE = "#00629B"
DARK = "#253746"
CORE_FILL = "#EAF4FA"
INPUT_FILL = "#F4F5F6"
STATE_FILL = "#E9F3EE"
SHED_FILL = "#FBEDEA"


def box(ax, x, y, width, height, text, *, fill=CORE_FILL, edge=BLUE,
        dashed=False, fontsize=10.0):
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.045,rounding_size=0.07",
        facecolor=fill,
        edgecolor=edge,
        linewidth=1.45,
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=DARK,
        linespacing=1.12,
    )


def arrow(ax, start, end, *, color=DARK, dashed=False, label=None,
          label_xy=None, connection="arc3,rad=0"):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.35,
        color=color,
        linestyle="--" if dashed else "-",
        connectionstyle=connection,
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(patch)
    if label and label_xy:
        ax.text(
            *label_xy,
            label,
            ha="center",
            va="center",
            fontsize=9.0,
            color=color,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0),
        )


def main():
    fig, ax = plt.subplots(figsize=(8.0, 2.65))
    ax.set_xlim(0.0, 11.1)
    ax.set_ylim(0.0, 5.1)
    ax.axis("off")

    # Main request path: one uninterrupted left-to-right sequence.
    box(ax, 0.12, 2.12, 1.48, 0.88, "Request\narrival", fontsize=9.5)
    box(ax, 2.12, 2.12, 1.82, 0.88, "Admission\ntest", fontsize=9.5)
    box(ax, 4.48, 2.12, 1.52, 0.88, "Reserved\nuplink", fontsize=9.3)
    box(ax, 6.55, 2.12, 1.62, 0.88, "Shared cloud\nFIFO", fontsize=9.2)
    box(ax, 8.72, 2.12, 1.62, 0.88, "On-time\nor late", fontsize=9.3)

    arrow(ax, (1.60, 2.56), (2.12, 2.56))
    arrow(ax, (3.94, 2.56), (4.48, 2.56))
    arrow(ax, (6.00, 2.56), (6.55, 2.56))
    arrow(ax, (8.17, 2.56), (8.72, 2.56))

    # External inputs and the reject branch remain visually separate.
    box(ax, 2.00, 4.00, 2.06, 0.68, "Delayed telemetry",
        fill=INPUT_FILL, edge="#687078", dashed=True, fontsize=9.2)
    box(ax, 6.48, 4.00, 1.76, 0.68, "Exogenous\ncloud traffic",
        fill=INPUT_FILL, edge="#687078", dashed=True, fontsize=8.9)
    box(ax, 4.70, 4.00, 1.10, 0.68, "Shed",
        fill=SHED_FILL, edge="#B24735", fontsize=9.8)

    arrow(ax, (3.03, 4.00), (3.03, 3.00), dashed=True)
    arrow(ax, (7.36, 4.00), (7.36, 3.00), dashed=True)
    arrow(ax, (3.62, 3.00), (4.70, 4.10), color="#B24735")

    # Completion feedback occupies its own lower lane, avoiding crossed arrows.
    box(ax, 8.34, 0.38, 2.28, 0.72, "Completion residual  $e_j$",
        fill=STATE_FILL, edge="#317A59", fontsize=8.8)
    box(ax, 5.13, 0.38, 2.38, 0.72, "EWMA + cap + decay",
        fill=STATE_FILL, edge="#317A59", fontsize=9.1)
    box(ax, 2.22, 0.38, 1.82, 0.72, "Margin  $m_t$",
        fill=STATE_FILL, edge="#317A59", fontsize=9.2)

    arrow(ax, (9.53, 2.12), (9.53, 1.10), color="#317A59")
    arrow(ax, (8.34, 0.74), (7.51, 0.74), color="#317A59")
    arrow(ax, (5.13, 0.74), (4.04, 0.74), color="#317A59")
    arrow(ax, (3.13, 1.10), (3.13, 2.12), color="#317A59")

    fig.tight_layout(pad=0.10)
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(OUTPUT_PNG, dpi=600, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


if __name__ == "__main__":
    main()
