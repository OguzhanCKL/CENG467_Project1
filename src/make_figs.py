"""1

Generates:
  - scatter_accuracy_cost.{pdf,png}  : accuracy vs. API calls per problem
  - depth_ablation.{pdf,png}         : ToT depth=3 vs depth=6 (dual-axis)

Style choices follow common scientific publication conventions:
  - serif font, single accent color per family
  - subtle dashed grid, no top/right spines
  - Pareto frontier shown explicitly
  - vector PDF + 300dpi PNG fallback
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---- Global style -----------------------------------------------------------
mpl.rcParams.update({
    "font.family":     "serif",
    "font.serif":      ["DejaVu Serif", "Times New Roman", "Times"],
    "font.size":        11,
    "axes.labelsize":   12,
    "axes.titlesize":   13,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
    "axes.grid":        True,
    "grid.alpha":       0.25,
    "grid.linestyle":   "--",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "savefig.dpi":       300,
    "savefig.bbox":     "tight",
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
})

# Accuracy values after correcting one GSM8K gold-label annotation error
# (carnival problem: official gold $2,280 is wrong; correct total is $2,180).
# - ZS-CoT and ToT(d=6) gain +1 because they had produced the mathematically
#   correct answer (2180) which the original gold label scored as wrong.
# - FS-CoT (1880), ToT d=3 (1870), and SC (1880) independently produced
#   incorrect answers, so their scores are unchanged.
# Sample sizes (N) remain at the originally evaluated counts.
STRATEGIES = [
    # (label, API calls per problem, accuracy %, marker style group)
    ("Zero-Shot CoT",          1,  88.0, "cot"),
    ("Few-Shot CoT",           1,  85.0, "cot"),
    ("ToT (depth=3)",         21,  81.0, "tot"),
    ("ToT (depth=6)",         45,  85.7, "tot"),
    ("Self-Consistency (k=5)", 5,  90.8, "sc"),
]

STYLE = {
    "cot": dict(color="#1f4e79", marker="o", label_family="CoT"),
    "tot": dict(color="#c0504d", marker="s", label_family="ToT"),
    "sc":  dict(color="#2e7d32", marker="D", label_family="Self-Consistency"),
}


# ---- Figure 1: Accuracy vs. cost --------------------------------------------
def make_scatter():
    fig, ax = plt.subplots(figsize=(6.8, 4.3))

    # Plot points
    for name, cost, acc, fam in STRATEGIES:
        s = STYLE[fam]
        ax.scatter(cost, acc, s=130, c=s["color"], marker=s["marker"],
                   edgecolors="black", linewidth=0.8, zorder=3)

    # Pareto frontier (lower cost & higher accuracy strictly dominate)
    pts = sorted([(c, a, n) for n, c, a, _ in STRATEGIES], key=lambda t: t[0])
    frontier = []
    best_acc = -1
    for c, a, n in pts:
        if a > best_acc:
            frontier.append((c, a, n))
            best_acc = a
    fx = [p[0] for p in frontier]
    fy = [p[1] for p in frontier]
    ax.plot(fx, fy, color="black", linestyle="--", linewidth=1.2,
            alpha=0.55, zorder=2, label="Pareto frontier")

    # Annotations
    offsets = {
        "Zero-Shot CoT":          ( 10,   6),
        "Few-Shot CoT":           ( 10,  -14),
        "ToT (depth=3)":          (-12,   8),
        "ToT (depth=6)":          (-12,  10),
        "Self-Consistency (k=5)": ( 12,  -2),
    }
    for name, cost, acc, _ in STRATEGIES:
        dx, dy = offsets[name]
        ax.annotate(name, xy=(cost, acc), xytext=(dx, dy),
                    textcoords="offset points", fontsize=10,
                    ha="left" if dx >= 0 else "right")

    # Family legend (separate from frontier)
    family_handles = [
        plt.Line2D([0], [0], marker=STYLE[f]["marker"], color="w",
                   markerfacecolor=STYLE[f]["color"], markeredgecolor="black",
                   markersize=10, label=STYLE[f]["label_family"])
        for f in ("cot", "tot", "sc")
    ]
    frontier_handle = plt.Line2D([0], [0], color="black", linestyle="--",
                                 linewidth=1.2, label="Pareto frontier")

    ax.legend(handles=family_handles + [frontier_handle],
              loc="lower left", frameon=True, framealpha=0.96,
              edgecolor="gray", borderpad=0.8)

    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50])
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlim(0.6, 100)
    ax.set_ylim(78, 95)
    ax.set_xlabel("API calls per problem (log scale)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs. Inference Cost on GSM8K (Llama 3.1 8B)")

    plt.savefig("scatter_accuracy_cost.pdf")
    plt.savefig("scatter_accuracy_cost.png")
    plt.close(fig)


# ---- Figure 2: Depth ablation -----------------------------------------------
def make_depth_ablation():
    depths = ["depth = 3", "depth = 6"]
    accs   = [81.0, 85.7]
    costs  = [21, 45]
    delta_acc = accs[1] - accs[0]
    factor_cost = costs[1] / costs[0]

    fig, ax1 = plt.subplots(figsize=(6.5, 4.2))

    x = np.arange(len(depths))
    width = 0.34

    acc_color  = "#1f4e79"
    cost_color = "#c0504d"

    bars1 = ax1.bar(x - width / 2, accs, width,
                    color=acc_color, edgecolor="black", linewidth=0.7,
                    label="Accuracy (%)")
    ax1.set_ylabel("Accuracy (%)", color=acc_color)
    ax1.tick_params(axis="y", labelcolor=acc_color)
    ax1.set_ylim(0, 100)
    ax1.set_xticks(x)
    ax1.set_xticklabels(depths)
    ax1.set_xlabel("ToT maximum search depth")

    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    bars2 = ax2.bar(x + width / 2, costs, width,
                    color=cost_color, edgecolor="black", linewidth=0.7,
                    label="API calls per problem")
    ax2.set_ylabel("API calls per problem", color=cost_color)
    ax2.tick_params(axis="y", labelcolor=cost_color)
    ax2.set_ylim(0, 60)

    for b, v in zip(bars1, accs):
        ax1.annotate(f"{v:.1f}%", xy=(b.get_x() + b.get_width() / 2, v),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", fontsize=10, color=acc_color, weight="bold")
    for b, v in zip(bars2, costs):
        ax2.annotate(f"{v}", xy=(b.get_x() + b.get_width() / 2, v),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", fontsize=10, color=cost_color, weight="bold")

    # Highlight the trade-off as inline text (placed between the two depth groups,
    # below the value labels and above the bar tops, so it never overlaps the legend)
    tradeoff = (f"$\\Delta$ accuracy: $+{delta_acc:.1f}$ pp at "
                f"${factor_cost:.1f}\\times$ cost")
    ax1.text(0.5, 0.55, tradeoff, transform=ax1.transAxes,
             ha="center", fontsize=10.5,
             bbox=dict(boxstyle="round,pad=0.4",
                       facecolor="white", edgecolor="gray", alpha=0.95))

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right",
               frameon=True, framealpha=0.96, edgecolor="gray")

    ax1.set_title("ToT Depth Ablation: Accuracy and Cost (GSM8K, Llama 3.1 8B)")
    ax1.grid(axis="y", alpha=0.25, linestyle="--")

    plt.savefig("depth_ablation.pdf")
    plt.savefig("depth_ablation.png")
    plt.close(fig)


if __name__ == "__main__":
    make_scatter()
    make_depth_ablation()
    print("Saved: scatter_accuracy_cost.{pdf,png}, depth_ablation.{pdf,png}")
