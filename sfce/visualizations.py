import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats



def scatter_with_fit(
    x,
    y,
    title: str,
    suptitle: str,
    xlabel,
    ylabel,
    filename="scatter_with_fit.png",
):
    """Create a styled scatter plot with regression line and Spearman correlation.

    Parameters
    ----------
    x, y : array-like
        Input data of the same length. Both axes will use a log scale,
        so all values must be positive.
    title : str
        Axes-level title.
    suptitle : str
        Figure-level subtitle displayed below the title.
    xlabel, ylabel : str
        Axis labels.
    filename : str, optional
        Path where the figure will be saved.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    # Spearman correlation (rank-based, robust to outliers)
    res = stats.spearmanr(x, y)
    r = res.statistic
    p = res.pvalue

    sns.set_theme(style="whitegrid", context="talk")

    fig, ax = plt.subplots(figsize=(16,10))

    # Log-scale scatter
    sns.scatterplot(
        x=x,
        y=y,
        ax=ax,
        s=70,
        color="#3182bd",
        edgecolor="white",
        linewidth=0.8,
        alpha=0.8,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    # Regression line fitted in log-log space, then drawn back in data space
    log_x = np.log10(x)
    log_y = np.log10(y)
    slope, intercept, *_ = stats.linregress(log_x, log_y)
    x_line = np.logspace(np.log10(x.min()), np.log10(x.max()), 300)
    y_line = 10 ** (intercept + slope * np.log10(x_line))
    ax.plot(x_line, y_line, color="#e6550d", linewidth=2.5)

    # Title and labels (ylabel was mistakenly set to xlabel in the original)
    ax.set_title(title, fontsize=16, pad=15)
    plt.suptitle(suptitle, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)

    # Spearman correlation annotation
    annot = f"Spearman ρ = {r:.2f}\n(p = {p:.3g})"
    ax.text(
        0.02,
        0.98,
        annot,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="#9e9e9e",
            alpha=0.9,
        ),
    )

    sns.despine(trim=True)
    fig.tight_layout()

    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return r, p


def histogram(
    data,
    title: str,
    suptitle: str,
    xlabel: str,
    label_a: str = "Group A",
    label_b: str = "Group B",
    filename: str = "histogram.png",
):
    """Plot two overlapping histograms for Group A and Group B.

    The input array is split exactly in half — the first half becomes
    Group A and the second half becomes Group B.

    Parameters
    ----------
    data : array-like
        Input data of even length. Split down the middle into two groups.
    title : str
        Axes-level title.
    suptitle : str
        Figure-level subtitle displayed below the title.
    xlabel : str
        Label for the x-axis (the measured variable).
    label_a : str, optional
        Legend label for the first group.
    label_b : str, optional
        Legend label for the second group.
    filename : str, optional
        Path where the figure will be saved.
    """
    data = np.asarray(data, dtype=float)

    if len(data) % 2 != 0:
        raise ValueError("data must have an even length to split into two equal groups")

    mid = len(data) // 2
    group_a = data[:mid]
    group_b = data[mid:]

    sns.set_theme(style="whitegrid", context="talk")

    fig, ax = plt.subplots(figsize=(16,10))

    sns.histplot(
        group_a,
        ax=ax,
        kde=True,
        color="#3182bd",   # soft blue
        label=label_a,
        alpha=0.55,
        linewidth=0.6,
    )
    sns.histplot(
        group_b,
        ax=ax,
        kde=True,
        color="#e6550d",   # warm orange
        label=label_b,
        alpha=0.55,
        linewidth=0.6,
    )

    ax.set_title(title, fontsize=16, pad=15)
    plt.suptitle(suptitle, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.legend(fontsize=12)

    sns.despine(trim=True)
    fig.tight_layout()

    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)

def log_histogram(data,
    title: str,
    suptitle: str,
    xlabel: str,
    label_a: str = "Group A",
    label_b: str = "Group B",
    filename: str = "histogram.png",
):
    """Plot two overlapping histograms for Group A and Group B on a log y-axis.

    The input array is split exactly in half — the first half becomes
    Group A and the second half becomes Group B.

    Parameters
    ----------
    data : array-like
        Input data of even length. Split down the middle into two groups.
    title : str
        Axes-level title.
    suptitle : str
        Figure-level subtitle displayed below the title.
    xlabel : str
        Label for the x-axis (the measured variable).
    label_a : str, optional
        Legend label for the first group.
    label_b : str, optional
        Legend label for the second group.
    filename : str, optional
        Path where the figure will be saved.
    """
    data = np.asarray(data, dtype=float)

    if len(data) % 2 != 0:
        raise ValueError("data must have an even length to split into two equal groups")

    mid = len(data) // 2
    group_a = data[:mid]
    group_b = data[mid:]

    sns.set_theme(style="whitegrid", context="talk")

    # Shared bin edges across the full dataset
    bins = np.histogram_bin_edges(data, bins=30)
    widths = np.diff(bins)
    centers = bins[:-1] + widths / 2

    counts_a, _ = np.histogram(group_a, bins=bins)
    counts_b, _ = np.histogram(group_b, bins=bins)

    fig, ax = plt.subplots(figsize=(16,10))

    ax.bar(centers, counts_a, width=widths * 0.9, bottom=0,
           color="#f0e929", alpha=0.6, linewidth=0.8, label=label_a)

    ax.bar(centers, counts_b, width=widths * 0.9, bottom=0,
           fill=True, alpha=0.5, color="#3182bd", linewidth=1.5, label=label_b)

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(plt.LogFormatter(labelOnlyBase=False))

    ax.set_title(title, fontsize=16, pad=15)
    plt.suptitle(suptitle, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel("Count (log scale)", fontsize=13)
    ax.legend(fontsize=12)

    sns.despine(trim=True)
    fig.tight_layout()

    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def boxplot(
    data,
    title: str,
    suptitle: str,
    xlabel: str,
    label_a: str = "Group A",
    label_b: str = "Group B",
    filename: str = "boxplot.png",
):
    """Plot two side-by-side box-and-whisker plots for Group A and Group B.

    The input array is split exactly in half — the first half becomes
    Group A and the second half becomes Group B.

    Parameters
    ----------
    data : array-like
        Input data of even length. Split down the middle into two groups.
    title : str
        Axes-level title.
    suptitle : str
        Figure-level subtitle displayed below the title.
    xlabel : str
        Label for the x-axis (the group axis).
    label_a : str, optional
        Legend label for the first group.
    label_b : str, optional
        Legend label for the second group.
    filename : str, optional
        Path where the figure will be saved.
    """
    data = np.asarray(data, dtype=float)

    if len(data) % 2 != 0:
        raise ValueError("data must have an even length to split into two equal groups")

    mid = len(data) // 2
    df = pd.DataFrame({
        "value": data,
        "group": [label_a] * mid + [label_b] * (len(data) - mid),
    })

    sns.set_theme(style="whitegrid", context="talk")

    fig, ax = plt.subplots(figsize=(16,10))

    sns.boxplot(
        data=df,
        x="group",
        y="value",
        ax=ax,
        palette={"Group A": "#3182bd", "Group B": "#e6550d"} if label_a == "Group A"
                else {label_a: "#3182bd", label_b: "#e6550d"},
        width=0.45,
        linewidth=1.5,
        flierprops=dict(
            marker="o",
            markerfacecolor="white",
            markeredgewidth=1.2,
            markersize=5,
            alpha=0.7,
        ),
    )

    # Overlay a strip of individual points for transparency
    sns.stripplot(
        data=df,
        x="group",
        y="value",
        ax=ax,
        palette={label_a: "#3182bd", label_b: "#e6550d"},
        size=4,
        alpha=0.35,
        jitter=True,
        linewidth=0.4,
    )

    ax.set_title(title, fontsize=16, pad=15)
    plt.suptitle(suptitle, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel("Value", fontsize=13)

    sns.despine(trim=True)
    fig.tight_layout()

    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)

def trimmed_quantile_boxplot(
    data,
    title: str,
    suptitle: str,
    xlabel: str,
    label_a: str = "Group A",
    label_b: str = "Group B",
    trim_quantile: float | None = 0.001,
    filename: str = "boxplot.png",
):
    """Plot two side-by-side box-and-whisker plots for Group A and Group B.

    Parameters
    ----------
    data : array-like
        1-D array of numeric values.
    title : str
        Axes-level title.
    suptitle : str
        Figure-level subtitle displayed below the title.
    xlabel : str
        Label for the x-axis (the group axis).
    label_a : str, optional
        Display name for the first group.
    label_b : str, optional
        Display name for the second group.
    trim_quantile : float or None, optional
        If set, clips the data to [trim_quantile, 1 - trim_quantile]
        using global percentiles before plotting. Must be in (0, 0.5).
    filename : str, optional
        Path where the figure will be saved.
    """
    data = np.asarray(data, dtype=float)

    if len(data) % 2 != 0:
        raise ValueError("data must have an even length to split into two equal groups")

    mid = len(data) // 2
    group_col = np.array([label_a] * mid + [label_b] * (len(data) - mid))

    if trim_quantile is not None:
        if not (0.0 < trim_quantile < 0.5):
            raise ValueError("trim_quantile must be between 0 and 0.5 exclusive")
        lo = np.quantile(data, trim_quantile)
        hi = np.quantile(data, 1.0 - trim_quantile)
        mask = (data >= lo) & (data <= hi)
        data      = data[mask]
        group_col = group_col[mask]

    df = pd.DataFrame({"value": data, "group": group_col})

    palette = {label_a: "#3182bd", label_b: "#e6550d"}

    sns.set_theme(style="whitegrid", context="talk")

    fig, ax = plt.subplots(figsize=(16,10))

    sns.boxplot(
        data=df,
        x="group",
        y="value",
        ax=ax,
        palette=palette,
        order=[label_a, label_b],
        width=0.45,
        linewidth=1.5,
        flierprops=dict(
            marker="o",
            markerfacecolor="white",
            markeredgewidth=1.2,
            markersize=5,
            alpha=0.3,
        ),
    )

    sns.stripplot(
        data=df,
        x="group",
        y="value",
        ax=ax,
        palette=palette,
        order=[label_a, label_b],
        size=4,
        alpha=0.15,
        jitter=True,
        linewidth=0.4,
    )

    ax.set_title(title, fontsize=16, pad=15)
    plt.suptitle(suptitle, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel("Value", fontsize=13)

    sns.despine(trim=True)
    fig.tight_layout()

    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)