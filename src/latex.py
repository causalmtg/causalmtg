import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def value_per_quanta(
    x,
    ys,
    num_bins=10,
    quantile_based=True,
    ci_level=0.95,
    xlabel='x',
    ylabel='y',
    title='Value-per-Quanta Visualization',
    labels=None,
    show=True,
    show_y_equals_x=False,
    show_last_bin=False,
    x_transform=False,
    x_tick_digits=2
):
    """
    Plot value-per-quanta for one or multiple y variables.

    Parameters:
        x (array-like): x-values
        ys (array-like or list of array-like): one or multiple y variables
        num_bins (int): Number of bins/quanta
        quantile_based (bool): If True, use quantile-based bins;
                               otherwise equal-width bins
        ci_level (float or None): Confidence interval level, or None
        xlabel (str): x-axis label
        ylabel (str): Single y-axis label
        title (str): Plot title
        labels (list of str or None): Name/legend label for each y
        show (bool): If True, display the plot
        show_y_equals_x (bool): If True, show dashed y=x reference line
        show_last_bin (bool): If True, include the last bin; default is False
        x_transform (bool): If True, plot the bins at evenly spaced
                            positions instead of according to their x-values
        x_tick_digits (int): Number of decimal digits used to round x-axis
                             tick labels
    """

    x = np.asarray(x)

    # Allow a single y array as well as multiple y arrays
    if isinstance(ys, np.ndarray) and ys.ndim == 1:
        ys = [ys]
    else:
        ys = [np.asarray(y) for y in ys]

    # Labels for legend
    if labels is None:
        labels = [f'y{i+1}' for i in range(len(ys))]

    if len(labels) != len(ys):
        raise ValueError("labels must have the same length as ys")

    if any(len(y) != len(x) for y in ys):
        raise ValueError("x and all y arrays must have the same length")

    # Define bin edges based only on x
    if quantile_based:
        edges = np.quantile(
            x,
            np.linspace(0, 1, num_bins + 1)
        )
    else:
        edges = np.linspace(
            np.min(x),
            np.max(x),
            num_bins + 1
        )

    # Construct identical bins for all y's
    bin_masks = []
    bin_centers = []

    num_bins_to_use = num_bins if show_last_bin else num_bins - 1

    for i in range(num_bins_to_use):

        if i < num_bins - 1:
            mask = (x >= edges[i]) & (x < edges[i + 1])
        else:
            mask = (x >= edges[i]) & (x <= edges[i + 1])

        if np.sum(mask) == 0:
            continue

        bin_masks.append(mask)
        bin_centers.append(np.mean(x[mask]))

    bin_centers = np.asarray(bin_centers)

    # Use evenly spaced positions for the bins if requested
    if x_transform:
        plot_x = np.arange(len(bin_centers))
    else:
        plot_x = bin_centers

    # Plot
    plt.figure(figsize=(5, 5))

    # Plot each y
    for y, label in zip(ys, labels):

        bin_means = []
        ci_lower = []
        ci_upper = []

        for mask in bin_masks:

            y_bin = y[mask]
            mean_y = np.mean(y_bin)
            bin_means.append(mean_y)

            if ci_level is not None:
                se = stats.sem(y_bin)
                df = len(y_bin) - 1

                t_val = (
                    stats.t.ppf((1 + ci_level) / 2, df)
                    if df > 0
                    else 0
                )

                ci_lower.append(mean_y - t_val * se)
                ci_upper.append(mean_y + t_val * se)

        bin_means = np.asarray(bin_means)

        line, = plt.plot(
            plot_x,
            bin_means,
            lw=2,
            marker='o',
            label=label
        )

        if ci_level is not None:
            plt.fill_between(
                plot_x,
                ci_lower,
                ci_upper,
                color=line.get_color(),
                alpha=0.15
            )

    # Optional y=x reference
    if show_y_equals_x:
        x_min = np.min(bin_centers)
        x_max = np.max(bin_centers)

        if x_transform:
            ref_x = [0, len(bin_centers) - 1]
        else:
            ref_x = [x_min, x_max]

        plt.plot(
            ref_x,
            [x_min, x_max],
            '--',
            color='black',
            lw=1.5,
            label=r'$y=x$'
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if title:
        plt.title(title)

    # Show rounded original bin-center values as x-axis labels.
    # Skip a tick if its rounded value equals the previous displayed value.
    if x_transform:
        tick_positions = []
        tick_labels = []

        previous_label = None

        for pos, value in zip(plot_x, bin_centers):
            rounded_value = round(value, x_tick_digits)

            if rounded_value == previous_label:
                continue

            tick_positions.append(pos)
            tick_labels.append(f'{rounded_value:.{x_tick_digits}f}')
            previous_label = rounded_value

        plt.xticks(tick_positions, tick_labels)

    plt.legend()
    plt.grid(True)

    if show:
        plt.show()

METRICS_BY_SUBSET = {
    "all": [
        ("rmse", r"RMSE ($\times100$) $\downarrow$"),
        ("mae", r"MAE ($\times100$) $\downarrow$"),
        ("ccc", r"CCC (\%) $\uparrow$"),        
        ("accuracy95", r"Acc@95 (\%) $\uparrow$"),        
    ],
}

METRICS_CAPTIONS = [
    ("rmse", r"RMSE ($\times100$) $\downarrow$"),
    ("mae", r"MAE ($\times100$) $\downarrow$"),
    ("ccc", r"CCC (\%) $\uparrow$"),        
    ("accuracy95", r"Acc@95 (\%) $\uparrow$"),        
]

SUBSETS = [
    ("all", "ALL"),
]


# ============================================================
# Formatting
# ============================================================

TRUTH_DIVISOR = 1
DIGITS = 2


def scale_metric(metric, value, sci):
    """
    Convert normalized results into the presentation scale.

    RMSE / MAE:
        normalized error -> original outcome scale

    accuracy95 / SignAcc / CCC:
        fraction -> percentage
    """
    
    if pd.isna(value):
        return value, value

    if metric in ("rmse", "mae"):
        factor = 100

    elif metric in ("accuracy95", "ccc"):
        factor = 100

    else:
        factor = 1

    value = value * factor

    if not pd.isna(sci):
        sci = sci * factor

    return value, sci


# ============================================================
# Main LaTeX generator
# ============================================================
def make_results_latex(
    df,
    subsets,
    METHOD_NAMES,
    method_col="method",    
):
    lines = []

    # --------------------------------------------------------
    # Table setup
    # --------------------------------------------------------

    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\setlength{\tabcolsep}{3pt}")

    # Build column specification dynamically
    col_spec_parts = ["l"]

    for subset_code, _subset_name in subsets:
        n_metrics = len(METRICS_BY_SUBSET[subset_code])
        col_spec_parts.append("c" * n_metrics)

    # Add vertical separators between subset groups
    col_spec = "|".join(col_spec_parts)

    lines.append(
        rf"\begin{{tabular}}{{{col_spec}}}"
    )

    lines.append(r"\toprule")

    # --------------------------------------------------------
    # First header row: subset groups
    # --------------------------------------------------------

    subset_headers = []

    for i, (subset_code, subset_name) in enumerate(subsets):

        n_metrics = len(METRICS_BY_SUBSET[subset_code])

        # Vertical separator after every group except last
        if i < len(subsets) - 1:
            subset_headers.append(
                rf"\multicolumn{{{n_metrics}}}{{c|}}{{{subset_name}}}"
            )
        else:
            subset_headers.append(
                rf"\multicolumn{{{n_metrics}}}{{c}}{{{subset_name}}}"
            )

    if len(subset_headers) > 1:
        lines.append(
            r"\multirow{2}{*}{Method} & "
            + " & ".join(subset_headers)
            + r" \\"
        )

    # --------------------------------------------------------
    # Second header row: metrics
    # --------------------------------------------------------

    metric_headers = []

    for subset_code, _subset_name in subsets:

        for _metric_code, metric_name in METRICS_BY_SUBSET[
            subset_code
        ]:
            metric_headers.append(metric_name)

    lines.append(
        " & ".join([""] + metric_headers)
        + r" \\"
    )

    lines.append(r"\midrule")

    # --------------------------------------------------------
    # Methods
    # --------------------------------------------------------

    available_methods = [
        method
        for method in METHOD_NAMES
        if method in df[method_col].values
    ]

    for method_code in available_methods:

        row = df[
            df[method_col] == method_code
        ].iloc[0]

        cells = [METHOD_NAMES[method_code]]

        for subset_code, _subset_name in subsets:

            for metric_code, _metric_name in METRICS_BY_SUBSET[
                subset_code
            ]:

                value_col = f"{subset_code}_{metric_code}"
                sci_col = f"{subset_code}_{metric_code}_sci"

                value = row.get(
                    value_col,
                    float("nan")
                )

                sci = row.get(
                    sci_col,
                    float("nan")
                )

                value, sci = scale_metric(
                    metric_code,
                    value,
                    sci
                )

                if pd.isna(value):
                    text = "--"

                elif pd.isna(sci):
                    text = f"{value:.{DIGITS}f}"

                else:
                    text = (
                        f"{value:.{DIGITS}f}"
                        r" $\pm$ "
                        f"{sci:.{DIGITS}f}"
                    )

                cells.append(text)

        lines.append(
            " & ".join(cells) + r" \\"
        )

    # --------------------------------------------------------
    # Finish
    # --------------------------------------------------------

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    lines.append(
        r"""
\caption{Performance of ATT estimators on the benchmark. Each entry reports the
metric estimate and the half-width of its bootstrap confidence interval.
RMSE and MAE are scaled by $100$ for readability, while Accuracy@95 and
CCC are reported as percentages. Lower values indicate better performance
for RMSE and MAE, while higher values indicate better performance for CCC and
Accuracy@95.}"""
    )

    lines.append(r"\label{tab:main_results}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)



###################################


import pandas as pd


import pandas as pd


METRICS = {
    "mae": "lower",
    "rmse": "lower",
    "ccc": "higher",
    "accuracy95": "higher",
}


import pandas as pd

def compute_ranking_correlations(results_by_variant):
    # The direction no longer matters for the correlation calculation
    metrics = ["mae", "rmse", "ccc", "accuracy95"]
    correlations = {}

    for metric in metrics:
        
        # methods × variants
        values = pd.DataFrame({
            name: res.set_index("method")[metric]
            for name, res in results_by_variant.items()
        })

        correlations[metric] = values.corr(method="spearman")
    return correlations


def print_ranking_correlations(correlations):
    for metric, corr in correlations.items():
        print(f"\n{metric.upper()}")
        print(
            corr.to_string(
                float_format=lambda x: f"{x:.2f}"
            )
        )

def format_ranking_correlations_latex(correlations):

    metric_info = [
        ("mae", "MAE"),
        ("rmse", "RMSE"),
        ("ccc", "CCC"),
        ("accuracy95", "Accuracy@95"),
    ]

    subtables = []

    for i, (metric, caption) in enumerate(metric_info):
        corr = correlations[metric]

        # Left column: include row labels
        include_index = True #(i % 2 == 0)

        body = corr.to_latex(
            float_format=lambda x: f"{x:.2f}",
            escape=False,
            index=include_index,
        )

        subtables.append(
            f"""\\begin{{subtable}}{{0.48\\columnwidth}}
\\small
\\centering
\\caption{{{caption}}}
{body}
\\end{{subtable}}"""
        )

    return f"""\\begin{{table}}[t]
\\centering

{subtables[0]}
\\hfill
{subtables[1]}

\\vspace{{0.5em}}

{subtables[2]}
\\hfill
{subtables[3]}

\\caption{{Pairwise Spearman correlations between estimator rankings across MTG set-codes, shown separately for each evaluation metric. Higher correlations indicate greater consistency in estimator rankings across set-codes.}}
\\label{{tab:set_code_correlations}}

\\end{{table}}
"""


def rsc_table_to_latex(
    table,
    vary="r",
    fixed_value=1.96,
    label=None,
    caption=None,
    decimals=2,
):
    metric_labels = dict(METRICS_CAPTIONS)

    metrics = [m for m, _ in METRICS_CAPTIONS]

    # Column specification
    cols = [f"${vary}$", "Remaining"] + [
         metric_labels[m] for m in metrics
    ]

    # Header
    latex = []
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")

    if caption is not None:
        latex.append(f"\\caption{{{caption}}}")

    if label is not None:
        latex.append(f"\\label{{{label}}}")

    latex.append(r"\begin{tabular}{" + "c" * len(cols) + "}")
    latex.append(r"\toprule")
    latex.append(" & ".join(cols) + r" \\")
    latex.append(r"\midrule")

    # Rows
    for _, row in table.iterrows():

        value = row["value"]

        if vary == "r":
            value_str = f"{value:g}"
        else:
            value_str = f"{value:g}"

        remaining = f"{100 * row['remaining']:.0f}\\%"

        metric_values = [
            f"{row[m]:.{decimals}f}"
            for m in metrics
        ]

        latex.append(
            " & ".join([value_str, remaining] + metric_values)
            + r" \\"
        )

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    return "\n".join(latex)