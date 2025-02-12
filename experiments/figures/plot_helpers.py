import pandas as pd


def get_agg_metric(
    df,
    group_key=["base_name"],
    new_col_prefix="baseline",
    metrics=[
        "quality",
        "illegal_frac",
        "unique_illegal_frac",
        "illegal_count",
        "unique_illegal_count",
    ],
    retain_keys=["illegal_prob"],
):
    """
    Return a df where the columns of interest are aggregated by mean and std & renamed (prefix with a phrase)
    new_col_prefix (str): the prefix to append to aggregated cols
    """
    if any(i not in df.columns for i in retain_keys):
        raise ValueError(
            "retain_keys, e.g., `illegal_prob` column not found in the dataframe. This is needed as a dummy column to count the number of observations."
        )
    for metric in metrics:
        df[f"{metric}_std"] = df[metric]
        df = df.rename(columns={metric: f"{metric}_mean"})

    # aggregate (by mean or std)
    agg = dict()
    for col in retain_keys:
        if col == "illegal_prob":
            agg[col] = "count"
        else:
            agg[col] = "first"
    for metric in metrics:
        agg[f"{metric}_mean"] = "mean"
        agg[f"{metric}_std"] = "std"
    try:
        agg_df = df.groupby(group_key).agg(agg).reset_index()
        # rename aggregated cols with the prefix
        agg_cols = [f"{metric}_mean" for metric in metrics] + [
            f"{metric}_std" for metric in metrics
        ]
        rename_dict = {col: f"{new_col_prefix}_{col}" for col in agg_cols}
        rename_dict["illegal_prob"] = f"{new_col_prefix}_no_observations"
        agg_df = agg_df[agg_cols + group_key + retain_keys].rename(columns=rename_dict)
    except Exception as e:
        print(e)
        raise ValueError(
            "Unable to groupby or rename col, most likely the specified group_key or retain_cols don't exist."
        )
    return agg_df


def get_delay_from_name(string):
    # 4_0.01__diff_true -> 4
    delay = string.split("__")[0].split("_")[0]
    if delay != "b":
        return float(delay)
    else:
        return -1


import seaborn as sns

# orange, blue, teal, purpple, yellow, pink
BLUEPALETTE = ["#F18447", "#3863AC", "#209B8A", "#550F6B", "#F8D625", "#BC3684"]


## Format violin plot


def patch_violinplot(ax, palette=BLUEPALETTE, n=1, alpha=1, multicolor=True):
    """
    Recolor the outlines of violin patches using a palette
    - palette (list of str): color palette for the patches
    - n (int): number of colors to use from the palette
    - multicolor (bool): whether to color the patches differently. If False, use the default color (orange)
    - alpha (float): transparency
    """
    from matplotlib.collections import PolyCollection

    violins = [art for art in ax.get_children() if isinstance(art, PolyCollection)]
    for i in range(len(violins)):
        if multicolor is False:
            violins[i].set_edgecolor(c="#F18447")
        else:
            colors = sns.color_palette(palette, n_colors=n) * (len(violins) // n)
            violins[i].set_edgecolor(colors[i])
        violins[i].set_alpha(alpha)


def point_violinplot(
    ax,
    palette=BLUEPALETTE,
    n=1,
    pointsize=200,
    edgecolor="white",
    multicolor=True,
    linewidth=1.5,
):
    """
    Recolor points in the plot based on the violin facecolor
    - palette (list of str): color palette for the patches
    - n (int): number of colors to use from the palette
    - edgecolor (str): point outline color
    - pointsize (int): point size
    - multicolor (bool): whether to color the patches differently. If False, use the default color (orange)
    - alpha (float): transparency
    """
    from matplotlib.collections import PathCollection

    violins = [art for art in ax.get_children() if isinstance(art, PathCollection)]
    for i in range(len(violins)):
        violins[i].set_sizes([pointsize])  # size
        violins[i].set_edgecolor(edgecolor)  # outline
        violins[i].set_linewidth(linewidth)
        if multicolor is False:
            violins[i].set_facecolor(c="#F18447")
        else:
            colors = sns.color_palette(palette, n_colors=n) * (len(violins) // n)
            violins[i].set_facecolor(colors[i])


## Bonferroni correction ##


def bonferroni_correction(p_value, n_comparisons):
    """
    Perform Bonferroni correction on a given p-value for multiple comparisons.
    Parameters:
    -----------
    - p_value (float): The original p-value.
    - n_comparisons (int): The number of comparisons being made.
    Returns:
    -----------
    - float: The Bonferroni corrected p-value.
    """
    bonferroni_corrected_p_value = p_value * n_comparisons
    # Ensure the corrected p-value is not greater than 1
    bonferroni_corrected_p_value = min(1.0, bonferroni_corrected_p_value)
    return bonferroni_corrected_p_value


def get_corrected_p(plot_data, focal_col="s_H", sig_level=0.05):
    # Return a dataframe with corrected p-values for all pairs of probabilities in the plot data
    from scipy.stats import mannwhitneyu

    results = []
    probs = sorted(plot_data[focal_col].dropna().unique())
    pairs = [
        (probs[i], probs[j])
        for i in range(len(probs))
        for j in range(i + 1, len(probs))
    ]
    for pair in pairs:
        pop1 = plot_data[plot_data[focal_col] == pair[0]]["pct_change"]
        pop2 = plot_data[plot_data[focal_col] == pair[1]]["pct_change"]
        U, p = mannwhitneyu(pop1, pop2, method="exact")
        results.append({"pair": pair, "U": U, "p": p})
    stats = pd.DataFrame(results)
    stats["corrected_p"] = stats["p"].apply(
        lambda x: bonferroni_correction(x, len(pairs))
    )
    print("p-values:", stats)
    significant_pairs = stats[stats.corrected_p < sig_level]["pair"].tolist()
    print(f"Significant pairs (p<{sig_level}): ", significant_pairs)
    return stats, significant_pairs


## Annotation ##

# Define the annotation legend
annotation_legend = {
    (0.05, 1.00): "ns",
    (0.01, 0.05): "ns",
    (0.001, 0.01): "*",
    (0.0001, 0.001): "**",
    (0.0, 0.0001): "***",
}


# Function to get the annotation based on p-value
def get_annotation(p_value):
    for (lower, upper), annotation in annotation_legend.items():
        if lower < p_value <= upper:
            return annotation
    return "Invalid p-value"


# Function to add annotations
def add_annotation(ax, x1, x2, y, text):
    ax.plot([x1, x1, x2, x2], [y, y + 1, y + 1, y], lw=1.5, color="k")
    ax.text(
        (x1 + x2) * 0.5, y + 1, text, ha="center", va="bottom", color="k", fontsize=12
    )
