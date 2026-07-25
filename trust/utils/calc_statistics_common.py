"""Shared implementation for the per-dataset CalcStatistics_* scripts.

For every (strategy, budget) this reports two metrics, aggregated over seeds
from the average-over-classes column of the per-round CSV:

- **Mean Gain**: last-round minus first-round accuracy. Note this is only
  comparable across runs that share the same initial model; runs whose
  round-0 accuracy differs (e.g. a cached initial model vs a freshly
  initialized one) get a systematically smaller gain, which inflates the
  standard deviation.
- **Mean Final Acc**: last-round accuracy, which is independent of the
  initial model and is therefore the more robust comparison.

Outputs a summary CSV, a plot, and LaTeX tables for both metrics.
"""
import csv
import os

import matplotlib.pyplot as plt
import pandas as pd

COLORS = [
    '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#00FFFF', '#FF00FF',
    '#FF4500', '#8A2BE2', '#A52A2A', '#DEB887', '#5F9EA0', '#7FFF00',
    '#D2691E', '#FF7F50', '#6495ED', '#DC143C', '#00CED1', '#9400D3',
    '#FF1493', '#00BFFF',
]


def compute_stats(gains):
    mean_gain = sum(gains) / len(gains)
    variance = sum((g - mean_gain) ** 2 for g in gains) / len(gains)
    return mean_gain, variance, variance ** 0.5


def generate_latex_table(data, dataset_label, metric='Mean Gain'):
    budgets = sorted(data['Budget'].unique())
    strategies = set(data['Strategy'].unique())
    withsoft_strategies = {s for s in strategies if 'withsoft' in s}
    main_strategies = sorted(strategies - withsoft_strategies)

    max_values = {b: data[data['Budget'] == b][metric].max()
                  for b in budgets}

    table = "\\begin{table*}[h!]\n\\centering\n\\begin{scriptsize}\n"
    table += "\\begin{tabular}{|l|*{%d}{c|}}\n\\hline\n" % len(budgets)
    table += "Strategy & " + " & ".join(map(str, budgets)) + " \\\\\n"
    table += "\\hline\n\\hline\n"

    for strategy in main_strategies:
        row = [strategy.replace("_", "\\_")]
        for budget in budgets:
            normal = data[(data['Strategy'] == strategy)
                          & (data['Budget'] == budget)]
            withsoft = data[(data['Strategy'] == strategy + "_withsoft")
                            & (data['Budget'] == budget)]
            nval = normal[metric].values[0] if not normal.empty else '-'
            wval = withsoft[metric].values[0] if not withsoft.empty else '-'
            nfmt = (f"\\textbf{{{nval}}}"
                    if nval == max_values[budget] and nval != '-' else str(nval))
            wfmt = (f"\\textbf{{{wval}}}"
                    if wval == max_values[budget] and wval != '-' else str(wval))
            row.append(f"{nfmt}({wfmt})" if wval != '-' else nfmt)
        table += " & ".join(row) + " \\\\\n\\hline\n"

    slug = metric.lower().replace(' ', '')
    table += "\\end{tabular}\n\\end{scriptsize}\n"
    table += (f"\\caption{{{metric} for various strategies across budgets for "
              f"{dataset_label}}}\n")
    table += (f"\\label{{tab:{dataset_label.lower()}{slug}}}\n"
              "\\end{table*}\n")
    return table


def run_statistics(base_dir, budgets, rounds, avg_col, strategies,
                   experiments, filename, strategy_group, dataset_label):
    output_path = os.path.join(
        base_dir, f"{filename}_group_{strategy_group}_rounds_{rounds}")

    data = {}
    with open(output_path + "_allclasses.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Strategy", "Budget", "Mean Gain", "Variance",
                         "Standard Deviation", "Mean Final Acc",
                         "Final Acc SD"])
        for budget in budgets:
            for strategy in strategies:
                cell = os.path.join(base_dir, strategy, str(budget))
                if not os.path.exists(cell):
                    continue
                gains = []
                finals = []
                for experiment in experiments:
                    path = os.path.join(cell, experiment)
                    if not os.path.exists(path):
                        continue
                    for csv_file in os.listdir(path):
                        if not csv_file.endswith('.csv'):
                            continue
                        df = pd.read_csv(os.path.join(path, csv_file),
                                         header=None)
                        final = df.iloc[rounds - 1, avg_col]
                        finals.append(final)
                        gains.append(final - df.iloc[0, avg_col])
                if not gains:
                    continue
                mean_gain, variance, sd_gain = compute_stats(gains)
                mean_final, _, sd_final = compute_stats(finals)
                mean_gain, variance, sd_gain = (round(mean_gain, 2),
                                                round(variance, 2),
                                                round(sd_gain, 2))
                mean_final, sd_final = round(mean_final, 2), round(sd_final, 2)
                writer.writerow([strategy, budget, mean_gain, variance,
                                 sd_gain, mean_final, sd_final])
                print(f"Strategy: {strategy}, Budget: {budget}, "
                      f"Mean Gain: {mean_gain} (SD {sd_gain}), "
                      f"Mean Final Acc: {mean_final} (SD {sd_final})")
                entry = data.setdefault(strategy,
                                        {'means': [], 'sds': [], 'budgets': []})
                entry['means'].append(mean_gain)
                entry['sds'].append(sd_gain)
                entry['budgets'].append(budget)
    print(f"Statistics saved to {output_path}_allclasses.csv")

    plt.figure(figsize=(10, 6))
    for color_index, (strategy, values) in enumerate(data.items()):
        color = COLORS[color_index % len(COLORS)]
        plt.plot(values['budgets'], values['means'], label=strategy,
                 color=color)
        plt.text(values['budgets'][-1], values['means'][-1], strategy,
                 fontsize=12, color=color)
    plt.xlabel('Budget')
    plt.ylabel('Mean Gain for all classes')
    plt.title(f'Mean Gain for all classes for {rounds} AL rounds '
              f'({dataset_label})')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_path + "_allclasses.png", dpi=300,
                bbox_inches='tight', pad_inches=0.1)

    df = pd.read_csv(output_path + "_allclasses.csv")
    for metric, suffix in (('Mean Gain', '_allclasses.tex'),
                           ('Mean Final Acc', '_allclasses_finalacc.tex')):
        with open(output_path + suffix, "w") as text_file:
            text_file.write(generate_latex_table(df, dataset_label, metric))
        print(f"LaTeX table ({metric}) saved to {output_path}{suffix}")
