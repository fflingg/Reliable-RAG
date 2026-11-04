"""Plot the completed sufficient/insufficient response comparison from metrics.csv."""
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=Path(__file__).resolve().parents[2] / 'results/rag_behavior/musique_valid_no_abstain_hint')
    args = parser.parse_args()
    with (args.results_dir / 'metrics.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    names = {'qwen38': 'Qwen3.8-27B (NF4)', 'llama31': 'Llama-3.1-8B-Instruct', 'mistral': 'Mistral-7B-Instruct-v0.3'}
    lookup = {(r['model'], r['context_group']): r for r in rows}
    for name in names:
        for group in ['sufficient', 'insufficient']:
            row = lookup.get((name, group))
            if not row or row['complete'] != 'True' or int(row['total']) == 0:
                raise ValueError('All three models must finish both non-empty groups before plotting.')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharey=True)
    categories = ['correct', 'abstain', 'hallucinate']
    colors = ['#2ca25f', '#3182bd', '#de2d26']
    for i, group in enumerate(['sufficient', 'insufficient']):
        for j, (name, title) in enumerate(names.items()):
            ax = axes[i, j]
            row = lookup[name, group]
            values = [float(row[c + '_rate']) for c in categories]
            ax.bar(['Correct', 'Abstain', 'Hallucinate'], values, color=colors)
            for x, value in enumerate(values):
                ax.text(x, min(value + .025, 1.03), f'{value:.1%}', ha='center', fontsize=9)
            ax.set_ylim(0, 1.1)
            ax.set_yticks([0, .2, .4, .6, .8, 1])
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.set_title(title if i == 0 else '')
            if j == 0:
                ax.set_ylabel(f'{group.capitalize()} context\n(n={row["total"]})')
    fig.suptitle('Categorizing RAG Responses: Sufficient vs Insufficient Context')
    fig.tight_layout()
    fig.savefig(args.results_dir / 'response_categories.png', dpi=200)
    fig.savefig(args.results_dir / 'response_categories.pdf')
    print(args.results_dir / 'response_categories.png')


if __name__ == '__main__':
    main()
