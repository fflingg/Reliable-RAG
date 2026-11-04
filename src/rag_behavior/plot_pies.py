"""Save a paper-style RAG response pie grid as PNG."""
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=Path(__file__).resolve().parents[2] / 'results/rag_behavior/musique_valid_no_abstain_hint')
    parser.add_argument('--allow-partial', action='store_true', help='Allow plotting classified subsets of incomplete results')
    args = parser.parse_args()
    with (args.results_dir / 'metrics.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    names = [('qwen38', 'Qwen3.8-27B\n(NF4)'), ('llama31', 'Llama-3.1-8B\nInstruct'),
             ('mistral', 'Mistral-7B\nInstruct-v0.3')]
    groups = ['all', 'sufficient', 'insufficient']
    lookup = {(r['model'], r['context_group']): r for r in rows}
    selected = [lookup[name, group] for name, _ in names for group in groups]
    partial = any(int(r['unclassified']) for r in selected)
    if partial and not args.allow_partial:
        raise ValueError('Incomplete results: pass --allow-partial to plot classified answers only.')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    colors = ['#9FBAEF', '#F47760', '#58A65A']
    categories = ['abstain', 'hallucinate', 'correct']
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 9.4))
    fig.subplots_adjust(left=.04, right=.81, bottom=.15, top=.84, wspace=.18, hspace=.24)
    fig.suptitle('Categorizing Model Responses (MuSiQue Dataset)', x=.46, y=.965, fontsize=19, color='#444444')
    fig.legend(handles=[Patch(facecolor=c, label=l) for c, l in zip(colors, ['Abstain', 'Hallucinate', 'Correct'])],
               loc='upper center', bbox_to_anchor=(.44, .923), ncol=3, frameon=False, columnspacing=2)
    for i, (name, title) in enumerate(names):
        for k, group in enumerate(groups):
            row = lookup[name, group]
            values = [int(row[c]) for c in categories]
            if not sum(values):
                raise ValueError(f'No classified answers: {name}/{group}')
            axes[i, k].pie(values, colors=colors, startangle=0, counterclock=True,
                           autopct=lambda pct: f'{pct:.1f}%' if pct > .01 else '', pctdistance=.68,
                           textprops={'fontsize': 12}, wedgeprops={'linewidth': 1, 'edgecolor': 'white'})
            axes[i, k].set_aspect('equal')
        axes[i, -1].text(1.16, .5, title, transform=axes[i, -1].transAxes, ha='left', va='center', fontsize=13)
    total = int(lookup['qwen38', 'all']['total'])
    labels = ['With RAG\nAll valid contexts',
              f'With RAG\nSufficient context\n{int(lookup["qwen38", "sufficient"]["total"])/total:.1%} of dataset',
              f'With RAG\nInsufficient context\n{int(lookup["qwen38", "insufficient"]["total"])/total:.1%} of dataset']
    for k, label in enumerate(labels):
        position = axes[-1, k].get_position()
        fig.text((position.x0 + position.x1)/2, .115, label, ha='center', va='top', fontsize=12, linespacing=1.25)
    target = args.results_dir / 'response_categories_pies.png'
    fig.savefig(target, dpi=240, bbox_inches='tight', facecolor='white')
    print(target)
    plt.close(fig)


if __name__ == '__main__':
    main()
