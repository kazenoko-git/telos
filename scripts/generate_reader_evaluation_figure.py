#!/usr/bin/env python3
"""Render saved Reader-1 diagnostics in the Télos research figure palette."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=root / 'reports/xerj_reader_2026-11-14/extended-run.json')
    parser.add_argument('--output', type=Path, default=root / 'figures/xerj_reader_evaluation_2026-11-14.png')
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    rows = data['results']
    groups = defaultdict(list)
    for row in rows:
        groups[row['category']].append(row)
    total = len(rows)
    passed = sum(r['grade']['answer_correct'] for r in rows)
    replayed = [r for r in rows if 'replay_matches' in r]
    high_errors = sum(not r['grade']['answer_correct'] and r['response']['confidence'] >= .9 for r in rows)
    ink, muted, page, card = '#2D2A26', '#5C554D', '#FDF9F5', '#F4EBE1'
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Outfit', 'Avenir Next', 'Helvetica Neue', 'Arial'], 'text.color': ink})
    fig = plt.figure(figsize=(16, 9), dpi=220, facecolor=page)
    bg = fig.add_axes([0, 0, 1, 1]);bg.axis('off')
    bg.add_patch(FancyBboxPatch((.04,.12),.57,.68, boxstyle='round,pad=0.012,rounding_size=0.015',facecolor=card,edgecolor='#E8DFD5'))
    fig.text(.045,.925,'TÉLOS',fontsize=27,weight='bold')
    fig.text(.18,.928,'READER-1 / DOCUMENT EXTRACTION',fontsize=13,weight='medium')
    fig.text(.18,.894,'A closer look at what the program gets right.',fontsize=12,color=muted)
    fig.text(.96,.925,'EVALUATION NOTEBOOK · 14 NOV 2026',fontsize=9,ha='right',color=muted)
    ax = fig.add_axes([.19,.18,.38,.56],facecolor=card)
    labels = [name.replace('_',' ').title() for name in groups]
    values = [100*sum(r['grade']['answer_correct'] for r in group)/len(group) for group in groups.values()]
    colors = ['#23765D' if v>=80 else '#9B5B3E' if v>=50 else '#A33F36' for v in values]
    ax.barh(range(len(values)),values,color=colors,height=.56,zorder=3)
    for i,(value,group) in enumerate(zip(values,groups.values())):
        correct=sum(r['grade']['answer_correct']for r in group)
        ax.text(min(value+2,103),i,f'{correct}/{len(group)}',va='center',fontsize=10,color=ink)
    ax.set_yticks(range(len(values)),labels,fontsize=11,color=muted);ax.invert_yaxis()
    ax.set_xlim(0,115);ax.set_xticks([0,25,50,75,100],['0','25','50','75','100%'],fontsize=9,color=muted)
    ax.grid(axis='x',color='#DDD0C0',linestyle=(0,(3,4)),zorder=0)
    ax.tick_params(axis='both',length=0,pad=9)
    for spine in ax.spines.values():spine.set_visible(False)
    fig.text(.065,.765,'Correct answers by diagnostic category',fontsize=15,weight='medium')
    fig.text(.66,.745,f'{100*passed/total:.1f}%',fontsize=57,weight='medium')
    fig.text(.66,.695,f'{passed} correct / {total} synthetic cases',fontsize=16)
    fig.text(.66,.64,'Semantic selection is the limiting factor.',fontsize=13,color=muted)
    fig.text(.66,.53,f'{sum(r["replay_matches"]for r in replayed)} / {len(replayed)}',fontsize=28)
    fig.text(.66,.49,'Sampled formulas replayed identically',fontsize=12,color=muted)
    fig.text(.66,.385,str(high_errors),fontsize=28)
    fig.text(.66,.345,'Incorrect answers with confidence ≥ 0.90',fontsize=12,color=muted)
    fig.text(.66,.225,'Exact execution does not guarantee\nthe right evidence or operation.',fontsize=16,linespacing=1.6)
    fig.text(.045,.065,'Synthetic, template-based diagnostics selected after a pilot. Not a representative benchmark or production accuracy estimate.',fontsize=10,color=muted)
    fig.text(.045,.035,'Source: extended-run.json · Gold answers fixed before execution · Replays sampled every fourth case · No private documents submitted',fontsize=9,color=muted)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,facecolor=page,dpi=220)
    plt.close(fig)
    print(args.output)


if __name__=='__main__':
    main()
