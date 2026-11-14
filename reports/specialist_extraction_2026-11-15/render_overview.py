"""Render separate extraction/pipeline tracks in the established Télos palette."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data=json.loads((ROOT/'summary.json').read_text())['tracks']
    extraction=[r for r in data if 'direct' not in r['artifact']]
    assert len(extraction)==3 and all(r['complete'] for r in extraction),'All three extraction runs must be complete'
    direct=next(r for r in data if 'direct' in r['artifact']);assert direct['complete']
    names={'HuggingFaceTB/SmolLM2-360M-Instruct':'SmolLM2 · 360M','numind/NuExtract-1.5-tiny':'NuExtract Tiny · 0.5B','fastino/gliner2-base-v1':'GLiNER2 Base · 205M'}
    extraction.sort(key=lambda r:list(names).index(r['model_id']))
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Avenir Next','Helvetica Neue','Arial'],'text.color':'#2D2A26'})
    fig=plt.figure(figsize=(12,6),dpi=180,facecolor='#FDF9F5')
    logo=fig.add_axes([.055,.78,.17,.16]);logo.imshow(plt.imread(REPO/'figures/test_transparent_logo.png'));logo.axis('off')
    fig.text(.28,.88,'Tiny extraction models',fontsize=23,weight='medium')
    fig.text(.28,.81,'512 synthetic tasks · 15 November 2026',fontsize=11,color='#5C554D')
    columns=[.065,.40,.60,.82]
    for x,label in zip(columns,['Model','Exact extraction','Calculator pipeline','Format failures']):fig.text(x,.69,label,fontsize=11,color='#5C554D')
    for i,r in enumerate(extraction):
        y=.58-i*.12
        fig.text(columns[0],y,names[r['model_id']],fontsize=14)
        fig.text(columns[1],y,f"{r['extraction_exact']}/512",fontsize=16,color='#9B5B3E')
        fig.text(columns[2],y,f"{r['pipeline_correct']}/512",fontsize=16,color='#23765D')
        fig.text(columns[3],y,str(r['format_failures']),fontsize=16)
    notes=[
        'Extraction checks source operands; the calculator scores final status/value.',
        'Calculator operations are human-specified. Pipeline scores are separate from direct QA.',
        f"SmolLM2 direct QA: {direct['direct_correct']}/512; {direct['format_failures']} format/status failures; unconstrained decoding.",
        'Exact source matching is not semantic evidence verification. See SUMMARY.md for protocol and artifacts.'
    ]
    for i,line in enumerate(notes):fig.text(.065,.22-i*.045,line,fontsize=9,color='#5C554D')
    for ext in ['png','pdf']:fig.savefig(ROOT/f'overview.{ext}',facecolor=fig.get_facecolor())
    plt.close(fig)

if __name__=='__main__':main()
