"""Draw measured aggregate outcomes only; no placeholder scores."""
from common import ROOT,read


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results=read(ROOT/'results/BINDING_150_RESULTS.json')
    assert results['status']=='complete' and results['n']==150
    metrics=[('whole_configuration_accuracy','Whole configuration\n150 requests'),
             ('binding_f1','Ingredient-binding F1'),
             ('argument_materialization_pass_rate','Argument materialization\n120 fully specified requests')]
    arms=['single_agent','farm_feedback','farm_no_feedback'];labels=['Strong single agent','FARM with repair','FARM without repair']
    colors=['#365F91','#178577','#B06C35']
    fig,ax=plt.subplots(figsize=(8.7,4.4),layout='constrained')
    for j,(arm,label,color) in enumerate(zip(arms,labels,colors)):
        x=[k+(j-1)*.24 for k in range(len(metrics))]
        values=[100*results['arms'][arm][key] for key,_ in metrics]
        bars=ax.bar(x,values,width=.22,label=label,color=color)
        ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=8)
    ax.set_xticks(range(len(metrics)),[label for _,label in metrics]);ax.set_ylabel('Percent')
    ax.set_ylim(0,112);ax.spines[['top','right']].set_visible(False)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,1.19),ncol=3,frameon=False,fontsize=9)
    ax.set_title('Controlled binding evaluation · supplied correct endpoints',fontsize=11,pad=12)
    fig.savefig(ROOT/'results/binding_comparison.pdf',bbox_inches='tight')
    fig.savefig(ROOT/'results/binding_comparison.png',dpi=220,bbox_inches='tight')
    plt.close(fig)
    print('Saved measured binding comparison PDF and PNG.')


if __name__=='__main__':plot()
