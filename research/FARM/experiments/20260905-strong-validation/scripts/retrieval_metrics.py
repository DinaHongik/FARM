"""Pure scoring for ordered function pairs and paired family uncertainty."""
from collections import Counter
import math
import numpy as np

def score_rankings(groups, corpus, rankings):
    records=[]
    for i,g in enumerate(groups):
        valid={tuple(p) for p in g['valid_pairs']}
        t,a=rankings['trigger'][i].tolist(),rankings['action'][i].tolist()
        prediction=(int(t[0]),int(a[0]))
        correct=int(prediction in valid)
        services={(corpus['trigger'][x]['service'],corpus['action'][y]['service']) for x,y in valid}
        projected=int((corpus['trigger'][prediction[0]]['service'],corpus['action'][prediction[1]]['service']) in services)
        records.append({'group_id':g['group_id'],'family_id':g['family_id'],'correct':correct,
                        'projected_service_correct':projected,
                        'coverage':{str(k):int(any(x in t[:k] and y in a[:k] for x,y in valid)) for k in [1,5,10]},
                        'prediction':[corpus[s][prediction[j]]['id'] for j,s in enumerate(['trigger','action'])]})
    n=len(records)
    assert n and len({r['group_id'] for r in records})==n
    correct=sum(r['correct'] for r in records)
    projected=sum(r['projected_service_correct'] for r in records)
    return {'rows':n,'correct':correct,'exact_pair_accuracy':correct/n,'projected_service_correct':projected,
            'projected_service_accuracy':projected/n,'wrong_service':n-projected,
            'correct_service_wrong_function':projected-correct,
            'candidate_product_coverage':{str(k):sum(r['coverage'][str(k)] for r in records)/n for k in [1,5,10]}},records

def paired_comparison(fixed,refreshed,samples=10000,seed=20260905):
    left={r['group_id']:r for r in fixed}
    right={r['group_id']:r for r in refreshed}
    assert left.keys()==right.keys()
    identifiers=sorted(left)
    assert all(left[k]['family_id']==right[k]['family_id'] for k in identifiers)
    families=sorted({left[k]['family_id'] for k in identifiers})
    index={k:i for i,k in enumerate(families)}
    sums=np.zeros(len(families));counts=np.zeros(len(families))
    rescues=regressions=0
    for k in identifiers:
        old,new=left[k]['correct'],right[k]['correct']
        rescues+=int(new and not old);regressions+=int(old and not new)
        j=index[left[k]['family_id']];sums[j]+=new-old;counts[j]+=1
    rng=np.random.default_rng(seed);boot=[]
    for _ in range(samples):
        j=rng.integers(0,len(families),size=len(families))
        boot.append(float(sums[j].sum()/counts[j].sum()))
    discordant=rescues+regressions
    pvalue=1.0 if not discordant else min(1.,2*math.ldexp(float(sum(math.comb(discordant,j) for j in range(min(rescues,regressions)+1))),-discordant))
    return {'rows':len(identifiers),'families':len(families),'rescues':rescues,'regressions':regressions,
            'difference_percentage_points':100*(rescues-regressions)/len(identifiers),
            'paired_family_bootstrap_95_percentage_points':(100*np.quantile(boot,[.025,.975])).tolist(),
            'bootstrap_samples':samples,'bootstrap_seed':seed,'exact_mcnemar_two_sided_p':pvalue,
            'mcnemar_scope':'case-level discordances; family dependence is handled by the separate bootstrap'}
