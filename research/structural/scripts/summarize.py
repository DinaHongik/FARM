"""Aggregate frozen configuration outcomes without treating them as binding gold."""
import argparse
from pathlib import Path
import numpy as np
from common import ROOT, read, rows, save, sha
from workflows import ARMS


def metrics(record, stage='final_checks'):
    c=record[stage]
    return {'protocol_valid':bool(c['protocol_valid']), 'field_decisions_complete':bool(c['complete']),
            'no_demonstrated_provenance_violation':bool(c['protocol_valid'] and c['no_provenance_violation']),
            'complete_without_demonstrated_error':bool(c['protocol_valid'] and c['complete'] and not c['errors']),
            'locally_valid':bool(c.get('locally_valid')), 'has_unresolved_evidence':bool(c['unresolved'])}


def summarize(directory, gold_path, output):
    identity=read(directory/'manifest.json');status=read(directory/'status.json')
    if status['state']!='complete': raise RuntimeError('Only summarize the completed frozen run')
    gold={r['case_id']:r for r in rows(gold_path)}
    data={arm:{v['case_id']:v for p in (directory/'records'/arm).glob('*.json') for v in [read(p)]} for arm in ARMS}
    ids=sorted(data[ARMS[0]])
    assert all(set(data[a])==set(ids) for a in ARMS)
    assert len(ids)==identity['n']==150
    result={'status':'complete','n':150,'condition':identity['condition'], 'model':identity['model'],
            'semantic_binding_accuracy':None,'whole_configuration_semantic_accuracy':None,
            'execution_success':None,'manifest_sha256':sha(directory/'manifest.json'),'arms':{},'paired':{}}
    for arm, records in data.items():
        totals={key:sum(metrics(r)[key] for r in records.values()) for key in metrics(next(iter(records.values())))}
        proposed=sum(r['final_checks']['value_assignments'] for r in records.values())
        unsupported=sum(r['final_checks']['unsupported_assignments'] for r in records.values())
        calls=[c for r in records.values() for c in r['calls']]
        result['arms'][arm]={'numerators':totals,'denominator':150,
            'percentages':{k:100*v/150 for k,v in totals.items()},
            'value_assignments':proposed,'unsupported_assignments':unsupported,
            'unsupported_assignment_rate':unsupported/proposed if proposed else None,
            'logical_model_calls':len(calls), 'mean_calls_per_case':len(calls)/150,
            'prompt_tokens':sum(c['prompt_tokens'] for c in calls),
            'completion_tokens':sum(c['completion_tokens'] for c in calls),
            'provider_seconds':sum(c['provider_seconds'] for c in calls),
            'repairs':{key:{'rescues':sum(not metrics(r,'initial_checks')[key] and metrics(r)[key] for r in records.values()),
                             'regressions':sum(metrics(r,'initial_checks')[key] and not metrics(r)[key] for r in records.values())}
                       for key in ['field_decisions_complete','complete_without_demonstrated_error']}}
    families=sorted({gold[k]['family_id'] for k in ids});fi={f:i for i,f in enumerate(families)}
    counts=np.zeros(len(families))
    for k in ids:counts[fi[gold[k]['family_id']]]+=1
    for other in ['single_agent','farm_no_feedback']:
        comparisons={}
        for metric in ['field_decisions_complete','complete_without_demonstrated_error']:
            diff=np.array([int(metrics(data['farm_feedback'][k])[metric])-int(metrics(data[other][k])[metric]) for k in ids])
            sums=np.zeros(len(families))
            for k,v in zip(ids,diff):sums[fi[gold[k]['family_id']]]+=v
            rng=np.random.default_rng(20260906);boots=[]
            for _ in range(10000):
                j=rng.integers(0,len(families),len(families));boots.append(100*sums[j].sum()/counts[j].sum())
            comparisons[metric]={'gain_percentage_points':100*float(diff.mean()),
                                  'rescues':int(sum(diff==1)),'regressions':int(sum(diff==-1)),
                                  'family_bootstrap_95_pp':np.quantile(boots,[.025,.975]).tolist()}
        result['paired']['farm_feedback_vs_'+other]=comparisons
    save(output,result)
    print('Saved aggregate outcomes for 150 cases. Semantic binding and execution remain unmeasured.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=ROOT/'private/full_run')
    p.add_argument('--gold',type=Path,default=ROOT/'private/gold.jsonl')
    p.add_argument('--output',type=Path,default=ROOT/'results/AGENTIC_150_RESULTS.json')
    a=p.parse_args();summarize(a.run,a.gold,a.output)
