#!/usr/bin/env python3
"""Aggregate-only audit of actual candidate construction and completed predictions."""
from collections import Counter,defaultdict
import json,random,statistics,sys
from pathlib import Path
ROOT=Path('/raid/session/aicontents/farm/experiments/20260905-five-training-techniques')
sys.path.insert(0,str(ROOT))
from prepare import SIDES,read,write


def main():
    d=ROOT/'derived'/'function';groups=read(d/'train.json');corpus=read(d/'corpus.json')
    order=list(range(len(groups)));random.Random(43).shuffle(order)
    byquery=defaultdict(list)
    for group in groups:byquery[' '.join(group['query'].casefold().split())].append(group)
    duplicated=[values for values in byquery.values() if len(values)>1]
    report={'exact_normalized_query_duplicate_groups':sum(len(v) for v in duplicated),
            'duplicate_queries':len(duplicated),'sides':{}}
    for side_index,side in enumerate(SIDES):
        labels=[{p[side_index] for p in g['valid_pairs']} for g in groups]
        pools=read(d/('negatives_'+side+'.json'));totals={x:Counter() for x in ('global','within')}
        jaccards=[];same_services={i:{corpus[side][p]['service'] for p in label} for i,label in enumerate(labels)}
        for start in range(0,len(order),16):
            batch=order[start:start+16];sets={}
            for strategy in ('global','within'):
                candidates=set().union(*(labels[i]|set(pools[i][strategy]) for i in batch));sets[strategy]=candidates
                for idx in batch:
                    positive_text={corpus[side][p]['text'] for p in labels[idx]}
                    excluded=set().union(*(set(corpus[side][p]['equivalent_indices']) for p in labels[idx]))
                    negatives={n for n in candidates if n not in excluded and corpus[side][n]['text'] not in positive_text}
                    sibling={n for n in negatives if corpus[side][n]['service'] in same_services[idx]}
                    totals[strategy]['queries']+=1;totals[strategy]['negative_exposures']+=len(negatives)
                    totals[strategy]['same_service_negative_exposures']+=len(sibling)
                    totals[strategy]['queries_with_a_sibling']+=bool(sibling)
                    totals[strategy]['candidate_documents']+=len(candidates)
            jaccards.append(len(sets['global']&sets['within'])/len(sets['global']|sets['within']))
        report['sides'][side]={
          'groups_with_multiple_positive_labels':sum(len(x)>1 for x in labels),
          'curriculum_batch_candidate_jaccard_mean':statistics.mean(jaccards),
          'policies':{policy:{**value,'mean_negatives_per_query':value['negative_exposures']/value['queries'],
                     'mean_sibling_negatives_per_query':value['same_service_negative_exposures']/value['queries'],
                     'sibling_fraction_of_negatives':value['same_service_negative_exposures']/value['negative_exposures']}
                     for policy,value in totals.items()}}
    reports={arm:read(ROOT/'arms'/arm/'results_dev.json') for arm in ('s1','f1','f2','f3','f4')}
    f3={r['group_id']:r for r in reports['f3']['per_sample']}
    s1={r['group_id']:r for r in reports['s1']['per_sample']}
    maps={side:{row['id']:row for row in corpus[side]} for side in SIDES}
    service_names=read(ROOT/'derived'/'service_only'/'corpus.json')
    def svc(side,url):return service_names[side][maps[side][url]['service']]['id']
    stats=Counter()
    for gid,row in f3.items():
        valid={tuple(p) for p in row['valid_pairs']};pred=tuple(row['prediction'])
        hit=pred in valid
        valid_services={(svc('trigger',t),svc('action',a)) for t,a in valid}
        pred_services=(svc('trigger',pred[0]),svc('action',pred[1]))
        stats['rows']+=1;stats['function_correct']+=hit
        stats['correct_services_wrong_functions']+=not hit and pred_services in valid_services
        stats['wrong_services']+=pred_services not in valid_services
        coverage=bool(row['candidate_lattice_coverage'].get('10',row['candidate_lattice_coverage'].get(10)))
        stats['wrong_but_covered_top10']+=not hit and coverage
        stats['gold_pair_missing_top10']+=not coverage
        service_pred=s1[gid]['prediction'];conditioned=[]
        for side_index,side in enumerate(SIDES):
            matching=[url for url in row['top_'+side+'_ids'] if svc(side,url)==service_pred[side_index]]
            conditioned.append(matching[0] if matching else None)
        available=all(conditioned)
        actual=tuple(conditioned) if available else pred
        candidate_hit=actual in valid
        stats['s1_conditioning_candidates_available']+=available
        stats['s1_conditioning_correct_with_f3_fallback']+=candidate_hit
        stats['s1_conditioning_rescues']+=candidate_hit and not hit
        stats['s1_conditioning_regressions']+=hit and not candidate_hit
        oracle=False
        for st,sa in valid_services:
            t=next((url for url in row['top_trigger_ids'] if svc('trigger',url)==st),None)
            a=next((url for url in row['top_action_ids'] if svc('action',url)==sa),None)
            oracle|=(t,a) in valid
        stats['oracle_service_first_matching_function_pair_correct']+=oracle
    report['f3_error_diagnostics']=dict(stats)
    report['oracle_policy']='diagnostic only: gold services select first matching candidates, never a system result'
    report['s1_conditioning_policy']='exploratory fixed top-1 service filtering over F3 top10; fallback to original F3 if missing'
    logs={}
    for arm in ('s1','f1','f2','f3','f4'):
        rows=[]
        for filename in (arm+'.smoke.log',arm+'.log'):
            for line in (ROOT/filename).read_text().splitlines():
                if not line.startswith('{'):continue
                try:row=json.loads(line)
                except ValueError:continue
                if row.get('status')=='training':rows.append(row)
        logs[arm]={}
        for phase in sorted({r['phase'] for r in rows}):
            logs[arm][phase]={}
            for epoch in range(3):
                part=[r for r in rows if r['phase']==phase and epoch*502<r['step']<=(epoch+1)*502]
                logs[arm][phase][str(epoch+1)]={
                    'logged_steps':len(part),'mean_loss':statistics.mean(r['loss'] for r in part),
                    'mean_last20_loss':statistics.mean(r['loss'] for r in part[-20:]),
                    'gradient_norm_median':statistics.median(r['gradient_norm'] for r in part)}
    report['sampled_training_logs']=logs
    write(Path(__file__).with_name('CANDIDATE_AND_ERROR_AUDIT.json'),report)
    print(json.dumps({k:v for k,v in report.items() if k!='sampled_training_logs'},indent=2))


if __name__=='__main__':main()
