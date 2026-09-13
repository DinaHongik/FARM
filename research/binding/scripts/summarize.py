"""Score completed frozen outputs and produce table/CSV with family intervals."""
import argparse
from collections import Counter
import csv
import random
from common import ROOT, read, rows, save, sha, digest
from evaluate import score

ARMS=['single_agent','farm_feedback','farm_no_feedback']


def aggregate(records):
    keys=['fields','correct_fields','binding_tp','binding_fp','binding_fn','static_fields','correct_static_fields',
          'required_fields','required_with_value','required_with_correct_value','resolvable_required_fields',
          'resolvable_required_with_value','expected_missing_fields','correct_missing_fields','unnecessary_abstentions']
    result={k:sum(r[k] for r in records) for k in keys}
    result.update(n=len(records),whole_correct=sum(r['whole_correct'] for r in records),
                  binding_cases=sum(r['binding_applicable'] for r in records),
                  exact_binding_cases=sum(r['exact_binding_set'] for r in records),
                  fully_specified_cases=sum(r['fully_specified_reference'] for r in records),
                  materialization_passes=sum(r['materialization_pass'] for r in records),
                  declared_schema_valid_fully_bound=sum(r['declared_schema_valid_fully_bound'] for r in records))
    def ratio(a,b):return a/b if b else None
    tp,fp,fn=[result['binding_'+k] for k in ['tp','fp','fn']]
    result.update(whole_configuration_accuracy=ratio(result['whole_correct'],result['n']),
                  field_accuracy=ratio(result['correct_fields'],result['fields']),
                  binding_precision=ratio(tp,tp+fp),binding_recall=ratio(tp,tp+fn),binding_f1=ratio(2*tp,2*tp+fp+fn),
                  exact_binding_accuracy=ratio(result['exact_binding_cases'],result['binding_cases']),
                  static_value_accuracy=ratio(result['correct_static_fields'],result['static_fields']),
                  required_field_coverage=ratio(result['required_with_value'],result['required_fields']),
                  correct_required_field_coverage=ratio(result['required_with_correct_value'],result['required_fields']),
                  resolvable_required_field_coverage=ratio(result['resolvable_required_with_value'],result['resolvable_required_fields']),
                  missing_input_accuracy=ratio(result['correct_missing_fields'],result['expected_missing_fields']),
                  declared_schema_valid_rate=ratio(result['declared_schema_valid_fully_bound'],result['n']),
                  argument_materialization_pass_rate=ratio(result['materialization_passes'],result['fully_specified_cases']))
    result['errors']=dict(Counter(e.split(':')[1] if ':' in e else e for r in records for e in r['errors']))
    return result


def paired(a,b):
    families={r['family_id'] for r in a};deltas=[]
    for family in sorted(families):
        aa=[r for r in a if r['family_id']==family];bb=[r for r in b if r['family_id']==family]
        deltas.append((sum(r['whole_correct'] for r in aa)-sum(r['whole_correct'] for r in bb),len(aa)))
    rng=random.Random(20260906);samples=[]
    for _ in range(10000):
        picked=rng.choices(deltas,k=len(deltas));samples.append(100*sum(d[0] for d in picked)/sum(d[1] for d in picked))
    samples.sort()
    def quantile(p):
        k=(len(samples)-1)*p;i=int(k)
        return samples[i]+(samples[min(i+1,len(samples)-1)]-samples[i])*(k-i)
    return {'whole_accuracy_difference_pp':100*sum(d[0] for d in deltas)/sum(d[1] for d in deltas),
            'family_bootstrap_95_pp':[quantile(.025),quantile(.975)],'clusters':len(families),'resamples':10000}


def summarize(directory,smoke=False):
    prefix='smoke_' if smoke else ''
    cases=rows(ROOT/f'private/{prefix}inputs.jsonl')
    gold={g['case_id']:g for g in read(ROOT/f'private/{prefix}references.json')}
    manifest=read(directory/'manifest.json')
    assert manifest['inputs_sha256']==sha(ROOT/f'private/{prefix}inputs.jsonl')
    assert manifest['reference_sha256']==sha(ROOT/f'private/{prefix}references.json')
    for name,value in manifest['scripts_sha256'].items():assert sha(ROOT/'scripts'/name)==value
    assert manifest['checker_sha256']==sha(ROOT/'reference_code/configuration.py')
    assert len(list((directory/'records').glob('*/*.json')))==len(cases)*3
    scores={};initial={};result={'status':'complete','n':len(cases),'mode':'known_by_construction',
        'condition':'controlled_requests_supplied_endpoints','real_platform_execution':None,
        'model':manifest['model'],'reference_sha256':manifest['reference_sha256'],
        'max_generated_tokens_per_call':manifest['max_generated_tokens_per_call'], 'thinking':manifest['thinking'],
        'output_completion':read(ROOT/'results'/('SMOKE_OUTPUT_VALIDATION.json' if smoke else 'OUTPUT_VALIDATION.json')),
        'arms':{},'paired':{}}
    record_hashes={}
    for arm in ARMS:
        scores[arm]=[];initial[arm]=[];calls=[]
        for case in cases:
            path=directory/'records'/arm/(digest(case['case_id'])+'.json');record=read(path)
            assert record['case_id']==case['case_id'] and record['arm']==arm
            assert gold[case['case_id']]['input_sha256']==digest(case)
            scores[arm].append(score(case,gold[case['case_id']],record['final']))
            initial[arm].append(score(case,gold[case['case_id']],record['initial']))
            calls.extend(record['calls']);record_hashes[str(path.relative_to(directory))]=sha(path)
        summary=aggregate(scores[arm]);summary['by_category']={cat:aggregate([r for r in scores[arm] if r['category']==cat]) for cat in sorted({r['category'] for r in scores[arm]})}
        summary['repair']={'rescues':sum(not b['whole_correct'] and a['whole_correct'] for b,a in zip(initial[arm],scores[arm])),
                           'regressions':sum(b['whole_correct'] and not a['whole_correct'] for b,a in zip(initial[arm],scores[arm]))}
        summary['resources']={'logical_calls':len(calls),'mean_calls':len(calls)/len(cases),
                              'prompt_tokens':sum(c.get('prompt_tokens',0) for c in calls),
                              'completion_tokens':sum(c.get('completion_tokens',0) for c in calls),
                              'provider_seconds':sum(c.get('provider_seconds',0) for c in calls)}
        result['arms'][arm]=summary
    for other in ['single_agent','farm_no_feedback']:result['paired']['farm_feedback_vs_'+other]=paired(scores['farm_feedback'],scores[other])
    stem='SMOKE_RESULTS' if smoke else 'BINDING_150_RESULTS'
    save(ROOT/'results'/(stem+'.json'),result)
    save(ROOT/'private'/(stem+'_case_scores.json'),{'final':scores,'initial':initial})
    save(ROOT/'private'/(stem+'_prediction_hashes.json'),record_hashes)
    columns=['whole_configuration_accuracy','binding_f1','static_value_accuracy','required_field_coverage',
             'missing_input_accuracy','argument_materialization_pass_rate']
    with (ROOT/'results'/(stem+'.csv')).open('w') as f:
        w=csv.writer(f);w.writerow(['workflow',*columns])
        for arm in ARMS:w.writerow([arm,*[result['arms'][arm][k] for k in columns]])
    def pct(v):return '--' if v is None else f'{100*v:.2f}'
    labels=['Strong single agent','FARM with repair','FARM without repair']
    lines=['% Controlled constructed requests; correct endpoints supplied. Percentages.',
           '% Materialization uses 120 fully specified cases, not live IFTTT execution.',
           r'\begin{tabular}{lrrrrrr}',r'\toprule',
           r'Workflow & Whole & Binding F1 & Static & Coverage & Missing & Materialization \\',r'\midrule']
    for label,arm in zip(labels,ARMS):lines.append(label+' & '+' & '.join(pct(result['arms'][arm][k]) for k in columns)+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    (ROOT/'results'/(stem+'.tex')).write_text('\n'.join(lines)+'\n')
    print({a:{k:result['arms'][a][k] for k in columns} for a in ARMS})
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=__import__('pathlib').Path,default=ROOT/'private/full_run');p.add_argument('--smoke',action='store_true')
    a=p.parse_args();summarize(a.directory,a.smoke)
