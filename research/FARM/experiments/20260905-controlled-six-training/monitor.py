"""Detached progress/paired-result collection; never changes training weights."""
import json,subprocess,time
from pathlib import Path
from experiment import RUNS,configuration
from prepare import read,write,sha,SIDES


def first_epoch_comparison(root,seed):
    name=f'FIRST_EPOCH_MATCH_seed{seed}.json';out=root/name
    if out.exists():return
    a=root/'arms'/f'f3_fixed_seed{seed}'
    b=root/'arms'/f'f3_refresh_seed{seed}'
    if not all((p/s/'epoch-01'/'complete.json').exists() for p in (a,b) for s in SIDES):return
    from boundary_probe import compare
    checks={}
    try:
        for side in SIDES:
            checks[side]=compare(a/side/'epoch-01'/side,b/side/'epoch-01'/side)
        write(out,dict(status='passed',seed=seed,checks=checks,scope='full first epoch before refresh affects updates'))
    except Exception as exc:
        write(out,dict(status='failed',seed=seed,error=repr(exc),action='inspect before interpreting matched treatment'))


def main():
    root=Path(__file__).resolve().parent
    while True:
        states={};terminal=0
        for run_id in RUNS:
            run=root/'arms'/run_id
            if (run/'summary.json').exists():
                result=read(run/'summary.json');states[run_id]=dict(status='complete',metrics=result['metrics']);terminal+=1
            elif (run/'failure.json').exists():
                states[run_id]=read(run/'failure.json');terminal+=1
            else:
                session='farm-six-'+run_id
                alive=subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==0
                states[run_id]=read(run/'progress.json') if (run/'progress.json').exists() else dict(status='starting')
                if not alive:states[run_id]=dict(status='process_exited_without_result');terminal+=1
        for seed in (42,1337):first_epoch_comparison(root,seed)
        write(root/'STATUS.json',dict(updated_at_unix=time.time(),terminal_jobs=terminal,total_jobs=6,runs=states))
        if terminal==6:break
        time.sleep(30)
    from evaluate import paired_comparisons,paired_interval
    import numpy as np
    rows=[]
    for run_id in RUNS:
        run=root/'arms'/run_id
        if (run/'results_dev.json').exists():
            result=read(run/'results_dev.json');cfg=configuration(run_id)
            groups=read(root/'derived'/('service_only' if cfg['arm']=='s1' else 'function')/'dev.json')
            result['paired_comparisons']=paired_comparisons(run,groups,[r['hits']['R@1'] for r in result['per_sample']],cfg)
            corrected=root.parent/'20260905-previous-window-corrected'
            references={'e0_service':corrected/'results'/'e0_dev.json',
                        'e3_function':corrected/'results'/'e3_dev.json',
                        'previous_s1':corrected/'arms'/'s1'/'results_dev.json',
                        'previous_f3':corrected/'arms'/'f3'/'results_dev.json'}
            for label in list(result['paired_comparisons']):
                if label=='matched_fixed_control':continue
                path=references[label]
                if path.exists():
                    reference={r['group_id']:r['hits']['R@1'] for r in read(path)['per_sample']}
                    assert set(reference)=={g['group_id'] for g in groups}
                    old=[reference[g['group_id']] for g in groups]
                    hits=[r['hits']['R@1'] for r in result['per_sample']]
                    result['paired_comparisons'][label]=dict(status='complete',reference_sha256=sha(path),
                        reference_window_policy='stored runtime window preserved',
                        rescues=sum(n and not o for n,o in zip(hits,old)),regressions=sum(o and not n for n,o in zip(hits,old)),
                        **paired_interval(np.array(hits)-np.array(old),[g['family_id'] for g in groups]))
                else:
                    result['paired_comparisons'][label]['reference_window_policy']='historical reload behavior unverified; descriptive comparison only'
            write(run/'results_dev.json',result)
            write(run/'summary.json',{k:v for k,v in result.items() if k!='per_sample'})
            metric=result['metrics'];rows.append(f"| {run_id} | complete | {result['task_level']} | {metric['correct']}/{result['rows']} | {100*metric['exact_pair_top1']:.2f}% |")
        else:rows.append(f"| {run_id} | {states[run_id]['status']} | — | — | — |")
    (root/'RESULTS.md').write_text('# Controlled six-run results\n\nPreviously exposed development data. Service and function tasks are distinct.\n\n| Run | Status | Task | Correct | Accuracy |\n|---|---|---|---|---|\n'+'\n'.join(rows)+'\n')
    write(root/'COLLECTED.json',dict(collected_at_unix=time.time(),all_terminal=True,
              all_successful=all(v['status']=='complete' for v in states.values()),runs=states))


if __name__=='__main__':main()
