#!/usr/bin/env python3
"""Detached CPU collector: refresh paired references and write final review tables."""
import os
from pathlib import Path
import time
from prepare import read,write

ROOT=Path(__file__).resolve().parent
ARMS=('s1','f1','f2','f3','f4')


def collect():
    rows=[];all_terminal=True
    for arm in ARMS:
        run=ROOT/'arms'/arm
        progress=read(run/'progress.json') if (run/'progress.json').exists() else {'status':'starting'}
        result=read(run/'results_dev.json') if (run/'results_dev.json').exists() else None
        if result:
            from evaluate import paired_comparisons
            view='service_only' if arm=='s1' else 'function'
            groups=read(ROOT/'derived'/view/'dev.json')
            result['paired_comparisons']=paired_comparisons(run,groups,[r['hits']['R@1'] for r in result['per_sample']])
            write(run/'results_dev.json',result)
            write(run/'summary.json',{k:v for k,v in result.items() if k!='per_sample'})
            rows.append({'arm':arm,'status':'complete','task':result['task_level'],
                         'correct':result['metrics']['correct'],'rows':result['rows'],
                         'accuracy':result['metrics']['exact_pair_top1'],
                         'paired_comparisons':result['paired_comparisons']})
        elif (run/'failure.json').exists():
            rows.append({'arm':arm,**read(run/'failure.json')})
        else:
            all_terminal=False
            rows.append({'arm':arm,**progress})
    write(ROOT/'STATUS.json',{'updated_at_unix':time.time(),'jobs':rows,'all_terminal':all_terminal})
    lines=['# FARM five-method training status','',
           'These are development results. Service and function accuracies measure different tasks.','',
           '| Arm | Status | Task | Correct / requests | Accuracy |',
           '|---|---|---|---|---|']
    for row in rows:
        lines.append('| {} | {} | {} | {} | {} |'.format(row['arm'],row['status'],row.get('task','—'),
          f"{row['correct']} / {row['rows']}" if 'correct' in row else '—',
          f"{100*row['accuracy']:.2f}%" if 'accuracy' in row else '—'))
    if all_terminal:
        lines+=['','All jobs reached a terminal status; inspect failed jobs separately.',
                'Full metrics, strata and paired comparisons are in each arm’s summary.json.']
    temp=ROOT/'RESULTS.md.tmp';temp.write_text('\n'.join(lines)+'\n');os.replace(temp,ROOT/'RESULTS.md')
    return all_terminal


if __name__=='__main__':
    os.umask(0o077)
    while True:
        if collect():break
        time.sleep(60)
