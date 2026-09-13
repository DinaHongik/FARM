"""Reevaluate five original checkpoints with their stored runtime window, no training."""
import fcntl,importlib.util,json,os,subprocess,time
from pathlib import Path
import torch
import numpy as np
from prepare import read,write,sha,SIDES
from evaluate import load as preserved_load
from evaluate import encoded,paired_interval


def corrected_baselines(root,corrected):
    """Same source documents and exact annotated-pair metric as E0/E3."""
    original=root.parent/'20260831T171222Z-farm-four-pipelines'
    reports={}
    for name,view in [('e0','service_only'),('e3','function')]:
        path=corrected/'results'/f'{name}_dev.json'
        if path.exists():reports[name]=read(path)['metrics'];continue
        groups=read(root/'derived'/view/'dev.json')
        canonical=read(root/'derived'/view/'corpus.json')
        predictions={};windows={}
        for side in SIDES:
            model_path=original/'checkpoints'/name/side/'final'
            model=preserved_load(model_path)
            windows[side]=model[0].auto_model.config.sliding_window
            if name=='e0':
                corpus=read(original/'derived_data'/'e0_service'/f'corpus_{side}.json')
                ids=np.array([r['service_id'] for r in corpus]);texts=[r['text'] for r in corpus]
            else:
                corpus=canonical[side];ids=np.array([r['id'] for r in corpus]);texts=[r['text'] for r in corpus]
            d=model.encode_document(texts,batch_size=64,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
            q=model.encode_query([g['query'] for g in groups],batch_size=64,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
            assert np.isfinite(q).all() and np.isfinite(d).all()
            predictions[side]=[str(ids[np.lexsort((ids,-row))[0]]) for row in q@d.T]
            del model
            import gc
            gc.collect();torch.cuda.empty_cache()
        records=[]
        for i,g in enumerate(groups):
            valid={(canonical['trigger'][t]['id'],canonical['action'][a]['id']) for t,a in g['valid_pairs']}
            predicted=[predictions[s][i] for s in SIDES]
            records.append(dict(group_id=g['group_id'],family_id=g['family_id'],prediction=predicted,
                                hits={'R@1':int(tuple(predicted) in valid)}))
        correct=sum(r['hits']['R@1'] for r in records)
        metrics={'correct':correct,'exact_pair_top1':correct/len(groups)}
        write(path,dict(status='complete',rows=len(groups),metrics=metrics,per_sample=records,
            stored_runtime_windows=windows,evaluation_batch_size=64,scope='preserved-window dev reevaluation; unchanged weights and source views',
            original_result_sha256=sha(original/'results'/f'{name}_dev.json')))
        reports[name]=metrics
    return reports


def comparisons(corrected,arm,groups,hits):
    names={'e0_service':corrected/'results'/'e0_dev.json'} if arm=='s1' else {'e3_function':corrected/'results'/'e3_dev.json'}
    if arm in ('f2','f3','f4'):names['f1_function']=corrected/'arms'/'f1'/'results_dev.json'
    result={}
    for name,path in names.items():
        prior={r['group_id']:r['hits']['R@1'] for r in read(path)['per_sample']}
        assert set(prior)=={g['group_id'] for g in groups}
        old=[prior[g['group_id']] for g in groups]
        result[name]=dict(status='complete',reference_sha256=sha(path),reference_window_policy='stored runtime window preserved',
             rescues=sum(n and not o for n,o in zip(hits,old)),regressions=sum(o and not n for n,o in zip(hits,old)),
             **paired_interval(np.array(hits)-np.array(old),[g['family_id'] for g in groups]))
    return result


def main():
    os.umask(0o077);torch.set_num_threads(4)
    root=Path(__file__).resolve().parent;project=root.parents[1]
    s1=root/'arms'/'s1_seed1337'
    while not any((s1/n).exists() for n in ('summary.json','failure.json')):
        time.sleep(10)
    lock=open(project/'.farm-gpu-7.lock','w');fcntl.flock(lock,fcntl.LOCK_EX)
    old=root.parent/'20260905-five-training-techniques'
    corrected=root.parent/'20260905-previous-window-corrected';corrected.mkdir(exist_ok=True)
    spec=importlib.util.spec_from_file_location('legacy_window_evaluator',old/'evaluate.py')
    legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    legacy.load=preserved_load
    baseline_reports=corrected_baselines(root,corrected)
    legacy.paired_comparisons=lambda run,groups,hits:comparisons(corrected,run.name,groups,hits)
    reports={}
    for arm in ('s1','f1','f2','f3','f4'):
        run=corrected/'arms'/arm;run.mkdir(parents=True,exist_ok=True)
        if (run/'summary.json').exists():
            reports[arm]=read(run/'summary.json')['metrics'];continue
        phases=('shared',) if arm=='s1' else ('joint',) if arm=='f4' else SIDES
        windows={}
        for phase in phases:
            (run/phase).mkdir(exist_ok=True)
            destination=run/phase/'final'
            if not destination.exists():destination.symlink_to(old/'arms'/arm/phase/'final',target_is_directory=True)
            roles=('shared',) if phase=='shared' else SIDES if phase=='joint' else (phase,)
            for role in roles:windows[role]=read(destination/role/'config.json')['sliding_window']
        cfg=read(old/'arms'/arm/'config.json')
        write(run/'CORRECTION.json',dict(status='evaluating',change='preserve saved runtime attention window',
              old_summary_sha256=sha(old/'arms'/arm/'summary.json'),stored_windows=windows,
              weights_changed=False,scope='previously exposed development; E0/E3 reevaluated with preserved runtime window'))
        legacy.evaluate(run,old/'derived',cfg)
        # Do not present unverified historical references as an isolated method comparison.
        for name in ('summary.json','results_dev.json'):
            result=read(run/name)
            result['evaluation_correction']='saved runtime sliding window preserved'
            write(run/name,result)
        reports[arm]=read(run/'summary.json')['metrics']
    write(corrected/'COLLECTED.json',dict(status='complete',completed_at_unix=time.time(),
          models=reports,baselines=baseline_reports,weights_changed=False,evaluation_only=True))
    write(root/'PREVIOUS_WINDOW_CORRECTION.json',dict(status='complete',directory=str(corrected),models=reports,baselines=baseline_reports))
    rows=[f"| {arm} | {m['correct']}/1145 | {100*m['exact_pair_top1']:.2f}% |" for arm,m in reports.items()]
    (corrected/'RESULTS.md').write_text('# Original models, preserved-window reevaluation\n\nEvaluation correction only; no retraining. Service and function tasks differ. E0/E3 references were reevaluated with the same window-preservation rule.\n\n| Arm | Correct | Accuracy |\n|---|---|---|\n'+'\n'.join(rows)+'\n')


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        write(Path(__file__).with_name('PREVIOUS_WINDOW_CORRECTION.json'),dict(status='failed',error=repr(exc),updated_at_unix=time.time()))
        raise
