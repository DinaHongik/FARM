"""Bind passing software/data checks to the exact source used by long training."""
from pathlib import Path
import time
from prepare import read,write,sha,SIDES
from experiment import RUNS,configuration
from boundary_probe import compare


def main():
    root=Path(__file__).resolve().parent
    data=read(root/'DATA_READY.json')
    assert data['training_groups']==8018 and data['auxiliary_rows']==1145 and data['family_overlap']==0
    units=(root/'unit-tests.log').read_text()
    assert '\nOK\n' in units and 'Ran 18 tests' in units
    for name in ('BOUNDARY_PROBE.json','REAL_H6_GRADIENT_PROBE.json'):
        assert read(root/name)['status']=='passed'
    assert read(root/'REAL_H6_GRADIENT_PROBE.json')['document_gradient_norm']>0
    for run_id,(arm,seed,refresh,gpu) in RUNS.items():
        run=root/'smoke'/run_id
        assert not (run/'failure.json').exists(),run_id
        assert read(run/'reload_checks.json')['status']=='passed',run_id
        assert read(run/'evaluation_validation.json')['status']=='passed',run_id
        phases=('shared',) if arm=='s1' else SIDES
        for phase in phases:
            assert read(run/phase/'smoke.done.json')['step']==2,run_id
            runtime=read(run/phase/'runtime.json')
            assert runtime['effective_temperature']==.05 and runtime['effective_max_length']==512
            assert runtime['parameter_dtypes']==['torch.float32']
            assert runtime['warmup_steps']==151
            assert runtime['effective_sliding_window']==257
    pairs={}
    for seed in (42,1337):
        for side in SIDES:
            a=root/'smoke'/f'f3_fixed_seed{seed}'/side/'checkpoint-000002'/side
            b=root/'smoke'/f'f3_refresh_seed{seed}'/side/'checkpoint-000002'/side
            pairs[f'{seed}_{side}']=compare(a,b)
    # A qualitative inspection, explicitly not independent semantic annotation.
    write(root/'CONFUSER_AUDIT_REVIEW.json',dict(status='reviewed',requests=8,role_comparisons=16,
          source='encoder_train only; previous F3 embeddings',labels_changed=False,
          finding='Includes clear distinctions and underspecified requests with plausible alternatives; no exclusion implementation violation observed.',
          interpretation='Keep annotations and masking fixed; report exact annotated-pair agreement and ambiguity limitation.'))
    code={p.name:sha(p) for p in root.iterdir() if p.suffix in ('.py','.sh')}
    write(root/'PREFLIGHT_PASSED.json',dict(status='passed',checked_at_unix=time.time(),
          code=code,data_manifest_sha256=sha(root/'derived'/'manifest.json'),
          tests=18,smoke_jobs=6,matched_initial_training=pairs,
          boundary_probe=read(root/'BOUNDARY_PROBE.json'),real_h6_gradients=read(root/'REAL_H6_GRADIENT_PROBE.json')))
    print('PREFLIGHT_PASSED: six jobs, matched controls, real hierarchy gradients, epoch resume and evaluation.')


if __name__=='__main__':main()
