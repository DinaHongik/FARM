"""Replay saved workflows with network disabled to validate traces and blinding."""
import argparse
from pathlib import Path
from common import ROOT, read, rows, sha, digest, canonical, save
from cloud import Cloud
from workflows import ARMS, initial_farm, run
from output_health import output_health


def validate(directory,smoke=False):
    manifest=read(directory/'manifest.json')
    prefix='smoke_' if smoke else ''
    assert manifest['inputs_sha256']==sha(ROOT/f'private/{prefix}inputs.jsonl')
    assert manifest['reference_sha256']==sha(ROOT/f'private/{prefix}references.json')
    assert manifest['protocol_sha256']==sha(ROOT/'PROTOCOL.md')
    for name,value in manifest['scripts_sha256'].items():assert sha(ROOT/'scripts'/name)==value
    class Replay(Cloud):
        def chat(self,identity,system,payload):
            path=self.directory/(digest(identity)+'.json')
            if not path.exists():raise RuntimeError('Replay refuses any network call')
            cached=read(path)
            assert cached['request_sha256']==digest(cached['request'])
            assert cached['request']['model']==manifest['model']
            # Cloud compares exact request body against the cached request hash.
            return super().chat(identity,system,payload)
    client=Replay('unused-network-disabled',directory/'cache',manifest['model'])
    cases=rows(ROOT/f'private/{prefix}inputs.jsonl');verified=0;parsed={arm:0 for arm in ARMS}
    for case in cases:
        shared=initial_farm(client,case)
        for arm in ARMS:
            expected=run(client,case,arm,shared if arm.startswith('farm_') else None)
            saved=read(directory/'records'/arm/(digest(case['case_id'])+'.json'))
            assert canonical(expected)==canonical(saved),(case['case_id'],arm)
            parsed[arm]+=saved['final_checks']['protocol_valid'];verified+=1
    assert verified==len(cases)*3
    # Gate on response completion/shape, never on reference scores.
    health=output_health(directory)
    save(ROOT/'results'/('SMOKE_OUTPUT_VALIDATION.json' if smoke else 'OUTPUT_VALIDATION.json'),health)
    if smoke:
        assert health['terminal_outputs']==12,'Smoke must contain all four cases across all three arms'
        assert health['configured_output_limits']==[8192],'Smoke output budget differs from the frozen rerun'
        assert health['configured_thinking_modes']==[False],'Smoke thinking mode differs from the frozen rerun'
        assert health['status']=='passed',('Smoke output-completion gate failed; full run must not start',health)
    result={'status':'passed','verified_terminal_traces':verified,'protocol_valid_by_arm':parsed,
            'network_calls_during_replay':0,'references_hidden_from_generation_and_repair':True,
            'manifest_sha256':sha(directory/'manifest.json')}
    result['output_completion_status']=health['status']
    save(ROOT/'results'/('SMOKE_VALIDATION.json' if smoke else 'RUN_VALIDATION.json'),result)
    print(result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,default=ROOT/'private/full_run');p.add_argument('--smoke',action='store_true')
    a=p.parse_args();validate(a.directory,a.smoke)
