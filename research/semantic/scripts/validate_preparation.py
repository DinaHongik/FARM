"""Verify blinding, source identities, immutable predictions, and pilot limits."""
from common import ROOT, REVISION, read, sha, digest, canonical, save
from reference import SYSTEM
from scoring import validate_scoring_reference

if __name__ == '__main__':
    manifest = read(ROOT/'private/ANNOTATION_MANIFEST.json')
    inputs = read(ROOT/'private/ANNOTATION_INPUTS.json')
    assert manifest['input_sha256'] == sha(ROOT/'private/ANNOTATION_INPUTS.json')
    assert manifest['rubric_sha256'] == digest(SYSTEM)
    assert manifest['reference_validator_sha256'] == sha(ROOT/'scripts/reference.py')
    assert manifest['annotator_script_sha256'] == sha(ROOT/'scripts/annotate.py')
    checked = 0
    for case in inputs:
        for label, model in manifest['models'].items():
            proposal = read(ROOT/'private/proposals'/label/(digest(case['case_id'])+'.json'))
            cache = read(ROOT/'private/cache'/(proposal['request_sha256']+'.json'))
            assert proposal['input_sha256'] == digest(case)
            assert cache['request_sha256'] == digest(cache['request'])
            assert cache['request']['model'] == cache['model'] == model
            assert cache['parsed'] == proposal['reference']
            assert cache['request']['messages'][0]['content'] == SYSTEM
            expected = {k:case[k] for k in ['query','endpoints','official_documentation']}
            assert cache['request']['messages'][1]['content'] == canonical(expected)
            assert proposal['human_review'] is None and proposal['reference_state']=='ai_provisional'
            checked += 1
    assert checked == 300
    frozen = read(ROOT/'private/PREDICTION_FREEZE.json')
    for relative, checksum in frozen.items():
        assert sha(REVISION/'agentic_validation/private/full_run/records'/relative)==checksum
    refs = read(ROOT/'private/AI_CONSENSUS.json')
    assert len(refs)==150
    for case, ref in zip(inputs, refs):
        assert case['case_id']==ref['case_id']
        assert not validate_scoring_reference(case,ref['reference'])
        assert ref['human_review'] is None
    pilot=read(ROOT/'results/AI_PILOT_RESULTS.json')
    assert pilot['human_reviewed_cases']==0
    assert pilot['semantic_binding_accuracy_claim_permitted'] is False
    assert pilot['reference_sha256']==sha(ROOT/'private/AI_CONSENSUS.json')
    summary=read(ROOT/'results/REFERENCE_PREPARATION.json')
    assert summary['agreed_fields']+summary['fields_requiring_resolution']==558
    record={'status':'passed','proposals_verified':300,'original_predictions_unchanged':len(frozen),
            'blinding':'exact allowlisted request/schema/documentation payloads; no workflow outputs or other proposal',
            'human_reviewed_cases':0,'reference_sha256':sha(ROOT/'private/AI_CONSENSUS.json'),
            'pilot_sha256':sha(ROOT/'results/AI_PILOT_RESULTS.json'),
            'reference_summary':summary,'scoring_source_sha256':sha(ROOT/'scripts/scoring.py')}
    save(ROOT/'results/PREPARATION_VALIDATION.json',record)
    print(canonical(record))
