"""Two blinded AI reference proposals; cached, resumable, and never human gold."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import threading
import time
import urllib.error
import urllib.request
from common import ROOT, REVISION, read, save, freeze, sha, digest, canonical
from reference import SYSTEM, validate_reference

MODELS = {'A': 'qwen3.5:397b', 'B': 'gpt-oss:120b'}

def load_key(path):
    for line in path.read_text().splitlines():
        if line.startswith('OLLAMA_API_KEY='):
            return line.split('=', 1)[1].strip().strip('\"').strip("'")
    raise RuntimeError('Ollama key is not configured')

def request(key, model, case):
    # Deliberately exclude predictions, workflow IDs, scoring gold, and other annotators.
    payload = {k: case[k] for k in ['query', 'endpoints', 'official_documentation']}
    body = {'model': model, 'stream': False, 'format': 'json',
            'think': 'low' if model.startswith('gpt-oss') else False,
            'messages': [{'role': 'system', 'content': SYSTEM},
                         {'role': 'user', 'content': canonical(payload)}],
            'options': {'temperature': 0, 'seed': 20260906, 'num_predict': 8192}}
    signature = digest(body)
    cache = ROOT / 'private/cache' / (signature + '.json')
    if cache.exists():
        return read(cache)
    began = time.monotonic()
    for attempt in range(3):
        try:
            req = urllib.request.Request('https://ollama.com/api/chat', data=canonical(body).encode(),
                headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
            with urllib.request.urlopen(req, timeout=240) as response:
                raw = json.load(response)
            break
        except urllib.error.HTTPError as e:
            if e.code not in [429, 500, 502, 503, 504] or attempt == 2:
                raise RuntimeError('Provider HTTP ' + str(e.code)) from None
            time.sleep(3 * (attempt + 1))
        except (TimeoutError, OSError):
            if attempt == 2: raise RuntimeError('Provider transport failure') from None
            time.sleep(3 * (attempt + 1))
    if raw.get('model') != model:
        raise RuntimeError('Model identifier mismatch')
    content = raw.get('message', {}).get('content', '')
    try: parsed = json.loads(content)
    except (ValueError, TypeError): parsed = None
    item = {'request_sha256': signature, 'request': body, 'parsed': parsed,
            'content': content, 'model': raw.get('model'), 'done_reason': raw.get('done_reason'),
            'input_tokens': raw.get('prompt_eval_count', 0), 'output_tokens': raw.get('eval_count', 0),
            'elapsed_seconds': time.monotonic() - began, 'response_sha256': digest(raw)}
    save(cache, item)
    return item

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=150)
    parser.add_argument('--workers', type=int, choices=[1,2,3,4], default=2)
    args = parser.parse_args()
    cases = read(ROOT / 'private/ANNOTATION_INPUTS.json')[:args.limit]
    key = load_key(REVISION.parent / 'FARM/.env')
    identity = {'models': MODELS, 'input_sha256': sha(ROOT / 'private/ANNOTATION_INPUTS.json'),
                'rubric_sha256': digest(SYSTEM), 'reference_validator_sha256': sha(ROOT / 'scripts/reference.py'),
                'annotator_script_sha256': sha(__file__), 'reference_mode': 'ai_assisted_provisional'}
    freeze(ROOT / 'private/ANNOTATION_MANIFEST.json', identity)
    guard = threading.Lock(); stop = threading.Event(); failures = []
    completed = 0
    def update(state):
        save(ROOT / 'STATUS.json', {'state': state, 'expected_proposals': len(cases)*2,
             'completed_proposals': completed, 'human_reviewed_cases': 0,
             'mode': 'ai_assisted_provisional', 'errors': failures,
             'updated_unix': time.time(), 'pc_must_remain_on': state=='running'})
    def one(job):
        global completed
        case, label = job
        if stop.is_set(): return
        path = ROOT / 'private/proposals' / label / (digest(case['case_id']) + '.json')
        try:
            if path.exists():
                item = read(path)
                assert item['input_sha256'] == digest(case)
            else:
                output = request(key, MODELS[label], case)
                problems = validate_reference(case, output['parsed'])
                item = {'case_id': case['case_id'], 'case_number': case['case_number'],
                        'annotator_label': label, 'model': MODELS[label], 'input_sha256': digest(case),
                        'request_sha256': output['request_sha256'], 'reference': output['parsed'],
                        'validation_errors': problems, 'reference_state': 'ai_provisional',
                        'human_review': None}
                save(path, item)
            with guard:
                completed += 1; update('running')
                if completed % 10 == 0: print(f'Reference proposals {completed}/{len(cases)*2}', flush=True)
        except Exception as e:
            with guard:
                failures.append({'type': type(e).__name__, 'message': str(e)[:120]})
                stop.set(); update('paused_error')
    update('running')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(one, [(case, label) for case in cases for label in MODELS]))
    update('paused_error' if failures else 'proposals_complete')
    if failures: raise SystemExit('Reference preparation paused; cached work is retained.')
    print('AI reference proposals completed. Human-reviewed gold has not been created.')
