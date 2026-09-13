"""Run/resume the three frozen workflows; retain private per-case checkpoints."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
from pathlib import Path
import queue
import threading
import time
from cloud import Cloud, load_key
from common import ROOT, digest, freeze, read, rows, save, sha
from workflows import ARMS, initial_farm, run


def execute(args):
    inputs = rows(args.inputs)
    if not args.smoke and len(inputs) != 150:
        raise RuntimeError('The full experiment requires exactly 150 inputs')
    assert len({c['case_id'] for c in inputs}) == len(inputs)
    args.output.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = (args.output/'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    key = load_key(args.env_file)
    reference_file=ROOT/'private'/('smoke_references.json' if args.smoke else 'references.json')
    manifest = {'n':len(inputs), 'condition':'controlled_smoke' if args.smoke else 'controlled_requests_supplied_endpoints',
                'model':args.model, 'inputs_sha256':sha(args.inputs),
                'reference_sha256':sha(reference_file),
                'protocol_sha256':sha(ROOT/'PROTOCOL.md'), 'arms':list(ARMS),
                'scripts_sha256':{p.name:sha(p) for p in sorted((ROOT/'scripts').glob('*.py'))},
                'checker_sha256':sha(ROOT/'reference_code/configuration.py'),
                'cloud_authorization':'User explicitly requested their Ollama API for this experiment',
                'max_calls_per_case_arm':7, 'max_generated_tokens_per_call':8192, 'thinking':False,
                'temperature':0, 'inference_seed':9052026}
    freeze(args.output/'manifest.json', manifest)
    pending = queue.Queue()
    for c in inputs: pending.put(c)
    guard = threading.Lock(); stop = threading.Event(); errors = []
    started = time.time(); completed = 0

    def status(state):
        paths = list((args.output/'records').glob('*/*.json'))
        save(args.output/'status.json', {'state':state, 'expected_cases':len(inputs),
             'expected_case_arm_results':len(inputs)*3, 'completed_case_arm_results':len(paths),
             'completed_cases_this_invocation':completed, 'started_unix':started,
             'updated_unix':time.time(), 'errors':errors, 'condition':manifest['condition'],
             'model':args.model, 'scoring_state':'awaiting_complete_predictions'})

    def worker():
        nonlocal completed
        client = Cloud(key, args.output/'cache', args.model)
        while not stop.is_set():
            try: case = pending.get_nowait()
            except queue.Empty: return
            case_hash = digest(case['case_id'])
            try:
                # Alternate work order without looking at labels or outputs.
                order = list(ARMS) if int(case_hash[0],16)%2 else [ARMS[1], ARMS[2], ARMS[0]]
                shared = None
                for arm in order:
                    path = args.output/'records'/arm/(case_hash+'.json')
                    if path.exists():
                        old=read(path)
                        assert old['case_id']==case['case_id'] and old['arm']==arm
                        continue
                    if arm.startswith('farm_') and shared is None:
                        shared = initial_farm(client, case)
                    record = run(client, case, arm, shared)
                    save(path, record)
                    with guard: status('running')
                with guard:
                    completed += 1; status('running')
                    print(f'Completed {completed}/{len(inputs)} cases across all three workflows', flush=True)
            except Exception as exc:
                # Infrastructure failures pause the experiment and remain resumable;
                # invalid model JSON is retained as an ordinary scored failure.
                with guard:
                    errors.append({'type':type(exc).__name__, 'message':str(exc)[:160]})
                    stop.set(); status('paused_error')
                return
            finally: pending.task_done()

    status('running')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(lambda _:worker(), range(args.workers)))
    status('paused_error' if errors else 'complete')
    if errors:
        raise RuntimeError('Experiment paused; inspect the private status record and resume with the same manifest')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--inputs',type=Path,default=ROOT/'private/oracle_inputs.jsonl')
    p.add_argument('--output',type=Path,default=ROOT/'private/full_run')
    p.add_argument('--env-file',type=Path,required=True)
    p.add_argument('--model',default='deepseek-v4-flash:0731')
    p.add_argument('--workers',type=int,choices=[1,2],default=2)
    p.add_argument('--smoke',action='store_true')
    execute(p.parse_args())
