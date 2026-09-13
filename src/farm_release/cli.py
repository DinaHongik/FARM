"""Run the released examples without access to private data or a DGX machine."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from . import __version__
from .common import digest, freeze, read, save, sha
from .providers import Client, ProviderError
from .workflows import ARMS, COMMON, check, envelope, initial_farm, run


def load_examples(path):
    examples = read(path)
    if not isinstance(examples, list) or not examples:
        raise ValueError('Input must be a nonempty JSON array')
    ids = set()
    for case in examples:
        identity = case['case_id']
        if not isinstance(identity, str) or not identity or identity in ids:
            raise ValueError('Example IDs must be nonempty and unique')
        ids.add(identity)
        if not isinstance(case['query'], str) or not case['query']:
            raise ValueError('Every example needs a query')
        if set(case['endpoints']) != {'trigger', 'action'}:
            raise ValueError('Every example needs trigger and action metadata')
        for side in ['trigger', 'action']:
            ep = case['endpoints'][side]
            if ep['side'] != side or not ep['endpoint_id'].startswith('https://ifttt.com/'):
                raise ValueError('Unexpected endpoint identity')
            if len({f['slug'] for f in ep['fields']}) != len(ep['fields']):
                raise ValueError('Duplicate field slugs')
            for f in ep['fields']:
                if f['required'] is not None and type(f['required']) is not bool:
                    raise ValueError('Requiredness must be boolean or unknown')
        envelope(case)
    return examples


def execute(args):
    examples = load_examples(args.input)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError('--limit must be positive')
        examples = examples[:args.limit]
    if args.provider != 'offline' and not args.model:
        raise ValueError('--model is required for a real provider')
    model = args.model or 'offline-fixture'
    arms = list(ARMS) if args.arm == 'all' else [args.arm]
    think = None if args.think == 'omit' else args.think == 'on'
    client = Client(args.provider, model, args.output/'cache', host=args.host,
                    max_tokens=args.max_tokens, timeout=args.timeout, think=think)
    manifest = {'version': __version__, 'input_sha256': sha(args.input), 'n': len(examples),
                'case_ids_sha256': digest([c['case_id'] for c in examples]), 'arms': arms,
                'provider': args.provider, 'model': model, 'endpoint': client.endpoint,
                'max_tokens': args.max_tokens, 'temperature': 0, 'ollama_seed': 9052026,
                'think': think, 'condition': 'supplied_annotated_endpoints',
                'semantic_binding_accuracy': None, 'live_execution': False,
                'offline_fixture': args.provider == 'offline',
                'code_sha256': {p.name: sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}}
    # Lock prevents two processes from writing the same resumable run.
    args.output.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (args.output/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        freeze(args.output/'manifest.json', manifest)
        records = []
        for index, case in enumerate(examples, 1):
            shared = None
            for arm in arms:
                path = args.output/'records'/arm/(digest(case['case_id'])+'.json')
                if path.exists():
                    record = read(path)
                    if record['case_id'] != case['case_id'] or record['arm'] != arm:
                        raise ValueError('Completed record identity mismatch')
                else:
                    if arm.startswith('farm_') and shared is None:
                        shared = initial_farm(client, case)
                    record = run(client, case, arm, shared)
                    save(path, record)
                records.append(record)
            print(f'Completed {index}/{len(examples)} examples', flush=True)
        summary = {'version': __version__, 'provider': args.provider, 'model': model,
                   'n': len(examples), 'offline_fixture': args.provider == 'offline',
                   'condition': manifest['condition'], 'semantic_binding_accuracy': None,
                   'live_execution': False, 'paper_results_reproduced': False, 'arms': {}}
        for arm in arms:
            subset = [r for r in records if r['arm'] == arm]
            summary['arms'][arm] = {
                'n': len(subset),
                'complete_decisions': sum(bool(r['final_checks']['complete']) for r in subset),
                'complete_without_detected_error': sum(bool(r['final_checks']['complete']) and not r['final_checks']['errors'] for r in subset),
                'locally_valid': sum(bool(r['final_checks'].get('locally_valid')) for r in subset),
                'logical_calls': sum(len(r['calls']) for r in subset),
                'mean_calls': sum(len(r['calls']) for r in subset)/len(subset)}
        save(args.output/'summary.json', summary)
        print(json.dumps(summary, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=f'FARM {__version__} reproducibility release (run from the repository root)')
    parser.add_argument('--version', action='version', version=__version__)
    subs = parser.add_subparsers(dest='command', required=True)
    validate = subs.add_parser('validate', help='Validate the released JSON examples')
    validate.add_argument('--input', type=Path, default=Path('data/ifttt_examples.json'))
    prompts = subs.add_parser('export-prompts', help='Export a per-example single-agent prompt for manual use')
    prompts.add_argument('--input', type=Path, default=Path('data/ifttt_examples.json'))
    prompts.add_argument('--output', type=Path, default=Path('outputs/prompts.json'))
    runner = subs.add_parser('run', help='Run/resume the public configuration examples')
    runner.add_argument('--input', type=Path, default=Path('data/ifttt_examples.json'))
    runner.add_argument('--output', type=Path, required=True)
    runner.add_argument('--provider', choices=['ollama', 'offline'], required=True)
    runner.add_argument('--model')
    runner.add_argument('--arm', choices=['all', *ARMS], default='all')
    runner.add_argument('--host', default='http://127.0.0.1:11434')
    runner.add_argument('--limit', type=int)
    runner.add_argument('--max-tokens', type=int, default=8192)
    runner.add_argument('--timeout', type=float, default=240)
    runner.add_argument('--think', choices=['off', 'on', 'omit'], default='off')
    args = parser.parse_args(argv)
    try:
        if args.command == 'validate':
            examples = load_examples(args.input)
            print(f'Validated {len(examples)} unique IFTTT examples; field-binding gold is not supplied.')
        elif args.command == 'export-prompts':
            save(args.output, [{'case_id': c['case_id'], 'system': COMMON,
                                'user': envelope(c)} for c in load_examples(args.input)])
            print('Wrote '+str(args.output))
        else:
            execute(args)
    except (RuntimeError, ValueError, KeyError, FileNotFoundError, BlockingIOError) as exc:
        parser.exit(1, f'FARM: {exc}\n')
