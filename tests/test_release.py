import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from farm_release.cli import load_examples, main
from farm_release.common import read
from farm_release.providers import Client, ProviderError
from farm_release.workflows import envelope, check, initial_farm, run

ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.examples = load_examples(ROOT/'data/ifttt_examples.json')

    def test_100_real_unique_examples_have_no_semantic_gold_in_prompts(self):
        self.assertEqual(len(self.examples), 100)
        self.assertEqual(len({c['query'] for c in self.examples}), 100)
        for case in self.examples:
            prompt = envelope(case)
            self.assertEqual(set(prompt), {'query', 'selected_endpoints'})
            self.assertIsNone(case['semantic_binding_reference'])
            for forbidden in ['case_id', 'source_applet_url', 'semantic_binding_reference', 'private_gold']:
                self.assertNotIn(forbidden, json.dumps(prompt))

    def test_author_reported_rates_are_retained_without_manuscript(self):
        self.assertFalse((ROOT/'paper').exists())
        rates = read(ROOT/'docs/reported_metrics.json')['binding']
        self.assertEqual([rates[a] for a in ['single_agent','farm_feedback','farm_no_feedback']], [75.5,82.2,48.67])

    def test_all_three_arms_resume_without_new_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'run'
            argv = ['run','--input',str(ROOT/'data/ifttt_examples.json'),'--provider','offline',
                    '--limit','3','--output',str(output)]
            with contextlib.redirect_stdout(io.StringIO()):
                main(argv)
                with patch.object(Client, 'chat', side_effect=AssertionError('resume called provider')):
                    main(argv)
            summary = read(output/'summary.json')
            self.assertEqual(len(list((output/'records').glob('*/*.json'))), 9)
            self.assertIsNone(summary['semantic_binding_accuracy'])
            self.assertFalse(summary['paper_results_reproduced'])
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main(argv+['--max-tokens','4096'])

    def test_farm_arms_share_identical_initial_draft_and_call_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            client = Client('offline','fixture',Path(directory))
            case = self.examples[0]
            shared = initial_farm(client,case)
            repaired = run(client,case,'farm_feedback',shared)
            unchanged = run(client,case,'farm_no_feedback',shared)
            self.assertEqual(repaired['initial'],unchanged['final'])
            self.assertLessEqual(len(repaired['calls']),7)
            self.assertEqual(len(unchanged['calls']),3)

    def test_fabricated_ingredient_and_invalid_span_are_rejected(self):
        case = next(c for c in self.examples if c['endpoints']['action']['fields'])
        with tempfile.TemporaryDirectory() as directory:
            client = Client('offline','fixture',Path(directory))
            draft = initial_farm(client,case)[0]
            draft['action_fields'][0]['source'] = {'kind':'trigger_output','ingredient_slug':'does_not_exist'}
            self.assertTrue(check(case,draft)['errors'])
            draft['action_fields'][0]['source'] = {'kind':'trigger_output','ingredient_slug':['invalid']}
            self.assertTrue(check(case,draft)['errors'])
            draft['action_fields'][0]['source'] = {'kind':'query_span','start':0,'end':4,'text':'fake','transform':'identity'}
            self.assertTrue(check(case,draft)['errors'])

    def test_local_http_request_uses_no_auth_and_caches(self):
        raw = {'model':'test-model','message':{'content':'{}'},'done_reason':'stop'}
        with tempfile.TemporaryDirectory() as directory:
            client=Client('ollama','test-model',Path(directory))
            with patch('farm_release.providers.urllib.request.urlopen',return_value=io.BytesIO(json.dumps(raw).encode())) as call:
                result=client.chat('test','Return JSON',{})
                request=call.call_args.args[0]
                self.assertNotIn('Authorization',request.headers)
                self.assertNotIn('X-api-key',request.headers)
                body=json.loads(request.data)
                self.assertFalse(body['think'])
                self.assertEqual(body['options']['num_predict'],8192)
                self.assertEqual(client.chat('test','Return JSON',{}),result)
                self.assertEqual(call.call_count,1)
            with self.assertRaises(ProviderError):
                client.chat('test','Different prompt',{})

    def test_truncated_response_is_invalid_even_when_json_parses(self):
        raw={'message':{'content':'{}'},'done_reason':'length',
             'prompt_eval_count':12,'eval_count':8}
        with tempfile.TemporaryDirectory() as directory:
            client=Client('ollama','test-model',Path(directory))
            with patch('farm_release.providers.urllib.request.urlopen',return_value=io.BytesIO(json.dumps(raw).encode())) as call:
                result=client.chat('test','Return JSON',{})
                self.assertIsNone(result['parsed'])
                self.assertEqual(result['prompt_tokens'],12)

    def test_local_adapter_rejects_remote_or_credential_bearing_host(self):
        for host in ['https://example.org','http://127.0.0.1:11434?token=x','http://user:secret@localhost:11434']:
            with self.assertRaises(ValueError):
                Client('ollama','test-model',Path('/unused'),host=host)


if __name__ == '__main__':
    unittest.main()
