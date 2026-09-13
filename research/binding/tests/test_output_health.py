import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from common import save
from output_health import output_health
from cloud import Cloud


class OutputHealthTests(unittest.TestCase):
    def health(self, call, valid_shape=True):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            save(root / 'cache/call.json', call)
            save(root / 'records/single_agent/case.json', {
                'final_checks': {'protocol_valid': valid_shape, 'errors': ['semantic_error']}
            })
            return output_health(root)

    def complete_call(self):
        return {'done_reason': 'stop', 'content': '{}', 'parsed': {},
                'completion_tokens': 500, 'request': {'think': False, 'options': {'num_predict': 8192}}}

    def test_empty_length_stopped_response_blocks_full_run(self):
        call = self.complete_call()
        call.update(done_reason='length', content='', parsed=None, completion_tokens=8192)
        result = self.health(call, False)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['length_stopped_calls'], 1)
        self.assertEqual(result['empty_content_calls'], 1)

    def test_length_stop_is_rejected_even_if_json_parses(self):
        call = self.complete_call(); call['done_reason'] = 'length'
        self.assertEqual(self.health(call)['status'], 'failed')

    def test_malformed_response_and_bad_terminal_shape_are_rejected(self):
        call = self.complete_call(); call.update(content='{', parsed=None)
        self.assertEqual(self.health(call)['status'], 'failed')
        self.assertEqual(self.health(self.complete_call(), False)['status'], 'failed')

    def test_semantic_scores_do_not_gate_completion(self):
        self.assertEqual(self.health(self.complete_call())['status'], 'passed')

    def test_missing_artifacts_cannot_pass(self):
        with tempfile.TemporaryDirectory() as name:
            self.assertEqual(output_health(Path(name))['status'], 'failed')

    def test_unexpected_thinking_content_fails_gate(self):
        call = self.complete_call(); call['thinking_characters'] = 100
        self.assertEqual(self.health(call)['status'], 'failed')

    def test_actual_cloud_request_disables_thinking_and_sets_budget(self):
        response = {'model': 'deepseek-v4-flash:0731', 'done_reason': 'stop',
                    'message': {'content': '{}', 'thinking': ''}, 'eval_count': 2}
        with tempfile.TemporaryDirectory() as name:
            with patch('cloud.urllib.request.urlopen', return_value=io.BytesIO(json.dumps(response).encode())) as send:
                result = Cloud('unused-test-key', Path(name)).chat('probe', 'Return JSON.', {})
            body = json.loads(send.call_args.args[0].data)
            self.assertIs(body['think'], False)
            self.assertEqual(body['options']['num_predict'], 8192)
            self.assertEqual(result['thinking_characters'], 0)
            self.assertEqual(result['parsed'], {})


if __name__ == '__main__':
    unittest.main()
