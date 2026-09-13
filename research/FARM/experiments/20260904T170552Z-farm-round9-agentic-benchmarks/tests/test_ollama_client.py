from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from farm_r9.cloud_limiter import OllamaCloudLimiter
from farm_r9.ollama_client import OllamaChatClient, OllamaJSONError, parse_json_object
from farm_r9.privacy import CloudRoutingDenied, DataClassification, DataSource


class OllamaClientTests(unittest.TestCase):
    def test_explicit_thinking_and_token_limit_are_sent_and_cache_bound(self):
        payloads = []
        def transport(endpoint, headers, body, timeout):
            payloads.append(json.loads(body))
            return 200, json.dumps({"model": "model:1", "message": {"content": "{}"}}).encode()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            client = OllamaChatClient(host="http://127.0.0.1:11434", api_key=None,
                model="model:1", cloud=False, limiter=None,
                cache_directory=root / "cache", journal_path=root / "journal.jsonl", transport=transport)
            kwargs = dict(semantic_id="bounded", benchmark_label="recipegen_noisy",
                data_classification=DataClassification.PUBLIC, data_source=DataSource.RECIPEGEN,
                messages=[{"role": "user", "content": "public fixture"}], think=False)
            client.chat(**kwargs, max_output_tokens=8192)
            self.assertIs(payloads[0]["think"], False)
            self.assertEqual(payloads[0]["options"]["num_predict"], 8192)
            from farm_r9.ollama_client import OllamaProtocolError
            with self.assertRaises(OllamaProtocolError):
                client.chat(**kwargs, max_output_tokens=4096)
            self.assertEqual(len(payloads), 1)

    def test_cloud_denies_confidential_before_transport(self):
        called = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            client = OllamaChatClient(
                host="https://example.test", api_key="not-persisted", model="model:1",
                cache_directory=root / "cache", journal_path=root / "journal.jsonl", cloud=True,
                limiter=OllamaCloudLimiter(root / "leases"), transport=lambda *args: called.append(args),
            )
            with self.assertRaises(CloudRoutingDenied):
                client.chat(
                    semantic_id="x", benchmark_label="farm_v2_test",
                    data_classification=DataClassification.CONFIDENTIAL,
                    data_source=DataSource.FARM_V2,
                    messages=[{"role": "user", "content": "private"}],
                )
        self.assertEqual(called, [])

    def test_response_cache_prevents_duplicate_paid_request(self):
        calls = []
        response = {"model": "model:1", "message": {"content": '{"ok":true}'}, "prompt_eval_count": 2, "eval_count": 3}
        def transport(*args):
            calls.append(args)
            return 200, json.dumps(response).encode()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            client = OllamaChatClient(
                host="https://example.test", api_key="not-persisted", model="model:1",
                cache_directory=root / "cache", journal_path=root / "journal.jsonl", cloud=True,
                limiter=OllamaCloudLimiter(root / "leases"), transport=transport,
            )
            first = client.chat(
                semantic_id="case/arm/call-1", benchmark_label="recipegen_gold",
                data_classification=DataClassification.PUBLIC,
                data_source=DataSource.RECIPEGEN,
                messages=[{"role": "user", "content": "public"}],
            )
            second = client.chat(
                semantic_id="case/arm/call-1", benchmark_label="recipegen_gold",
                data_classification=DataClassification.PUBLIC,
                data_source=DataSource.RECIPEGEN,
                messages=[{"role": "user", "content": "public"}],
            )
        self.assertEqual(len(calls), 1)
        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)
        self.assertEqual(parse_json_object(second.content)[0], {"ok": True})

    def test_json_parser_records_non_strict_recovery(self):
        self.assertEqual(parse_json_object('{"x":1}'), ({"x": 1}, True))
        self.assertEqual(parse_json_object('```json\n{"x":1}\n```'), ({"x": 1}, False))
        with self.assertRaises(OllamaJSONError):
            parse_json_object("not json")


if __name__ == "__main__":
    unittest.main()
