"""Local Ollama adapter for the recovered FARM workflows.

Credentials are read only from environment variables and never written to cache.
The offline adapter exercises orchestration; it is not a language model.
"""
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

from .common import canonical, digest, read, save


class ProviderError(RuntimeError):
    pass


class Client:
    def __init__(self, provider, model, directory, *, host='http://127.0.0.1:11434',
                 max_tokens=8192, timeout=240, think=False):
        if provider not in {'ollama', 'offline'}:
            raise ValueError('Unknown provider')
        if not model or max_tokens <= 0 or timeout <= 0:
            raise ValueError('Model, positive token budget and timeout are required')
        self.provider, self.model = provider, model
        self.directory = Path(directory)
        self.max_tokens, self.timeout, self.think = max_tokens, timeout, think
        if provider == 'ollama':
            parsed = urlsplit(host)
            if (parsed.scheme != 'http' or parsed.hostname not in {'localhost', '127.0.0.1', '::1'}
                    or parsed.username or parsed.password or parsed.query or parsed.fragment
                    or parsed.path not in {'', '/'}):
                raise ValueError('Use a local Ollama HTTP origin; tunnel remote servers to localhost')
        self.endpoint = host.rstrip('/')+'/api/chat' if provider == 'ollama' else 'offline'

    @staticmethod
    def _offline(system, payload):
        endpoints = payload['selected_endpoints']
        decisions = {side+'_fields': [
            {'field': f['slug'], 'source': ({'kind': 'omit'} if f['required'] is False else
                {'kind': 'needs_input', 'question': 'Please supply '+f['slug']+'.'})}
            for f in endpoints[side]['fields']] for side in ['trigger', 'action']}
        if 'You are the planner.' in system:
            return {'trigger_intent': 'offline fixture', 'action_intent': 'offline fixture', 'constraints': []}
        if 'You are the verifier.' in system:
            return {'repair_needed': False, 'issues': []}
        if 'You are the trigger specialist.' in system:
            return {'trigger_fields': decisions['trigger_fields']}
        return {**decisions, 'preview': 'Offline plumbing check; unresolved inputs require user answers.'}

    def chat(self, identity, system, payload):
        messages = [{'role': 'user', 'content': canonical(payload)}]
        body = {'model': self.model, 'stream': False, 'format': 'json',
                'messages': [{'role': 'system', 'content': system}, *messages],
                'options': {'temperature': 0, 'seed': 9052026, 'num_predict': self.max_tokens}}
        if self.think is not None:
            body['think'] = self.think
        signature = digest({'provider': self.provider, 'endpoint': self.endpoint, 'body': body})
        path = self.directory/(digest(identity)+'.json')
        if path.exists():
            cached = read(path)
            if cached['request_sha256'] != signature:
                raise ProviderError('Cached request differs; use a new output directory')
            return cached
        started = time.monotonic()
        if self.provider == 'offline':
            raw = {'content': canonical(self._offline(system, payload))}
            content, reason, prompt_tokens, output_tokens = raw['content'], 'offline', 0, 0
        else:
            headers = {'Content-Type': 'application/json'}
            request = urllib.request.Request(self.endpoint, data=canonical(body).encode(), headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = json.load(response)
            except urllib.error.HTTPError as exc:
                # HTTP response bodies and headers can contain credential material.
                raise ProviderError(f'{self.provider} returned HTTP {exc.code}') from None
            except (OSError, TimeoutError, ValueError):
                raise ProviderError(f'{self.provider} transport failed; resume with the same command') from None
            content = raw.get('message', {}).get('content', '')
            reason = raw.get('done_reason')
            prompt_tokens, output_tokens = raw.get('prompt_eval_count', 0), raw.get('eval_count', 0)
        try:
            parsed = json.loads(content)
        except (ValueError, TypeError):
            parsed = None
        # A truncated response is retained and scored as an invalid output.
        if reason in {'length', 'max_tokens'}:
            parsed = None
        value = {'request_sha256': signature, 'response_sha256': digest(raw),
                 'provider': self.provider, 'model': self.model, 'content': content, 'parsed': parsed,
                 'prompt_tokens': prompt_tokens, 'completion_tokens': output_tokens,
                 'elapsed_seconds': time.monotonic()-started, 'transport_attempts': 1,
                 'done_reason': reason, 'request': body}
        save(path, value)
        return value
