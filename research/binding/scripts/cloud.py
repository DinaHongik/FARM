"""Ollama Cloud transport for this explicitly user-authorized FARM experiment."""
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from common import canonical, digest, freeze, read, save


def load_key(path):
    values = {}
    for line in Path(path).read_text().splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            k, v = line.split('=', 1)
            values[k.strip()] = v.strip().strip('"').strip("'")
    key = values.get('OLLAMA_API_KEY')
    if not key:
        raise RuntimeError('OLLAMA_API_KEY is missing')
    return key


class Cloud:
    def __init__(self, key, directory, model='deepseek-v4-flash:0731'):
        self.key, self.directory, self.model = key, Path(directory), model

    def chat(self, identity, system, payload):
        body = {'model': self.model, 'stream': False, 'think': False, 'format': 'json',
                'messages': [{'role': 'system', 'content': system},
                             {'role': 'user', 'content': canonical(payload)}],
                'options': {'temperature': 0, 'seed': 9052026, 'num_predict': 8192}}
        path = self.directory/(digest(identity)+'.json')
        signature = digest(body)
        if path.exists():
            value = read(path)
            if value['request_sha256'] != signature:
                raise RuntimeError('Cached model request changed')
            return value
        started = time.monotonic()
        for attempt in range(3):
            try:
                request = urllib.request.Request('https://ollama.com/api/chat',
                    data=canonical(body).encode(), headers={'Content-Type': 'application/json',
                    'Authorization': 'Bearer '+self.key})
                with urllib.request.urlopen(request, timeout=240) as response:
                    raw = json.load(response)
                if raw.get('model') != self.model:
                    raise RuntimeError('Provider returned a different model identifier')
                break
            except urllib.error.HTTPError as exc:
                if exc.code not in [429, 500, 502, 503, 504] or attempt == 2:
                    raise RuntimeError('Provider HTTP '+str(exc.code)) from None
                time.sleep(5*(attempt+1))
            except (OSError, TimeoutError):
                if attempt == 2:
                    raise RuntimeError('Provider transport failed') from None
                time.sleep(3*(attempt+1))
        content = raw.get('message', {}).get('content', '')
        parsed = None
        try:
            parsed = json.loads(content)
        except (ValueError, TypeError):
            pass
        result = {'request_sha256': signature, 'response_sha256': digest(raw),
                  'model': raw.get('model'), 'content': content, 'parsed': parsed,
                  'prompt_tokens': raw.get('prompt_eval_count', 0),
                  'completion_tokens': raw.get('eval_count', 0),
                  'provider_seconds': raw.get('total_duration', 0)/1e9,
                  'elapsed_seconds': time.monotonic()-started, 'transport_attempts': attempt+1,
                  'done_reason': raw.get('done_reason'), 'thinking_characters': len(raw.get('message', {}).get('thinking', '')), 'request': body}
        save(path, result)
        return result
