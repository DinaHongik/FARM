"""Freeze inputs and archive official endpoint descriptions without predictions."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import re
import urllib.error
import urllib.request
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from common import ROOT, REVISION, read, rows, save, freeze, sha, digest

source = REVISION / 'agentic_validation/private/oracle_inputs.jsonl'
cases = rows(source)
assert len(cases) == len({c['case_id'] for c in cases}) == 150
urls = sorted({e['endpoint_id'] for c in cases for e in c['endpoints'].values()})

def snapshot(url):
    assert urlparse(url).scheme == 'https' and urlparse(url).netloc == 'ifttt.com'
    path = ROOT / 'private/documents' / (digest(url) + '.json')
    if path.exists():
        return read(path)
    record = {'url': url, 'fetched_utc': datetime.now(timezone.utc).isoformat(),
              'text': '', 'status': 'unavailable'}
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'FARM research documentation audit'})
        with urllib.request.urlopen(req, timeout=25) as response:
            body = response.read()
            final_url = response.geturl()
        soup = BeautifulSoup(body, 'html.parser')
        title = soup.find('h1')
        if title is None:
            raise ValueError('Missing endpoint title')
        # The first H1 follows site navigation. Stop before public applet examples.
        heading = title.get_text(' ', strip=True)
        if soup.head:
            soup.head.decompose()
        lines = soup.get_text('\n', strip=True).splitlines()
        start = next(i for i, line in enumerate(lines) if line == heading)
        lines = lines[start:]
        cleaned = []; skip = False
        for line in lines:
            if skip:
                skip = False; continue
            if line == 'Example':
                skip = True; continue
            cleaned.append(line)
        text = '\n'.join(cleaned)
        stops = [text.find(marker) for marker in ['Applets using this', 'Get the best business tools', 'Top Integrations']]
        stops = [p for p in stops if p >= 0]
        if stops:
            text = text[:min(stops)]
        expected = url.rsplit('/', 1)[-1]
        matched = (urlparse(final_url).path.rstrip('/') == urlparse(url).path.rstrip('/')
                   and ('Slug' in text or 'API endpoint slug' in text))
        if not matched:
            raise ValueError('Endpoint redirected or lacks a schema section')
        record.update(status='available', text=text.strip(), final_url=final_url,
                      html_sha256=__import__('hashlib').sha256(body).hexdigest())
    except urllib.error.HTTPError as exc:
        record['reason'] = 'HTTP ' + str(exc.code)
    except Exception as exc:
        record['reason'] = type(exc).__name__
    save(path, record)
    return record

if __name__ == '__main__':
    # Record hashes of predictions without opening their content in the annotation path.
    prediction_root = REVISION / 'agentic_validation/private/full_run/records'
    hashes = {str(p.relative_to(prediction_root)): sha(p)
              for p in sorted(prediction_root.glob('*/*.json'))}
    assert len(hashes) == 450
    freeze(ROOT / 'private/PREDICTION_FREEZE.json', hashes)
    with ThreadPoolExecutor(max_workers=3) as pool:
        docs = list(pool.map(snapshot, urls))
    by_url = {d['url']: d for d in docs}
    payloads = []
    for index, case in enumerate(cases, 1):
        references = {}
        for side, endpoint in case['endpoints'].items():
            doc = by_url[endpoint['endpoint_id']]
            frozen = {f['slug'] for f in endpoint['fields']}
            live_slugs = set(re.findall(r'(?:^|\n)Slug\n([^\n]+)', doc['text'])) if doc['text'] else set()
            # Live slugs also include ingredients. Only flag frozen fields absent
            # from the page; additional live slugs require human inspection.
            references[side] = {**doc, 'frozen_field_slugs_absent_from_page':
                                sorted(frozen - live_slugs) if doc['text'] else []}
        payloads.append({'case_number': index, 'case_id': case['case_id'],
                         'query': case['query'], 'endpoints': case['endpoints'],
                         'official_documentation': references})
    freeze(ROOT / 'private/ANNOTATION_INPUTS.json', payloads)
    manifest = {'n': 150, 'fields': sum(len(e['fields']) for c in cases for e in c['endpoints'].values()),
                'prediction_files': 450, 'prediction_freeze_sha256': sha(ROOT / 'private/PREDICTION_FREEZE.json'),
                'input_source_sha256': sha(source), 'annotation_inputs_sha256': sha(ROOT / 'private/ANNOTATION_INPUTS.json'),
                'protocol_sha256': sha(ROOT / 'PROTOCOL.md'),
                'endpoint_documents': len(docs), 'available_documents': sum(d['status']=='available' for d in docs),
                'human_reviewed_cases': 0, 'reference_status': 'unannotated'}
    freeze(ROOT / 'private/MANIFEST.json', manifest)
    print(json.dumps(manifest, indent=2))
