"""Recover exact source excerpts from formatting-only citation differences."""
import copy
import re

def strings(value):
    if isinstance(value, str): yield value
    elif isinstance(value, dict):
        for item in value.values(): yield from strings(item)
    elif isinstance(value, list):
        for item in value: yield from strings(item)

def exact_excerpt(quote, corpus):
    for text in corpus:
        if quote in text: return quote
    # Some JSON-producing annotators wrap the citation in another pair of quote
    # characters or double-escape its newlines. Decode citation formatting only;
    # the replacement must still occur verbatim in an archived source.
    if len(quote) >= 2 and (quote[0], quote[-1]) in [('"','"'), ("'","'"), ('“','”'), ('‘','’')]:
        quote = quote[1:-1]
    quote = quote.replace('\\n', '\n').replace('\\t', '\t')
    for text in corpus:
        if quote in text: return quote
    # Preserve all words, their order, case, and numbers. Allow line wrapping,
    # typographic quote variants, and label separators such as "Required: true".
    words = quote.split()
    patterns = []
    for word in words:
        word = word.removesuffix(':')
        part = ''.join("['‘’]" if c in "'‘’" else '[\"“”]' if c in '\"“”' else re.escape(c) for c in word)
        patterns.append(part)
    if not patterns: return None
    pattern = re.compile(r'[\s:]+'.join(patterns))
    for text in corpus:
        match = pattern.search(text)
        if match: return match.group(0)
    return None

def normalize_quotes(case, reference):
    result = copy.deepcopy(reference)
    changes = []
    if not isinstance(result, dict) or not isinstance(result.get('fields'), list): return result, changes
    corpus = list(strings({'query':case['query'], 'endpoints':case['endpoints'],
                           'documentation':case.get('official_documentation',{})}))
    for field in result['fields']:
        if not isinstance(field,dict) or not isinstance(field.get('evidence_quotes'),list): continue
        replaced = []
        for quote in field['evidence_quotes']:
            if not isinstance(quote,str): replaced.append(quote);continue
            pieces = [p.strip() for p in re.split(r'\.{3}|…',quote) if p.strip()]
            excerpts = [exact_excerpt(p,corpus) for p in pieces]
            if pieces and all(excerpts):
                replaced.extend(excerpts)
                if excerpts != [quote]:
                    changes.append({'side':field.get('side'),'field':field.get('field'),
                                    'original_quote':quote,'verified_excerpts':excerpts})
            else:
                replaced.append(quote)
        field['evidence_quotes'] = list(dict.fromkeys(replaced))
    return result, changes
