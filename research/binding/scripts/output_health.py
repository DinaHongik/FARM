"""Check saved response completion without consulting semantic references."""
from common import read


def output_health(directory):
    calls = [read(path) for path in sorted((directory / 'cache').glob('*.json'))]
    records = [read(path) for path in sorted((directory / 'records').glob('*/*.json'))]
    result = {
        'unique_provider_calls': len(calls),
        'terminal_outputs': len(records),
        'length_stopped_calls': sum(c.get('done_reason') == 'length' for c in calls),
        'empty_content_calls': sum(not c.get('content', '').strip() for c in calls),
        'unparseable_calls': sum(c.get('parsed') is None for c in calls),
        'invalid_terminal_shapes': sum(not r['final_checks']['protocol_valid'] for r in records),
        'max_completion_tokens': max((c.get('completion_tokens', 0) for c in calls), default=0),
        'configured_output_limits': sorted({c['request']['options']['num_predict'] for c in calls}),
        'configured_thinking_modes': sorted({c['request'].get('think') for c in calls}, key=str),
        'thinking_enabled_calls': sum(c['request'].get('think') is not False for c in calls),
        'thinking_characters': sum(c.get('thinking_characters', 0) for c in calls),
    }
    result['status'] = 'passed' if (
        calls and records and not any(result[k] for k in [
            'length_stopped_calls', 'empty_content_calls', 'unparseable_calls',
            'invalid_terminal_shapes', 'thinking_enabled_calls', 'thinking_characters'])
    ) else 'failed'
    return result
