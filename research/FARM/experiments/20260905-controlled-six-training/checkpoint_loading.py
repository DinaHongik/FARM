"""Preserve the trained bidirectional window across Transformers 4.57.6 reloads."""
import json
from pathlib import Path

RELOAD_SENTINELS = [
    'When a new item appears, archive it',
    'When a new photo appears, save the original image to the selected folder. '
    + 'weather sunshine rainfall wind calendar message document temperature destination ' * 40,
]


def saved_config_overrides(path):
    if path is None:
        return {}
    value=json.loads((Path(path)/'config.json').read_text())
    if value.get('model_type')=='gemma3_text' and value.get('use_bidirectional_attention'):
        # Gemma3TextConfig.__init__ halves a serialized runtime window again.
        # from_pretrained applies this explicit override AFTER __init__, before
        # constructing the model/attention layers. Never mutate the checkpoint.
        return {'sliding_window':value['sliding_window']}
    return {}


def assert_preserved_window(model, path=None):
    expected=saved_config_overrides(path).get('sliding_window',257)
    config=model[0].auto_model.config
    if config.model_type=='gemma3_text' and config.use_bidirectional_attention:
        assert config.sliding_window==expected,(config.sliding_window,expected)
        for module in model[0].auto_model.modules():
            if getattr(module,'is_sliding',False):
                assert module.sliding_window==expected


def test_serialized_window_override_avoids_second_halving():
    import tempfile
    from transformers import AutoConfig
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)
        (path/'config.json').write_text(json.dumps(dict(model_type='gemma3_text',
             use_bidirectional_attention=True,sliding_window=257)))
        broken=AutoConfig.from_pretrained(path)
        restored=AutoConfig.from_pretrained(path,**saved_config_overrides(path))
        assert broken.sliding_window==129
        assert restored.sliding_window==257
