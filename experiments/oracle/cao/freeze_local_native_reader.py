"""Re-render published reader inputs for a pinned local model before new gold."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def freeze(source, tokenizer_path, model_info, runner, out):
    from transformers import AutoTokenizer
    assert not out.exists()
    original = json.loads((source / 'manifest.json').read_text())
    assert sha(source / 'requests.jsonl') == original['requests_sha256']
    assert sha(source / 'mapping.json') == original['mapping_sha256']
    old_tok_path = Path('/Users/user/.cache/huggingface/hub/models--Qwen--Qwen3.5-397B-A17B/snapshots/8472618112abcbd45acbcdc58436aff4233c23f7')
    assert all(sha(old_tok_path / name) == value for name, value in original['tokenizer_sha256'].items())
    old_tok = AutoTokenizer.from_pretrained(str(old_tok_path), local_files_only=True)
    tok = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    info = json.loads(model_info.read_text())
    assert tokenizer_path.name == info['revision']
    prefix = '<|im_start|>user\n'
    suffix = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
    rows = [json.loads(line) for line in (source / 'requests.jsonl').read_text().splitlines()]
    assert len(rows) == 96
    unchanged = 0
    for row in rows:
        text = old_tok.decode(row['token_ids'], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        assert old_tok.encode(text, add_special_tokens=False) == row['token_ids']
        assert text.startswith(prefix) and text.endswith(suffix)
        body = text[len(prefix):-len(suffix)]
        assert '<|im_start|>' not in body and '<|im_end|>' not in body
        ids = tok.apply_chat_template([{'role': 'user', 'content': body}], tokenize=True,
                                     add_generation_prompt=True, enable_thinking=False, return_dict=False)
        assert isinstance(ids, list) and all(type(value) is int for value in ids)
        unchanged += ids == row['token_ids']
        row['source_user_text_sha256'] = hashlib.sha256(body.encode()).hexdigest()
        row['token_ids'] = ids
    pad = tok.encode(' ?', add_special_tokens=False)
    assert len(pad) == 1
    for offset in range(0, 96, 3):
        arms = rows[offset:offset + 3]
        assert len({row['question_id'] for row in arms}) == 1
        values = [row['token_ids'] for row in arms]
        shared = 0
        while all(len(v) > shared and v[-shared - 1] == values[0][-shared - 1] for v in values):
            shared += 1
        assert shared > 100
        length = max(map(len, values))
        for row in arms:
            ids = row['token_ids']
            row['token_ids'] = ids[:-shared] + pad * (length - len(ids)) + ids[-shared:]
            row['prompt_tokens'] = length
            row['common_suffix_tokens'] = shared
            assert len(row['token_ids']) == length and length + 512 <= 32768
    out.mkdir()
    (out / 'requests.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    (out / 'mapping.json').write_bytes((source / 'mapping.json').read_bytes())
    manifest = {key: value for key, value in original.items() if key not in {
        'model', 'project', 'rates', 'cost_upper_usd', 'prices_sha256', 'tokenizer_sha256', 'runner_sha256', 'scope'}}
    model_hashes = {name: value['sha256'] for name, value in info['files'].items() if name.endswith('.safetensors')}
    assert model_hashes and all(model_hashes.values())
    for path in [tokenizer_path / 'config.json', *tokenizer_path.glob('*.index.json')]:
        model_hashes[path.name] = sha(path)
    token_hashes = {path.name: sha(path) for path in tokenizer_path.iterdir()
                    if path.is_file() and path.suffix in ('.json', '.jinja', '.txt')}
    manifest.update(schema='native_local_mlx_reader_v1', model=info['model'], model_revision=info['revision'],
                    backend='mlx', quantization='4bit', context_cap=32768, api_calls=0, api_spend_usd=0,
                    tokenizer_path=str(tokenizer_path), tokenizer_sha256=token_hashes,
                    model_files_sha256=model_hashes, runner_sha256=sha(runner), freezer_sha256=sha(__file__),
                    source_frozen_manifest_path=str(source / 'manifest.json'),
                    source_frozen_manifest_sha256=sha(source / 'manifest.json'),
                    source_requests_sha256=original['requests_sha256'],
                    requests_sha256=sha(out / 'requests.jsonl'), mapping_sha256=sha(out / 'mapping.json'),
                    source_chats_identical_tokens_after_local_render=unchanged,
                    scope='New local quantized 35B reader experiment authorized for Mac GPU. Not the unexecuted 397B API experiment. Same original 8B target, 32 questions, original native observations, old facts, three arms and numerical baselines. All first greedy answers and failures retained; no new gold before prediction publication. Quantization/backend/model changes limit comparison with other readers.')
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'requests': len(rows), 'unchanged_token_sequences': unchanged,
                      'manifest_sha256': sha(out / 'manifest.json'), 'prefill_tokens': sum(r['prompt_tokens'] for r in rows)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('source', 'tokenizer_path', 'model_info', 'runner', 'out'):
        parser.add_argument('--' + key.replace('_', '-'), type=Path, required=True)
    args = parser.parse_args()
    freeze(**vars(args))
