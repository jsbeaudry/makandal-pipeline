"""Upload a generated instruction set to the private dataset repo."""
import os
import sys

from huggingface_hub import HfApi

REPO = os.environ.get('SFT_REPO', 'jsbeaudry/kreyol-sft')


def main(path, name):
    api = HfApi(token=os.environ['HF_TOKEN'])
    api.create_repo(REPO, repo_type='dataset', private=True, exist_ok=True)
    api.upload_file(path_or_fileobj=path, path_in_repo=name, repo_id=REPO, repo_type='dataset')
    print(f'[pod] uploaded {name} to {REPO}', flush=True)


if __name__ == '__main__':
    source = sys.argv[1] if len(sys.argv) > 1 else '/workspace/sft.jsonl'
    main(source, os.environ.get('SFT_NAME', 'sft-30k.jsonl'))
