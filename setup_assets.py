"""Fetch public benchmarks and tokenizer resources; optionally fetch TESS code."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parent
TESS_REVISION = '6e4ebff3cf15228f6827a6f2bdb48978c2cbc790'

def fetch(url, target, expected=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and (expected is None or hashlib.sha256(target.read_bytes()).hexdigest() == expected):
        return
    data = urllib.request.urlopen(url, timeout=120).read()
    if expected and hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('Public asset checksum differs: '+target.name)
    target.write_bytes(data)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tess',action='store_true',help='also fetch the pinned public TESS sdlm package')
    args=ap.parse_args()
    folder=ROOT/'assets/ifeval'
    for name, meta in json.loads((folder/'sources.json').read_text())['files'].items():
        fetch(meta['url'],folder/name,meta['sha256'])
    info=json.loads((ROOT/'assets/ifbench/sources.json').read_text())
    fetch('https://raw.githubusercontent.com/allenai/IFBench/'+info['revision']+'/data/IFBench_test.jsonl',ROOT/'assets/ifbench/data/IFBench_test.jsonl',info['data_sha256'])
    import nltk
    target=ROOT/'assets/nltk'
    target.mkdir(parents=True,exist_ok=True)
    for resource in ('punkt','punkt_tab','stopwords','averaged_perceptron_tagger_eng','words','cmudict'):
        if not nltk.download(resource,download_dir=str(target),quiet=True):
            raise RuntimeError('Could not fetch NLTK resource: '+resource)
    # The sentence tokenizer used for official IFEval scoring is content-pinned.
    for name, wanted in json.loads((ROOT/'assets/scoring_assets.json').read_text())['nltk_english'].items():
        if hashlib.sha256((target/name).read_bytes()).hexdigest()!=wanted:
            raise ValueError('Pinned NLTK data differs: '+name)
    if args.tess:
        url='https://codeload.github.com/hamishivi/tess-2/tar.gz/'+TESS_REVISION
        data=urllib.request.urlopen(url,timeout=120).read()
        with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
            for member in archive.getmembers():
                parts=Path(member.name).parts[1:]
                if not parts or not member.isfile() or '..' in parts:
                    continue
                if parts[0] != 'sdlm' and parts != ('LICENSE',):
                    continue
                if '__pycache__' in parts or Path(parts[-1]).suffix in ('.pyc','.pyo'):
                    continue
                out=ROOT/'assets/tess'/Path(*parts)
                out.parent.mkdir(parents=True,exist_ok=True)
                out.write_bytes(archive.extractfile(member).read())
    print('Public assets are ready.')

if __name__=='__main__':
    main()
