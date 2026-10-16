import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parent
URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
MD5 = 'c58f30108f718f92721af3b95e74349a'


def verify(root):
    expected = json.loads((ROOT/'configs/data_sha256.json').read_text())
    for name, digest in expected.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest, name


def download(destination):
    root = destination/'cifar-10-batches-py'
    if root.exists():
        verify(root)
        return
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination/'cifar-10-python.tar.gz'
    with urllib.request.urlopen(URL) as response, archive.open('xb') as handle:
        shutil.copyfileobj(response, handle)
    assert hashlib.md5(archive.read_bytes()).hexdigest() == MD5
    with tarfile.open(archive, 'r:gz') as handle:
        handle.extractall(destination, filter='data')
    verify(root)
    archive.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'data')
    download(parser.parse_args().output.expanduser().resolve())
