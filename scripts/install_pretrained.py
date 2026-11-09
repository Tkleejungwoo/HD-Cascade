"""Download or install the verified HD Cascade release into nnUNet_results."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def download_archive(url, output):
    """Download a Google Drive shared file or an ordinary HTTP(S) archive."""
    if not isinstance(url, str) or not url.strip():
        raise ValueError('Set archive_url in pretrained/manifest.json or pass --url with the weights ZIP link')
    parsed = urllib.parse.urlparse(url)
    
    if parsed.scheme not in ('http', 'https'):
        raise ValueError('The weights URL must use HTTP or HTTPS')
    output = Path(output)
    print(f'Downloading {url}', flush=True)
    
    if parsed.hostname in ('drive.google.com', 'docs.google.com'):
        if '/folders/' in parsed.path:
            raise ValueError('Provide a Google Drive file link for the weights ZIP, not a folder link')
        try:
            import gdown
        except ImportError as exc:
            raise RuntimeError('Google Drive downloads require gdown; run shell/setup.shell to install dependencies') from exc
        
        # Shared-file links and large-file confirmation pages are handled by gdown.
        # Do not read browser cookies or write a cookie cache in the user's home.
        result = gdown.download(url=url, output=str(output), quiet=False, use_cookies=False)
        
        if result is None or not output.is_file():
            raise RuntimeError('Google Drive download failed; check the shared-file link and download permission')
    else:
        with urllib.request.urlopen(url, timeout=60) as response, output.open('wb') as stream:
            shutil.copyfileobj(response, stream)
    return output


def install_archive(archive_path, manifest, results_dir):
    archive_path, results_dir = Path(archive_path), Path(results_dir).expanduser().resolve()
    if sha256(archive_path) != manifest['archive_sha256']:
        raise ValueError('Archive SHA-256 does not match the release manifest')
    expected = {}
    
    for item in manifest['files']:
        name = PurePosixPath(item['path'])
        if name.is_absolute() or '..' in name.parts or not name.parts:
            raise ValueError(f'Invalid release path: {name}')
        if item['path'] in expected:
            raise ValueError(f'Duplicate release path: {name}')
        expected[item['path']] = item
    
    with tempfile.TemporaryDirectory(prefix='hd-cascade-install-') as tmp:
        staging = Path(tmp)
        with zipfile.ZipFile(archive_path) as archive:
            if len(archive.namelist()) != len(expected) or set(archive.namelist()) != set(expected):
                raise ValueError('Archive contents differ from the release manifest')
            
            for name, item in expected.items():
                info = archive.getinfo(name)
                
                if info.file_size != item['bytes']:
                    raise ValueError(f'Unexpected file size: {name}')
                target = staging / name
                target.parent.mkdir(parents=True, exist_ok=True)
                
                with archive.open(info) as source, target.open('wb') as destination:
                    shutil.copyfileobj(source, destination)
                
                if sha256(target) != item['sha256']:
                    raise ValueError(f'File SHA-256 mismatch: {name}')
        # Check all conflicts before copying; existing different models are preserved.
        pending = []
        for name, item in expected.items():
            destination = results_dir / name
            if destination.exists():
                if not destination.is_file() or sha256(destination) != item['sha256']:
                    raise FileExistsError(f'Existing different model file: {destination}')
            else:
                pending.append((staging / name, destination))
        
        for source, destination in pending:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    
    print(f'Pretrained model installed: {results_dir}', flush=True)
    return len(pending)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--archive', type=Path, help='Use an already downloaded weights ZIP')
    source.add_argument('--url', help='Override the weights URL; Google Drive file sharing links are supported')
    parser.add_argument('--results-dir', type=Path,
                        default=Path(os.environ.get('nnUNet_results', REPO_ROOT / 'data/nnUNet_results')))
    parser.add_argument('--manifest', type=Path, default=REPO_ROOT / 'pretrained/manifest.json')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    
    with tempfile.TemporaryDirectory(prefix='hd-cascade-download-') as tmp:
        archive = args.archive
        if archive is None:
            archive = Path(tmp) / 'weights.zip'
            try:
                download_archive(args.url or manifest.get('archive_url'), archive)
            except (urllib.error.URLError, TimeoutError, ValueError, RuntimeError) as exc:
                parser.exit(1, f'Download failed: {exc}. You can also download the ZIP in your browser and pass --archive PATH.\n')
        install_archive(archive, manifest, args.results_dir)
    print(f'Model: {manifest["dataset"]}; trainer: {manifest["trainer"]}; fold: {manifest["fold"]}', flush=True)


if __name__ == '__main__':
    main()
