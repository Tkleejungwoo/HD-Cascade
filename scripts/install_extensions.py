"""Install custom modules into a dedicated nnU-Net 2.6.2 environment."""

import argparse
from importlib.metadata import version
from pathlib import Path
import shutil


SOURCE_ROOT = Path(__file__).resolve().parents[1] / 'extensions' / 'nnunetv2'


def install(target_root: Path, check_only: bool = False):
    target_root = target_root.resolve()
    if not (target_root / '__init__.py').is_file():
        raise ValueError(f'Not an nnunetv2 package directory: {target_root}')
    plan = []

    for source in sorted(SOURCE_ROOT.rglob('*.py')):
        destination = target_root / source.relative_to(SOURCE_ROOT)
        
        if destination.exists():
            if destination.read_bytes() != source.read_bytes():
                raise FileExistsError(f'Existing different file; nothing installed: {destination}')
            continue
        plan.append((source, destination))
    
    for source, destination in plan:
        print(f'{"Would install" if check_only else "Install"}: {source.relative_to(SOURCE_ROOT)}')
        
        if not check_only:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    print(f'{len(plan)} module(s) {"pending" if check_only else "installed"}; identical modules preserved')
    return len(plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Print changes without writing')
    args = parser.parse_args()
    found = version('nnunetv2')
    
    if found != '2.6.2':
        parser.error(f'Requires nnunetv2==2.6.2, found {found}')
    
    import nnunetv2
    install(Path(nnunetv2.__file__).resolve().parent, args.check)


if __name__ == '__main__':
    main()
