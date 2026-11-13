"""Extract a version's release notes without including later changelog entries."""
import re
import sys
from pathlib import Path


def release_notes(changelog: str, tag: str) -> str:
    version = tag.removeprefix('v')
    if not re.fullmatch(r'\d+\.\d+\.\d+', version):
        raise ValueError(f'Expected a stable vMAJOR.MINOR.PATCH tag, got {tag}')
    match = re.search(rf'^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## |\Z)',
                      changelog, flags=re.MULTILINE | re.DOTALL)
    if not match:
        raise ValueError(f'No changelog entry for {tag}')
    body = re.split(r'^\[\w[^\n]*\]: ', match[1], maxsplit=1, flags=re.MULTILINE)[0].strip()
    if not body:
        raise ValueError(f'Empty changelog entry for {tag}')
    return body + '\n'


if __name__ == '__main__':
    print(release_notes(Path('CHANGELOG.md').read_text(), sys.argv[1]), end='')
