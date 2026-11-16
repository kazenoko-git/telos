"""Offline document suite export and score replay under telos eval."""
import argparse
import json
from pathlib import Path

from .runner import atomic_write, replay_responses
from .suite import load_suite, reference_responses


def main(argv=None):
    parser = argparse.ArgumentParser(prog='telos eval documents', description='Packaged document diagnostics: export tasks or replay saved responses offline.')
    commands = parser.add_subparsers(dest='command', required=True)
    export = commands.add_parser('export', help='Export the bundled cases, plans and manifest')
    export.add_argument('--output', type=Path, required=True)
    replay = commands.add_parser('replay', help='Recompute scores without loading a model or calling an API')
    source = replay.add_mutually_exclusive_group(required=True)
    source.add_argument('input', nargs='?', type=Path)
    source.add_argument('--reference', action='store_true', help='Use bundled historical normalized response snapshots')
    replay.add_argument('--track', choices=['direct_qa', 'extraction', 'calculator', 'all'])
    replay.add_argument('--output', type=Path, required=True)
    replay.add_argument('--html-report', type=Path)
    args = parser.parse_args(argv)
    if args.command == 'export':
        cases, plans, manifest = load_suite()
        atomic_write(args.output, {'manifest': manifest, 'cases': cases, 'plans': list(plans.values())})
        print(f'Exported {len(cases)} cases and plans to {args.output}')
        return
    if args.input and args.input.resolve() == args.output.resolve():
        parser.error('Replay output must differ from the source artifact')
    data = reference_responses() if args.reference else json.loads(args.input.read_text())
    result = replay_responses(data, tracks=args.track)
    atomic_write(args.output, result)
    for run in result['runs']:
        score = run['summary']
        print(f"{run['model']} | {run['track']} | {score['correct']}/{score['attempted']} | coverage {score['attempted']}/{score['expected']}")
    if args.html_report:
        from ..display import write_html_report
        write_html_report(result, args.html_report, title='Document evaluation', source=args.output.name)
    return result
