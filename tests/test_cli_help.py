"""Every `cli.py <cmd> --help` must render (a bare % in argparse text crashes it)."""
import argparse

import cli


def _all_parsers(parser, seen=None):
    seen = [] if seen is None else seen
    seen.append(parser)
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for sub in action.choices.values():
                _all_parsers(sub, seen)
    return seen


def test_help_renders_for_top_parser_and_every_subparser():
    parsers = _all_parsers(cli.build_parser())
    assert len(parsers) > 1
    for p in parsers:
        assert p.format_help()
