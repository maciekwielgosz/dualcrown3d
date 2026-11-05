"""Remap absolute paths stored in manifests and protocols to another machine.

Manifests and frozen protocols hold absolute paths of the workstation they were
written on. Set ``DUALCROWN_PATH_MAP`` to ``old_prefix=new_prefix`` (several
pairs separated by ``;``) to run the same code from a copy elsewhere, e.g. on a
cluster. Without the variable every function is the identity.
"""
from __future__ import annotations

import os


def remap(path):
    text = str(path)
    for pair in filter(None, os.environ.get('DUALCROWN_PATH_MAP', '').split(';')):
        old, new = pair.split('=', 1)
        if text.startswith(old):
            return new + text[len(old):]
    return text


def remap_row(row):
    """Manifest row with every absolute-path value remapped."""
    if not os.environ.get('DUALCROWN_PATH_MAP'):
        return row
    return {key: remap(value) if isinstance(value, str) and value.startswith('/') else value for key, value in row.items()}
