"""Pin these tests to THIS checkout's `propab`, and make the fixtures importable by module path.

`propab` is installed editable against the primary working tree, so a plain `import propab` from a
git worktree silently resolves to the *other* checkout. Prepending here makes the tests exercise the
code that sits next to them. The same sys.path is handed to the auditor's fresh re-check process,
which is how `audit_targets` becomes importable there.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent          # packages/propab-core/tests/evolve/audit
_PKG_ROOT = _HERE.parents[2]                     # packages/propab-core

for _path in (str(_PKG_ROOT), str(_HERE)):
    if _path in sys.path:
        sys.path.remove(_path)
    sys.path.insert(0, _path)
