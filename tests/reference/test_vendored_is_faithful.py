"""Prove the vendored reference still matches upstream, function by function.

The oracle's value rests entirely on being an independent implementation. A
well-meant tidy-up — renaming a variable, inlining a helper, "fixing" what looks
like a bug — would quietly turn it into a second copy of our own opinions. This
test makes that impossible to do by accident.

It needs the upstream sdist, so it carries the `network` marker and is skipped
in the offline loop.
"""

import ast
import io
import tarfile
import urllib.request
from pathlib import Path

import pytest

pytestmark = pytest.mark.network

SDIST = "https://files.pythonhosted.org/packages/05/9c/6270e00769ca8fae3fcd0e1d46c33f1b0a155b83414bb40a0c5b11be71f2/apls-0.1.0.tar.gz"

VENDORED = {
    "apls_reference.py": "apls-0.1.0/apls/apls.py",
    "apls_utils.py": "apls-0.1.0/apls/apls_utils.py",
}


def _functions(source: str) -> dict[str, str]:
    """Map function name to its exact source text."""
    tree = ast.parse(source)
    lines = source.splitlines()
    return {
        node.name: "\n".join(lines[node.lineno - 1 : node.end_lineno]).rstrip()
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


@pytest.fixture(scope="module")
def upstream() -> dict[str, str]:
    with urllib.request.urlopen(SDIST, timeout=60) as response:
        raw = response.read()
    out = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        for member in VENDORED.values():
            handle = archive.extractfile(member)
            assert handle is not None, member
            out[member] = handle.read().decode()
    return out


@pytest.mark.parametrize(("local_name", "upstream_name"), sorted(VENDORED.items()))
def test_vendored_functions_are_byte_for_byte(upstream, local_name, upstream_name):
    ours = _functions((Path(__file__).parent / local_name).read_text())
    theirs = _functions(upstream[upstream_name])

    assert ours, f"no functions parsed from {local_name}"
    missing = sorted(set(ours) - set(theirs))
    assert not missing, f"{local_name} defines functions absent upstream: {missing}"

    for name, source in sorted(ours.items()):
        assert source == theirs[name], f"{local_name}::{name} diverges from upstream"


def test_only_the_shims_are_ours():
    """apls_plots.py is deliberately not upstream; it must stay a no-op stub."""
    from tests.reference import apls_plots

    assert apls_plots.plot_metric(object(), object(), scatter_png="x.png") is None
