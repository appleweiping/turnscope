# Releasing

TurnScope follows semantic versioning. A release is made only from a clean, reviewed `main` commit whose required
checks pass.

1. Update the version in `pyproject.toml`, public version exports, `CHANGELOG.md`, and `CITATION.cff`.
2. Run `ruff check .`, `ruff format --check .`, `mypy src`, `pytest`, and `python -m build`.
3. Install the wheel into an empty environment and run `turnscope --help` plus a minimal build/audit workflow.
4. Add curated notes at `docs/releases/vX.Y.Z.md` when appropriate; the workflow falls back to generated notes when
   that file is absent.
5. Before the first release, a repository administrator must enable GitHub's immutable releases setting. Create a
   signed or GitHub-protected `vX.Y.Z` tag pointing at the reviewed commit.
6. The release workflow rebuilds distributions, runs `twine check`, records SHA-256 checksums, creates provenance,
   and publishes every asset in the release creation operation so repository-level immutability can lock them.
7. PyPI publication is a separate, explicitly approved workflow dispatch from the release tag using a configured
   trusted publisher. Long-lived PyPI API tokens are not accepted.

If any artifact or check is wrong, publish a new patch version; never replace an existing release artifact.
