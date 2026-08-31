# Contributing

Open an issue before making a behavior or public-API change. Keep changes focused and include a regression test for
defects. New policies and rules must document their ordering, boundary, and malformed-input behavior.

Set up a development environment with `python -m pip install -e ".[dev]"`, then run:

```bash
ruff check .
ruff format --check .
mypy src
pytest
```

Pull requests should explain the user-visible contract, list commands actually run, update relevant documentation, and
avoid unrelated formatting. By participating, you agree to follow the code of conduct.
