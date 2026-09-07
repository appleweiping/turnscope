# TurnScope plugins

TurnScope discovers optional extensions through Python packaging entry points.
Nothing is imported unless the user explicitly asks for a plugin, so the core
install remains dependency-free.

Use these groups in a plugin distribution:

```toml
[project.entry-points."turnscope.tokenizers"]
my_tokenizer = "my_package:token_count"

[project.entry-points."turnscope.audit_rules"]
my_rule = "my_package:MyRule"
```

A tokenizer is a callable accepting text and returning a non-negative integer.
An audit rule is an object (or zero-argument class) exposing a unique non-empty
`name` and `check(conversation) -> tuple[Issue, ...]`.

Inspect installed extensions without importing them:

```console
$ turnscope plugins
[{"kind": "rules", "name": "my_rule", "value": "my_package:MyRule"}, ...]
```

Use an extension explicitly with `turnscope build --tokenizer-plugin NAME`,
`turnscope audit --tokenizer-plugin NAME`, or repeat
`turnscope audit --rule-plugin NAME`. Plugin code runs in the current process;
install only trusted distributions and pin them in reproducible environments.
