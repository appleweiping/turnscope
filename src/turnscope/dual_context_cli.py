"""Bounded private-data boundaries for shared dual-context model commands."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TextIO

from .dual_context import ContextEdge, ContextRecord, DualContextConfig, DualContextModel
from .dual_context_evaluation import evaluate_dual_context
from .io import parse_json_value

MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_LINE_BYTES = 1024 * 1024
MAX_OUTPUT_BYTES = 64 * 1024 * 1024
MAX_RECORDS = 30_000
MAX_EDGES = 60_000
RECORD_FIELDS = frozenset(("conversation_id", "utterance_id", "text"))
EDGE_FIELDS = frozenset(("conversation_id", "source_id", "context_id"))


def _read_rows(path: Path, fields: frozenset[str], maximum: int) -> Iterator[dict[str, str]]:
    """Read bounded, closed string-only JSONL without including text in errors."""
    if not path.is_file():
        raise ValueError("dual-context input must be a regular file")
    if path.stat().st_size > MAX_INPUT_BYTES:
        raise ValueError("dual-context input exceeds the byte limit")
    consumed = 0
    with path.open("rb") as stream:
        for index in range(maximum + 1):
            raw = stream.readline(MAX_LINE_BYTES + 1)
            if not raw:
                return
            consumed += len(raw)
            if len(raw) > MAX_LINE_BYTES or consumed > MAX_INPUT_BYTES:
                raise ValueError("dual-context input exceeds a byte limit")
            if index == maximum:
                raise ValueError("dual-context input exceeds the record limit")
            try:
                value = parse_json_value(raw.decode("utf-8"))
            except (ValueError, UnicodeError, RecursionError):
                raise ValueError(f"invalid dual-context JSONL record at line {index + 1}") from None
            if not isinstance(value, dict) or set(value) != fields:
                raise ValueError(f"invalid dual-context record fields at line {index + 1}")
            if any(not isinstance(item, str) for item in value.values()):
                raise ValueError(f"dual-context record values must be strings at line {index + 1}")
            yield value


def _encode(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode(
        "utf-8"
    )


def _encode_rows(header: dict[str, object], rows: Iterable[object]) -> bytes:
    """Admit each complete row before extending the bounded aggregate output."""
    if "rows" in header:
        raise ValueError("rows is reserved for the bounded output array")
    result = bytearray(_encode(header)[:-1])
    result.extend(b',"rows":[' if header else b'"rows":[')
    if len(result) + 3 > MAX_OUTPUT_BYTES:
        raise ValueError("dual-context output exceeds the byte limit")
    first = True
    for row in rows:
        encoded = _encode(row)
        addition = len(encoded) + (0 if first else 1)
        if len(result) + addition + 3 > MAX_OUTPUT_BYTES:
            raise ValueError("dual-context output exceeds the byte limit")
        if not first:
            result.extend(b",")
        result.extend(encoded)
        first = False
    result.extend(b"]}\n")
    return bytes(result)


def _check_output(output: Path | None, protected: Iterable[Path]) -> None:
    if output is None:
        return
    destination = output.resolve()
    for source in protected:
        if source.resolve() == destination or (output.exists() and output.samefile(source)):
            raise ValueError("dual-context output must differ from every input")
    if output.exists() or output.is_symlink():
        raise ValueError("dual-context output must be a new file")
    if not output.parent.is_dir():
        raise ValueError("dual-context output parent must already exist")


def _publish(data: bytes, output: Path | None) -> None:
    """Publish a complete report exclusively; never replace an existing destination."""
    if len(data) > MAX_OUTPUT_BYTES:
        raise ValueError("dual-context output exceeds the byte limit")
    if output is None:
        try:
            binary = getattr(sys.stdout, "buffer", None)
            if binary is None:
                text = data.decode("utf-8")
                written = sys.stdout.write(text)
                if type(written) is not int or written != len(text):
                    raise OSError("incomplete dual-context stdout publication")
                sys.stdout.flush()
            else:
                written = binary.write(data)
                if type(written) is not int or written != len(data):
                    raise OSError("incomplete dual-context stdout publication")
                binary.flush()
        except (OSError, ValueError):
            _silence_failed_stream(sys.stdout)
            raise
        return
    _check_output(output, ())
    temporary: Path | None = None
    published = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=".turnscope-dual-", suffix=".tmp", dir=output.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
        published = True
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except OSError:
                if not published:
                    raise
                _diagnostic("report published; private temporary-file cleanup failed")


def _silence_failed_stream(stream: TextIO) -> None:
    # CPython flushes again at shutdown. After a failed optional summary, stop
    # the broken descriptor from changing successful model publication to 120.
    try:
        descriptor = stream.fileno()
        null = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(null, descriptor)
        finally:
            os.close(null)
    except (OSError, ValueError, AttributeError):
        return


def _diagnostic(message: str) -> None:
    try:
        sys.stderr.write(f"turnscope dual-context: {message}\n")
        sys.stderr.flush()
    except (OSError, ValueError):
        _silence_failed_stream(sys.stderr)


def configure_dual_context_parser(parser: argparse.ArgumentParser) -> None:
    actions = parser.add_subparsers(dest="dual_context_action", required=True)
    fit = actions.add_parser("fit", help="fit one shared context space on explicit training edges")
    fit.add_argument("catalog", type=Path)
    fit.add_argument("forward", type=Path)
    fit.add_argument("backward", type=Path)
    fit.add_argument("model", type=Path)
    defaults = DualContextConfig()
    for name, value in defaults.to_dict().items():
        if name == "drop_first":
            fit.add_argument("--drop-first", action="store_true")
        else:
            fit.add_argument("--" + name.replace("_", "-"), type=int, default=value)
    transform = actions.add_parser("transform", help="project new catalog text without refitting")
    transform.add_argument("model", type=Path)
    transform.add_argument("catalog", type=Path)
    transform.add_argument("--output", "-o", type=Path)
    terms = actions.add_parser(
        "terms", help="export frozen term statistics and cluster assignments"
    )
    terms.add_argument("model", type=Path)
    terms.add_argument("--output", "-o", type=Path)
    evaluate = actions.add_parser("evaluate", help="evaluate explicit fixed-pool relevance")
    for name in ("model", "queries", "candidates", "edges"):
        evaluate.add_argument(name, type=Path)
    evaluate.add_argument("--direction", choices=("forward", "backward"), required=True)
    evaluate.add_argument("--k", action="append", type=int)
    for name, default in (
        ("max_queries", 1024),
        ("max_candidates", 4096),
        ("max_score_pairs", 1_000_000),
        ("max_relationships", 60_000),
        ("max_text_bytes", 32 * 1024 * 1024),
    ):
        evaluate.add_argument("--" + name.replace("_", "-"), type=int, default=default)
    evaluate.add_argument("--output", "-o", type=Path)


def _records(path: Path, maximum: int = MAX_RECORDS) -> Iterator[ContextRecord]:
    seen = set()
    for row in _read_rows(path, RECORD_FIELDS, maximum):
        record = ContextRecord(**row)
        if record.key in seen:
            raise ValueError("dual-context catalog contains duplicate composite IDs")
        seen.add(record.key)
        yield record


def _relationships(path: Path, maximum: int = MAX_EDGES) -> Iterator[ContextEdge]:
    for row in _read_rows(path, EDGE_FIELDS, maximum):
        yield ContextEdge(**row)


def _fit(args: argparse.Namespace) -> None:
    protected = (args.catalog, args.forward, args.backward)
    _check_output(args.model, protected)
    model = DualContextModel(
        **{name: getattr(args, name) for name in DualContextConfig().to_dict()}
    )
    model.fit(
        _records(args.catalog, min(MAX_RECORDS, model.config.max_catalog_records)),
        _relationships(args.forward, min(MAX_EDGES, model.config.max_edges_per_direction)),
        _relationships(args.backward, min(MAX_EDGES, model.config.max_edges_per_direction)),
    )
    payload = (
        _encode(
            {
                "format": "turnscope.dual-context.fit.v1",
                "private_data": True,
                "model_digest": model.digest,
                "training": model.training_summary(),
            }
        )
        + b"\n"
    )
    warning = model.save(args.model)
    if warning is not None:
        _diagnostic(warning)
    try:
        _publish(payload, None)
    except (OSError, ValueError):
        _diagnostic("model saved; stdout summary could not be delivered")


def _transform_rows(model: DualContextModel, path: Path) -> Iterator[object]:
    for record in _records(path):
        yield {
            "conversation_id": record.conversation_id,
            "utterance_id": record.utterance_id,
            "prediction": model.predict(record.text).to_dict(),
            "context": model.project_context(record.text).to_dict(),
        }


def run_dual_context_command(args: argparse.Namespace) -> int:
    """Run only local model work; no downloads, plugins, remote calls or overwrite."""
    try:
        if args.dual_context_action not in ("fit", "transform", "terms", "evaluate"):
            raise ValueError("unknown dual-context command")
        if args.dual_context_action == "fit":
            _fit(args)
            return 0
        protected = [args.model]
        if args.dual_context_action == "transform":
            protected.append(args.catalog)
        elif args.dual_context_action == "evaluate":
            protected.extend((args.queries, args.candidates, args.edges))
        _check_output(args.output, protected)
        model = DualContextModel.load(args.model)
        before = model.digest
        if args.dual_context_action == "evaluate":
            report = evaluate_dual_context(
                model,
                _records(args.queries),
                _records(args.candidates),
                _relationships(args.edges),
                direction=args.direction,
                ks=tuple(args.k) if args.k is not None else (1, 5, 10),
                **{
                    name: getattr(args, name)
                    for name in (
                        "max_queries",
                        "max_candidates",
                        "max_score_pairs",
                        "max_relationships",
                        "max_text_bytes",
                    )
                },
            )
            data = _encode_rows(
                {key: value for key, value in report.items() if key != "rows"}, report["rows"]
            )
        else:
            rows = (
                _transform_rows(model, args.catalog)
                if args.dual_context_action == "transform"
                else model.term_statistics()
            )
            data = _encode_rows(
                {
                    "format": f"turnscope.dual-context.{args.dual_context_action}.v1",
                    "private_data": True,
                    "model_digest": before,
                },
                rows,
            )
        if model.digest != before:
            raise ValueError("frozen model changed during transformation")
        _publish(data, args.output)
        return 0
    except (OSError, ValueError, ImportError, RecursionError):
        _diagnostic("command failed; check input schema, paths, model and configured limits")
        return 2
