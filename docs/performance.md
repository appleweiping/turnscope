# Performance and scaling

TurnScope treats output size as part of the cost: returning a context containing `k` utterances necessarily costs
O(k). In v0.2, the built-in policies no longer rescan or re-index the full prefix for every target.

| Policy | Selection time across n targets | Additional state | Notes |
|---|---:|---:|---|
| `TurnWindowPolicy(k)` | O(nk), equal to emitted context size | O(k) | A bounded deque stores only the latest turns. |
| `TokenBudgetPolicy(b)` | O(n + emitted items) | O(items fitting b) | A contiguous suffix is maintained; an over-budget message becomes a permanent barrier. |
| `ReplyChainPolicy(d)` | O(n + emitted ancestors) | O(unique IDs) | The last prior utterance for each ID is indexed once. |
| `TimeWindowPolicy(t)` | O(n + emitted items) for chronological input | O(n) | A moving boundary avoids prefix scans. |
| Custom v0.1 policy | Policy-defined | O(n) | The compatibility path still supplies the complete prior sequence. |

For a conversation with out-of-order timestamps, the time policy deliberately falls back to scanning prior positions.
This preserves v0.1 semantics: a later backward timestamp can make an earlier message eligible again. Run the chronology
audit before large builds if linear time-policy behavior matters. Token costs are cached on retained policy-state
entries, not in a conversation-wide text cache. This computes an utterance at most once while allowing bounded turn and
token states to release evicted text. Custom counters must be deterministic, as required by `TokenCounter`.

`ContextBuilder.iter_build()` does not materialize the input iterable or the output windows. Turn windows use bounded
memory; token windows retain only the maximal potentially selectable suffix; reply chains retain an ID index. Time
windows retain prior messages to preserve correct fallback behavior for chronology violations.

## Reproducible benchmark

Install the checkout, then run:

```bash
python benchmarks/benchmark_builder.py --policy turn --value 8 --sizes 1000 10000 100000 --repeats 5
```

The script creates deterministic, fixed-width messages before timing, warms up once, fully consumes lazy results, and
reports median wall-clock time plus a checksum. Results are JSON so separate runs can be archived and compared. Wall
clock measurements depend on Python, CPU, power state, and background load; treat them as local regression evidence,
not universal throughput claims.

A reference run on 2026-08-31 used CPython 3.14.5 on Windows with an Intel Core i5-1240P. For 10,000 messages, value 8,
and five repetitions, the measured medians were:

| Policy | Median | Messages/second |
|---|---:|---:|
| turn | 0.074831 s | 133,635 |
| token | 0.065142 s | 153,511 |
| time, chronological | 0.103640 s | 96,488 |
| reply chain | 0.181635 s | 55,055 |

These values describe one run, not a performance guarantee. Keep the JSON output, Python version, command arguments,
and hardware description when recording a new baseline.

Generate a repeatable JSONL corpus without retaining the whole dataset:

```bash
python benchmarks/generate_data.py benchmark.jsonl --conversations 100 --utterances 1000 --seed 20260831
```

The generator refuses to overwrite an existing path unless `--force` is supplied. Given the same arguments and
TurnScope version, it emits byte-identical content.
