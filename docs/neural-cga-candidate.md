# Fixed private CGA candidate fitting

`benchmarks/benchmark_neural_forecast.py` is deliberately a **candidate-fitting
stage**, not a complete quality benchmark. It uses the [pinned local source and
label-independent validation partition](neural-cga-protocol.md). It emits no
official-test model predictions or scores. Schema/eligibility auditing of the
entire supplied archive is recorded separately from fitting.

```bash
python benchmarks/benchmark_neural_forecast.py /private/cga-wiki.zip /private/new-candidate --seed 17
```

Use an already installed CPU training environment. The script neither installs
dependencies nor downloads models or datasets. The output directory must be new
and outside this checkout; all vocabulary/weights stay private there. A failed
run's directory is retained for diagnosis, not automatically retried or resumed.

## Frozen first-stage protocol

The only candidate seeds are 17, 101 and 202. They are declared before neural
test outcomes, not selected according to later test scores. Each seed uses the
same default 64-dimensional embedding/word/turn model, one recurrent layer at
each level, 128 ordinary-token head cap plus EOS, training-only 10,000-term
vocabulary, conversation-weighted prefix objective, Adam learning rate 0.001,
gradient clip 5, up to eight epochs, patience three and at most 16 conversations
per batch. All ordinary source/numerical/work/workspace admission limits remain
in force. Over-budget data fail; the driver does not silently downsample or
increase limits after a failure.

Official train supplies gradients/vocabulary. Whole connected components from
official validation are split once into epoch-selection and threshold-selection
partitions. The best epoch uses earliest minimum validation loss; the alert
threshold uses only separate policy validation. The official test cohort has
already been inspected by earlier lexical/context evaluations: later results
must be described as a fixed new-model evaluation on an established benchmark,
not a newly collected or previously unseen research test cohort.

For this explicit benchmark process, Torch CPU thread count is set to one and
deterministic algorithms are required during fitting. Previous thread and
determinism/warn-only settings are restored on exit, including a fitting failure.
This is not a package-wide change: the public training API still leaves the
caller's global policies unchanged. No BLAS-thread, OS scheduling, hard RSS or
cross-platform bitwise identity guarantee is implied.

## Evidence and failure stages

Before reading the source, the script writes `fit-plan.json` with the exact
protocol and SHA-256 bindings for installed runtime sources and the two shipped
benchmark helpers. `source-audit.json` retains source/member identities, group
partitions, exclusions and prepared supervision support; no raw dialogue or
per-conversation predictions are serialized there.

After fitting it checks that bound sources did not change, publishes
`private-model.tsn`, and reloads the file through the closed artifact validator.
The restored model digest and full training summary must match. Only then is
`candidate-receipt.json` written with training evidence, model/file digests,
runtime versions, source-load and fit timings, and explicit flags:

```json
{
  "candidate_fit_completed": true,
  "final_evaluation_completed": false,
  "official_test_predictions_produced": false,
  "real_data_quality_claimed": false,
  "whole_repository_parity_claimed": false
}
```

Source-load timing excludes the subsequent prepared audit. Fit timing includes
preparation, CPU policy setup/restoration, optimization and policy selection, but
excludes source loading and model publication/reload. These are wall-clock
durations, not throughput guarantees. No process-memory measurement is claimed.

If fitting fails, the plan/audit can remain but no success receipt is written.
If saving succeeds and later reload/report delivery fails, the model can remain
even though the operation exits unsuccessfully. There is no directory-wide
transaction or external exactly-once promise. A postpublication cleanup warning
is retained in the model receipt. Hashes provide integrity links, not trusted
proof that a claimed source's human labels are correct.

## Remaining quality work

A completed receipt is only a frozen candidate. Later acceptance still requires
the same-token cumulative NB and eligible-training-prior baselines, each with its
own policy-selected threshold; separately trained current-turn-only,
mean-word-encoding and order-erased variants; and a prefix-safe heldout ordering
intervention reported as an out-of-distribution diagnostic, not retraining.
All protocols must be fixed before those test scores are read. The first-stage
receipt lists these remaining comparisons; it does not imply they are already
implemented or evaluated. Three declared seeds are needed before making a
variability/robust-quality claim. Retain adverse probability-quality results,
truncation/OOV coverage, exclusions and each metric's denominator.

Even complete comparison of this forecasting family would not establish full
ConvoKit repository parity: pretrained transformer training, simulation/policy
families and other functionality remain separate work.
