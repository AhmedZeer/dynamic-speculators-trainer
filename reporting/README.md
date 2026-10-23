# Dynamic Speculators — evolving experimental report

Update this report **only when explicitly requested**. Markdown under
`docs/experiments/` remains the working experiment record. There is no daily
generation, scheduled task, or automatic reporting pipeline.

Read [Version 02](reports/dynamic-speculators-2026-10-23-v02.pdf), with evidence
through 2026-10-23. Version 01 remains preserved in [`reports/`](reports/).
This update expands the generator architecture, adds both supplied training
figures, and records the decision to retire bank warmup in favor of direct
training on the original examples. It records observations and execution status, with no final
conclusions or overall model ranking.

## Organization

- `main.tex`: ICML 2026 preprint layout and abstract.
- `sections/`: questions, method, setup, experiment records, implementation notes,
  open questions, reproducibility appendix, and change history.
- `assets/`: unchanged supplied training figures; their working evidence records
  live under `docs/experiments/results/`.
- `data/`: frozen numeric transcriptions for the current throughput comparison
  and cache pilot. Missing native experiment metadata remains explicitly missing.
- `report.json`: version, evidence cutoff, implementation revision, output name,
  and manual update policy.
- `template/`: unchanged source files copied from the provided sibling checkout's
  template. `SOURCE.json` records their original hashes. The conference example
  remains an example, not our report content.
- `.build/`: ignored temporary compilation files and generated table fragments.
- `reports/`: dated PDFs, build manifests, source snapshots, and available patches.

Each experiment uses a stable ID, question, setup, status, evidence, measurements,
record notes, and missing evidence. The current four-model comparison contains
no earlier generator benchmark rows. Historical training ablations have their
own report entries and Markdown records. Batching is implemented; GPU performance measurements remain pending.

## Build manually

Install Tectonic or a LaTeX distribution providing `pdflatex` and the packages
used in `main.tex`, then run from the repository root:

```bash
python3 reporting/build.py
```

Use `--compiler /path/to/tectonic` to select a compiler explicitly, or
`--prepare-only` to inspect generated table inputs without compiling. Tectonic
may download TeX support files on its initial invocation. Compilation runs in
`.build/` with copies of the unchanged template styles.

The numeric table cells are generated from `data/*.json`, rather than being
maintained independently in LaTeX. The build produces a PDF plus a manifest of
source hashes, compiler/command information, a source archive, and the available
working-tree patch. This report-level provenance does not replace missing
training/evaluation/serving provenance for the experiments themselves.

## Extending a commissioned report

1. Read the associated Markdown records and freeze newly supplied measurements
   into `data/`, retaining their original source and missing-evidence fields.
2. Extend the relevant experiment ID; create a new ID when the protocol changes.
   Separate measured evidence from implementation and planned experiments.
3. Update `report.json` and the change history. Increment the version/output name
   when publishing a new snapshot so earlier PDFs and manifests remain available.
4. Build, check the PDF and numeric tables, and include available command files,
   configuration snapshots, model/checkpoint hashes, and source patches.

Editing or rebuilding the report never advances its evidence cutoff by itself.
