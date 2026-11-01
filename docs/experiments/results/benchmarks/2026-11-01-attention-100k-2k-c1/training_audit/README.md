# Follow-up training provenance audit

Published step-2000 configs, commands, bundle metadata and available patches/
checksums were retrieved on 2026-11-01. `audit.json` lists immutable URLs and
SHA-256 hashes for every fetched source, including large timing/membership JSONs
that remain remote to avoid duplicating multi-MB records. The small command,
configuration, result, bundle and patch/checksum artifacts are archived here.

Unique-example counts are deterministic schedule reconstructions and agree with
logged presentation counts. Dataset IDs, model revisions, learning-rate values,
and target-weight shard checksums were checked. This is training/extraction
provenance; it does not establish the runtime used by the supplied serving
benchmarks. No GPU experiment was executed during this audit.
