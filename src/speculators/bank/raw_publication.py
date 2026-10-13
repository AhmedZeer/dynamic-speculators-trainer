"""Metadata-only staging; original activation payloads remain untouched."""

from speculators.bank.artifacts import digest, file_digest, write_json
from speculators.bank.progress import stage_progress


def stage_raw(cfg, manifest, data, included, raw, packaged, skipped):
    from speculators.bank.hub import profile  # noqa: PLC0415

    packages, rows, objects = [], {}, []
    size = 0

    def seal():
        nonlocal size
        if not rows:
            return
        identity = digest(sorted(rows))
        directory = packaged / identity
        directory.mkdir(exist_ok=True)
        public = {
            "version": 3,
            "prepared_fingerprint": manifest["prepared_fingerprint"],
            "rows": rows.copy(),
        }
        write_json(directory / "complete.json", public)
        record = {
            **public,
            "objects": [
                *objects,
                [
                    f"{profile(cfg)}/raw-groups/{identity}-complete.json",
                    "complete.json",
                ],
            ],
        }
        write_json(directory / "local.json", record)
        packages.append((directory, record))
        rows.clear()
        objects.clear()
        size = 0

    with stage_progress("Preparing raw-file upload metadata", interval=30):
        for i in included:
            if str(i) in skipped:
                continue
            source = raw / f"hs_{i}.safetensors"
            if not source.exists():
                continue
            item = data[i]
            path = (
                f"{profile(cfg)}/{item['subset_id']}/raw/"
                f"{item['primary_id']}.safetensors"
            )
            byte_count = source.stat().st_size
            rows[str(i)] = {
                "primary_id": item["primary_id"],
                "format": "raw",
                "prompt_length": item["prompt_length"],
                "shard": {
                    "repo": cfg.hub.activations_repo,
                    "path": path,
                    "revision": "__commit__",
                    "sha256": file_digest(source),
                    "bytes": byte_count,
                },
            }
            objects.append([path, str(source.resolve())])
            size += byte_count
            if (
                len(rows)
                >= min(cfg.hub.upload_batch_examples, cfg.hub.upload_max_files)
                or size >= cfg.hub.upload_batch_bytes
            ):
                seal()
        seal()
    return packages
