"""Launch view with verified original vLLM sources, without mutating site-packages."""

import json
import shutil
from pathlib import Path

from speculators.bank.artifacts import file_digest, write_json
from speculators.generator.serving.patch import (
    BACKUP_NAME,
    PATCH_DIR,
    package_root,
    patch_sources,
)


def vanilla_source_view(output, *, root=None):
    root = Path(root) if root is not None else package_root()
    status = patch_sources(root=root)
    manifest = json.loads((PATCH_DIR / "manifest.json").read_text())
    originals = root if status["state"] == "original" else root / BACKUP_NAME
    for relative, record in manifest["files"].items():
        if file_digest(originals / relative) != record["before_sha256"]:
            raise ValueError(f"Vanilla baseline requires verified original: {relative}")
    view = Path(output).resolve() / "vanilla-vllm"
    view.mkdir(parents=True, exist_ok=True)
    replaced = {Path(name) for name in manifest["files"]}

    def materialize(relative):
        source, destination = root / relative, view / relative
        if relative in replaced:
            if destination.is_symlink():
                destination.unlink()
            shutil.copy2(originals / relative, destination)
        elif any(name.is_relative_to(relative) for name in replaced):
            destination.mkdir(exist_ok=True)
            for child in source.iterdir():
                if child.name != "__pycache__":
                    materialize(relative / child.name)
        elif not destination.exists():
            destination.symlink_to(source, target_is_directory=source.is_dir())

    materialize(Path("vllm"))
    hashes = {str(name): file_digest(view / name) for name in replaced}
    write_json(
        Path(output) / "vanilla_source.json",
        {
            "version": manifest["version"],
            "commit": manifest["commit"],
            "source_root": str(root),
            "launch_root": str(view),
            "original_sha256": hashes,
        },
    )
    return view
