"""Explicit, checksum-guarded application of the vLLM Python-source patch."""

import importlib.metadata
import importlib.util
import json
import shutil
from pathlib import Path

from speculators.bank.artifacts import file_digest, write_json
from speculators.generator.serving.bundle import VLLM_VERSION
from speculators.provenance import atomic_write

PATCH_DIR = Path(__file__).parent / "patches"
BACKUP_NAME = ".speculators-generator-vllm-0.31.0"


def package_root():
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        raise ImportError("Install vLLM 0.31.0 before applying its adapter patch")
    return Path(spec.origin).resolve().parent.parent


def patch_sources(action="check", root=None, *, version=None):  # noqa: C901 -- transactional installer
    version = version or importlib.metadata.version("vllm")
    if version != VLLM_VERSION:
        raise ValueError(f"Patch supports only vLLM {VLLM_VERSION}, found {version}")
    if action not in ("check", "apply", "revert"):
        raise ValueError("Expected check, apply, or revert")
    root = Path(root) if root is not None else package_root()
    manifest = json.loads((PATCH_DIR / "manifest.json").read_text())
    status = {}
    for relative, record in manifest["files"].items():
        actual = file_digest(root / relative)
        if actual == record["before_sha256"]:
            status[relative] = "original"
        elif actual == record["after_sha256"]:
            status[relative] = "patched"
        elif actual in record.get("previous_sha256", []):
            status[relative] = "outdated"
        else:
            raise ValueError(
                f"Unrecognized vLLM source; refusing to overwrite {relative}"
            )
    states = set(status.values())
    if len(states) != 1 and not states <= {"patched", "outdated"}:
        raise ValueError(
            "Partially patched vLLM installation; repair it before proceeding"
        )
    state = "outdated" if "outdated" in states else next(iter(states))
    backup = root / BACKUP_NAME
    if action == "check":
        return {"version": version, "state": state, "root": str(root)}
    if action == "revert":
        if state == "original":
            return {"version": version, "state": state, "root": str(root)}
        for relative, record in manifest["files"].items():
            if file_digest(backup / relative) != record["before_sha256"]:
                raise ValueError(f"Missing or changed source backup: {relative}")
        for relative in manifest["files"]:
            shutil.copy2(backup / relative, root / relative)
        shutil.rmtree(backup)
        return {"version": version, "state": "original", "root": str(root)}
    if state == "patched":
        return {"version": version, "state": state, "root": str(root)}
    upgrading = state == "outdated"
    if upgrading:
        for relative, record in manifest["files"].items():
            if file_digest(backup / relative) != record["before_sha256"]:
                raise ValueError(f"Missing or changed source backup: {relative}")
    elif backup.exists():
        raise FileExistsError(f"Existing source backup needs review: {backup}")
    replacements = {}
    for relative, record in manifest["files"].items():
        text = ((backup if upgrading else root) / relative).read_text()
        for old, new in record["replacements"]:
            if text.count(old) != 1:
                raise ValueError(f"Expected a unique patch anchor: {relative}")
            text = text.replace(old, new, 1)
        replacements[relative] = text
    installed = {relative: (root / relative).read_text() for relative in replacements}
    modes = {relative: (root / relative).stat().st_mode for relative in replacements}
    if not upgrading:
        backup.mkdir()
    try:
        if not upgrading:
            for relative in replacements:
                saved = backup / relative
                saved.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(root / relative, saved)
        for relative, text in replacements.items():
            atomic_write(root / relative, text)
            (root / relative).chmod(modes[relative])
            if (
                file_digest(root / relative)
                != manifest["files"][relative]["after_sha256"]
            ):
                raise ValueError(f"Patched checksum mismatch: {relative}")
        write_json(backup / "manifest.json", manifest)
    except BaseException:
        for relative, text in installed.items():
            atomic_write(root / relative, text)
            (root / relative).chmod(modes[relative])
        if not upgrading:
            shutil.rmtree(backup)
        raise
    return {"version": version, "state": "patched", "root": str(root)}
