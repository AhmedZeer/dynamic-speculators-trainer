"""Read bank activations without retaining Drive-backed memory mappings."""

from pathlib import Path

from hs_connectors.transfer import wait_for_lock
from safetensors.torch import load

from hs_connectors import FileTransfer
from speculators.bank.progress import stage_progress


class BankFileTransfer(FileTransfer):
    def get_cached(self, file_idx):
        path = self.hidden_states_path / f"hs_{file_idx}.safetensors"
        # Normal reads are quiet; a blocked read identifies its row and path.
        with stage_progress(
            f"Reading hidden-state row={file_idx}, file={path}", quiet=True
        ):
            try:
                lock_path = str(path) + ".lock"
                if Path(lock_path).exists():
                    wait_for_lock(lock_path)
                if not path.exists():
                    return None
                # Normal reads own the payload, so a later mmap page fault on
                # Drive cannot terminate this process while consuming tensors.
                return load(path.read_bytes())
            except Exception as exc:
                exc.add_note(f"Hidden-state file: {path}")
                raise
