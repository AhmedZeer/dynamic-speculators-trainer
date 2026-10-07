"""Read bank activations without retaining Drive-backed memory mappings."""

import json
import struct
from pathlib import Path

from hs_connectors.transfer import wait_for_lock
from safetensors.torch import load

from hs_connectors import FileTransfer
from speculators.bank.progress import stage_progress


class BankFileTransfer(FileTransfer):
    def get_prompt_states(self, file_idx, prompt_length):
        """Buffered prefix read: avoid response tensors and Drive-backed mmap."""
        path = self.hidden_states_path / f"hs_{file_idx}.safetensors"
        with stage_progress(f"Reading prompt row={file_idx}, file={path}", quiet=True):
            try:
                lock_path = str(path) + ".lock"
                if Path(lock_path).exists():
                    wait_for_lock(lock_path)
                if not path.exists():
                    return None
                with path.open("rb") as handle:
                    header_size = struct.unpack("<Q", handle.read(8))[0]
                    header = json.loads(handle.read(header_size))
                    tensor = header["hidden_states"]
                    shape = tensor["shape"]
                    if not shape or not 1 <= prompt_length <= shape[0]:
                        raise ValueError("Invalid prompt boundary")
                    start, stop = tensor["data_offsets"]
                    length = (stop - start) // shape[0] * prompt_length
                    handle.seek(8 + header_size + start)
                    data = handle.read(length)
                    if len(data) != length:
                        raise ValueError("Truncated prompt hidden states")
                # Let safetensors decode/validate the compact tensor, retaining
                # all supported dtypes without implementing a second decoder.
                compact = json.dumps(
                    {
                        "hidden_states": {
                            "dtype": tensor["dtype"],
                            "shape": [prompt_length, *shape[1:]],
                            "data_offsets": [0, length],
                        }
                    }
                ).encode()
                compact += b" " * (-len(compact) % 8)
                return load(struct.pack("<Q", len(compact)) + compact + data)[
                    "hidden_states"
                ]
            except Exception as exc:
                exc.add_note(f"Hidden-state file: {path}")
                raise

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
